"""
Standalone collector for ClickHouse - the ClickHouse counterpart of
Collector.py. Two cadences run concurrently:

  * a weather loop, every INTERVAL_SECONDS (default 600s), that polls every
    source and stores everything except the 'power' metric into
    metering_weather;
  * a power loop, every POWER_INTERVAL_SECONDS (default 10s), that polls only
    the power-capable sources and stores the 'power' metric into
    metering_power.

Both write through one WeatherDbClickhouse, serialized by a lock so the two
loops never insert on the shared client at the same time. There is no
archiving loop as in Collector.py - ClickHouse's materialized views and TTLs
keep the rollups and retention current on their own.

Run:
    python CollectorClickhouse.py            # both loops until stopped
    python CollectorClickhouse.py --once     # one weather + one power cycle, then exit
    python CollectorClickhouse.py --dry-run  # collect and log, write nothing

Configuration comes from the environment or a .env file - see
collector.env.example (CH_* and POWER_INTERVAL_SECONDS). Install as a service
with weather-collector-clickhouse.service.
"""

import argparse
import asyncio
import logging
import signal
import time
from datetime import datetime

from clickhouse_connect.driver.exceptions import ClickHouseError

from Config import Config
from HubitatSource import HubitatSource
from Reading import Reading
from Source import Source
from WeatherDbClickhouse import WeatherDbClickhouse
from WeatherMcpSource import WeatherMcpSource

logger = logging.getLogger("collector-clickhouse")


class CollectorClickhouse:
    """
    Usage:
        collector = CollectorClickhouse(Config.from_env())
        await collector.run()
    """

    def __init__(self, config: Config, dry_run: bool = False):
        self._config = config
        self._dry_run = dry_run
        self._sources: list[Source] = [HubitatSource(config), WeatherMcpSource(config)]
        # Only Hubitat reports power, so the fast loop polls it alone rather
        # than waking the weather station's MCP server every few seconds.
        self._power_sources: list[Source] = [
            s for s in self._sources if isinstance(s, HubitatSource)
        ]
        self._db = WeatherDbClickhouse(config)
        # One shared client, so both loops' inserts are serialized onto it.
        self._db_lock = asyncio.Lock()
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        """Run the weather and power loops concurrently until stopped."""
        logger.info(
            "Collecting weather from %s every %ss and power from %s every %ss%s",
            ", ".join(s.name for s in self._sources),
            self._config.interval_seconds,
            ", ".join(s.name for s in self._power_sources),
            self._config.power_interval_seconds,
            " (dry run)" if self._dry_run else "",
        )
        await asyncio.gather(
            self._loop(self._weather_cycle, self._config.interval_seconds),
            self._loop(self._power_cycle, self._config.power_interval_seconds),
        )
        logger.info("Stopped")

    async def collect_once(self) -> None:
        """One weather cycle and one power cycle, for --once."""
        await self._weather_cycle()
        await self._power_cycle()

    async def _loop(self, cycle, interval: int) -> None:
        """Run `cycle` now, then once every `interval` seconds until stopped."""
        await cycle()
        while not await self._sleep_to_next_tick(interval):
            await cycle()

    async def _weather_cycle(self) -> int:
        """Poll every source and store all non-power readings."""
        taken_at = self._tick_timestamp(self._config.interval_seconds)
        readings = [r for r in await self._gather(self._sources, taken_at) if not r.is_power]
        return await self._store(readings, taken_at, "weather")

    async def _power_cycle(self) -> int:
        """Poll the power sources and store only the power readings."""
        taken_at = self._tick_timestamp(self._config.power_interval_seconds)
        readings = [r for r in await self._gather(self._power_sources, taken_at) if r.is_power]
        return await self._store(readings, taken_at, "power")

    async def _store(self, readings: list[Reading], taken_at: datetime, label: str) -> int:
        """Write `readings`, or log them in dry-run. Never raises - a failure
        now just means fewer rows this cycle, and the next one tries again."""
        if not readings:
            logger.warning("No %s readings collected at %s", label, taken_at)
            return 0

        if self._dry_run:
            statements = self._db.preview(readings)
            logger.info("Dry run (%s) - not writing, would run %d statements:", label, len(statements))
            for statement in statements:
                logger.info("  %s;", statement)
            return 0

        try:
            async with self._db_lock:
                saved = await asyncio.to_thread(self._db.save, readings)
        except ClickHouseError as e:
            logger.error("Could not store %d %s readings: %s", len(readings), label, e)
            return 0

        logger.info("Stored %d %s reading(s) at %s", saved, label, taken_at)
        return saved

    async def _gather(self, sources: list[Source], taken_at: datetime) -> list[Reading]:
        """Poll `sources` concurrently; a source that fails costs only its
        own readings."""
        results = await asyncio.gather(
            *(source.collect(taken_at) for source in sources),
            return_exceptions=True,
        )

        readings: list[Reading] = []
        for source, result in zip(sources, results):
            if isinstance(result, BaseException):
                logger.error("Source %s failed: %s", source.name, result)
                continue
            logger.info("Source %s returned %d readings", source.name, len(result))
            readings.extend(result)
        return readings

    def _tick_timestamp(self, interval: int) -> datetime:
        """Floor the current time to the interval boundary, so each cycle
        writes one row per sensor/metric at a predictable timestamp."""
        now = int(time.time())
        return datetime.fromtimestamp(now - now % interval)

    async def _sleep_to_next_tick(self, interval: int) -> bool:
        """Wait for the next interval boundary. True if we were asked to stop
        while waiting."""
        delay = interval - (time.time() % interval)
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            return True
        except asyncio.TimeoutError:
            return False

    def stop(self) -> None:
        logger.info("Shutdown requested, finishing current cycle")
        self._stopping.set()

    def close(self) -> None:
        self._db.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sensor data into ClickHouse")
    parser.add_argument("--once", action="store_true", help="run one weather + one power cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="collect but do not write")
    args = parser.parse_args()

    config = Config.from_env()
    if config.log_file == Config.log_file:  # LOG_FILE not overridden via env
        config.log_file = "logs/collector_clickhouse.log"
    config.configure_logging()
    collector = CollectorClickhouse(config, dry_run=args.dry_run or config.dry_run)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, collector.stop)

    try:
        if args.once:
            await collector.collect_once()
        else:
            await collector.run()
    finally:
        collector.close()


if __name__ == "__main__":
    asyncio.run(main())
