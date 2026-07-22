"""
Standalone collector: polls every source on a fixed interval and stores
what they report in MariaDB. The normal run loop also archives metering
rows from before the current calendar month, once now and then again each
time the 1st of the month arrives - see Collector.archive_old_data().

Run:
    python Collector.py              # loop forever, one cycle every INTERVAL_SECONDS
    python Collector.py --once       # single cycle, then exit (no archiving)
    python Collector.py --dry-run    # collect and log, write nothing (no archiving)
    python Collector.py --archive-data    # archive rows before this month, then exit

Configuration comes from the environment or a .env file - see
collector.env.example. Install as a service with weather-collector.service.
"""

import argparse
import asyncio
import logging
import signal
import time
from datetime import datetime

import pymysql

from Config import Config
from HubitatSource import HubitatSource
from Reading import Reading
from Source import Source
from WeatherDb import WeatherDb
from WeatherMcpSource import WeatherMcpSource

logger = logging.getLogger("collector")


class Collector:
    """
    Usage:
        collector = Collector(Config.from_env())
        await collector.run()
    """

    def __init__(self, config: Config, dry_run: bool = False):
        self._config = config
        self._dry_run = dry_run
        self._sources: list[Source] = [HubitatSource(config), WeatherMcpSource(config)]
        self._db = WeatherDb(config)
        self._stopping = asyncio.Event()
        # None means "never checked yet" - archive_old_data() runs on the
        # very first tick to catch up on any backlog, then again only once
        # the 1st of the next month arrives (see _maybe_archive()).
        self._next_archive_at: datetime | None = None

    async def run(self) -> None:
        """Collect now, then once per interval until stopped. Also checks
        whether metering should be archived on that same cadence - not just
        once at startup, so a new calendar month gets swept in on its own
        without restarting the process."""
        logger.info(
            "Collecting from %s every %ss%s",
            ", ".join(s.name for s in self._sources),
            self._config.interval_seconds,
            " (dry run)" if self._dry_run else "",
        )
        await self.collect_once()
        await self._maybe_archive()
        while not await self._sleep_to_next_tick():
            await self.collect_once()
            await self._maybe_archive()
        logger.info("Stopped")

    async def collect_once(self) -> int:
        """One full cycle. Never raises - a failure now just means fewer
        rows this interval, and the next cycle tries again from scratch."""
        taken_at = self._tick_timestamp()
        readings = await self._gather(taken_at)
        if not readings:
            logger.warning("No readings collected at %s", taken_at)
            return 0

        if self._dry_run:
            statements = self._db.preview(readings)
            logger.info("Dry run - not writing, would run %d statements:", len(statements))
            for statement in statements:
                logger.info("  %s;", statement)
            return 0

        try:
            saved = await asyncio.to_thread(self._db.save, readings)
        except pymysql.MySQLError as e:
            logger.error("Could not store %d readings: %s", len(readings), e)
            return 0

        logger.info("Stored %d of %d readings at %s", saved, len(readings), taken_at)
        return saved

    async def _gather(self, taken_at: datetime) -> list[Reading]:
        """Poll all sources concurrently; a source that fails costs only
        its own readings."""
        results = await asyncio.gather(
            *(source.collect(taken_at) for source in self._sources),
            return_exceptions=True,
        )

        readings = []
        for source, result in zip(self._sources, results):
            if isinstance(result, BaseException):
                logger.error("Source %s failed: %s", source.name, result)
                continue
            logger.info("Source %s returned %d readings", source.name, len(result))
            readings.extend(result)
        return readings

    def _tick_timestamp(self) -> datetime:
        """Floor the current time to the interval boundary, so every cycle
        writes one row per sensor/metric at a predictable timestamp and a
        re-run of the same interval is a no-op against metering_uq."""
        now = int(time.time())
        return datetime.fromtimestamp(now - now % self._config.interval_seconds)

    async def _sleep_to_next_tick(self) -> bool:
        """Wait for the next interval boundary. True if we were asked to
        stop while waiting."""
        delay = self._config.interval_seconds - (
            time.time() % self._config.interval_seconds
        )
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            return True
        except asyncio.TimeoutError:
            return False

    async def archive_metering(self, **kwargs) -> dict:
        """Archive old raw metering rows via WeatherDb.archive_metering() -
        see its docstring for the retention-spec kwargs (hours/months/
        older_than) and delete_after_archive."""
        return await asyncio.to_thread(self._db.archive_metering, **kwargs)

    async def archive_old_data(self) -> dict:
        """Archive every metering row from before the start of the current
        calendar month, keeping only the current month's data un-archived."""
        return await self.archive_metering(older_than=self._start_of_month(datetime.now()))

    async def _maybe_archive(self) -> None:
        """Run archive_old_data() once now if it has never run, then again
        only once the 1st of the next month arrives - nothing becomes newly
        archivable in between, since the cutoff is always "start of this
        month". Never raises - a failed attempt just means the next check
        (still the same scheduled date) tries again."""
        if self._dry_run:
            return
        now = datetime.now()
        if self._next_archive_at is not None and now < self._next_archive_at:
            return

        try:
            result = await self.archive_old_data()
            logger.info("Archived metering before this month: %s", result)
        except pymysql.MySQLError as e:
            logger.error("Could not archive metering: %s", e)

        self._next_archive_at = self._first_of_next_month(now)

    @staticmethod
    def _start_of_month(when: datetime) -> datetime:
        return datetime(when.year, when.month, 1)

    @staticmethod
    def _first_of_next_month(when: datetime) -> datetime:
        return datetime(when.year + (when.month == 12), when.month % 12 + 1, 1)

    def stop(self) -> None:
        logger.info("Shutdown requested, finishing current cycle")
        self._stopping.set()

    def close(self) -> None:
        self._db.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sensor data into MariaDB")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="collect but do not write")
    parser.add_argument(
        "--archive-data", action="store_true",
        help="archive metering rows older than the first day of the current month, then exit "
             "(does not collect)",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if config.log_file == Config.log_file:  # LOG_FILE not overridden via env
        config.log_file = "logs/collector.log"
    config.configure_logging()
    collector = Collector(config, dry_run=args.dry_run or config.dry_run)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, collector.stop)

    try:
        if args.archive_data:
            result = await collector.archive_old_data()
            logger.info("Archived metering: %s", result)
        elif args.once:
            await collector.collect_once()
        else:
            await collector.run()
    finally:
        collector.close()


if __name__ == "__main__":
    asyncio.run(main())
