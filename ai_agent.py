"""
AI agent - starts with the storage layer it reasons over.

MetricStorage wraps the metering data collected by Collector.py (see
db/weather.sql for the schema) behind three read-only functions, so the
rest of the agent never writes SQL directly:

    get_current([locations], [metrics])       -> latest value per location/metric
    get_stats(metric, period, [locations])    -> min/max/avg/count per location
    get_history(metric, start, end, [locations]) -> raw readings per location

pymysql is blocking, same as WeatherDb.py - call these through
asyncio.to_thread from async code.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pymysql

from Config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-agent")


@dataclass(frozen=True)
class MetricValue:
    """One metric's most recent reading at a location."""

    metric: str
    value: float
    taken_at: datetime
    sensor_name: str


@dataclass(frozen=True)
class MetricStats:
    """Aggregate of one metric at a location over a time window."""

    metric: str
    min: float
    max: float
    avg: float
    count: int


@dataclass(frozen=True)
class HistoryPoint:
    """One raw reading, as returned by get_history."""

    taken_at: datetime
    value: float


class MetricStorage:
    """
    Read-only access to the `weather` schema.

    Usage:
        storage = MetricStorage(config)
        storage.get_current(["Weather station"])
        storage.close()
    """

    def __init__(self, config: Config):
        self._config = config
        self._connection: pymysql.Connection | None = None

    def _connect(self) -> pymysql.Connection:
        """Live connection, reconnecting if the server dropped us."""
        if self._connection is not None:
            try:
                self._connection.ping(reconnect=True)
                return self._connection
            except pymysql.MySQLError as e:
                logger.warning("Lost connection to MariaDB, reconnecting: %s", e)
                self._connection = None

        self._connection = pymysql.connect(
            host=self._config.db_host,
            port=self._config.db_port,
            user=self._config.db_user_reasonly,
            password=self._config.db_password_readonly,
            database=self._config.db_name,
            charset="utf8mb4",
            connect_timeout=10,
            cursorclass=pymysql.cursors.DictCursor,
        )
        logger.info("Connected to MariaDB %s:%s/%s",
                    self._config.db_host, self._config.db_port, self._config.db_name)
        return self._connection

    def get_current(
        self, locations: list[str] | None = None, metrics: list[str] | None = None
    ) -> dict[str, dict[str, MetricValue]]:
        """Latest value of every metric reported at each location, keyed by
        location then metric. Defaults to every location; pass `locations`
        or `metrics` to narrow it. If several sensors at a location report
        the same metric, the most recent wins."""
        query = (
            "SELECT l.location AS location, m.metric AS metric, me.value AS value, "
            "me.mdatatime AS taken_at, s.name AS sensor_name "
            "FROM metering me "
            "JOIN metric m ON m.metricid = me.metric_metricid "
            "JOIN sensor s ON s.sensorid = me.sensor_sensorid "
            "JOIN location l ON l.locid = s.location_locid "
            "WHERE me.mdatatime = ("
            "  SELECT MAX(me2.mdatatime) FROM metering me2 "
            "  JOIN sensor s2 ON s2.sensorid = me2.sensor_sensorid "
            "  WHERE s2.location_locid = s.location_locid "
            "  AND me2.metric_metricid = me.metric_metricid"
            ")"
        )
        args: list = []
        if locations:
            placeholders = ", ".join(["%s"] * len(locations))
            query += f" AND l.location IN ({placeholders})"
            args.extend(locations)
        if metrics:
            placeholders = ", ".join(["%s"] * len(metrics))
            query += f" AND m.metric IN ({placeholders})"
            args.extend(metrics)

        current: dict[str, dict[str, MetricValue]] = {}
        for row in self._query(query, tuple(args)):
            by_metric = current.setdefault(row["location"], {})
            metric = row["metric"]
            # A tie between two sensors at the same instant is unlikely but
            # possible - keep whichever row is seen first.
            if metric not in by_metric:
                by_metric[metric] = MetricValue(
                    metric=metric,
                    value=row["value"],
                    taken_at=row["taken_at"],
                    sensor_name=row["sensor_name"],
                )
        return current

    def get_stats(
        self, metric: str, period: timedelta, locations: list[str] | None = None
    ) -> dict[str, MetricStats]:
        """Min/max/avg/count of `metric` over the last `period`, ending now,
        keyed by location. Defaults to every location with data in that
        window; pass `locations` to narrow it. A location with no readings
        in the window is simply absent from the result."""
        query = (
            "SELECT l.location AS location, MIN(me.value) AS min, MAX(me.value) AS max, "
            "AVG(me.value) AS avg, COUNT(*) AS count "
            "FROM metering me "
            "JOIN metric m ON m.metricid = me.metric_metricid "
            "JOIN sensor s ON s.sensorid = me.sensor_sensorid "
            "JOIN location l ON l.locid = s.location_locid "
            "WHERE m.metric = %s AND me.mdatatime >= %s"
        )
        since = datetime.now() - period
        args: list = [metric, since]
        if locations:
            placeholders = ", ".join(["%s"] * len(locations))
            query += f" AND l.location IN ({placeholders})"
            args.extend(locations)
        query += " GROUP BY l.location"

        stats: dict[str, MetricStats] = {}
        for row in self._query(query, tuple(args)):
            if not row["count"]:
                continue
            stats[row["location"]] = MetricStats(
                metric=metric,
                min=row["min"],
                max=row["max"],
                avg=row["avg"],
                count=row["count"],
            )
        return stats

    def get_history(
        self,
        metric: str,
        start: datetime,
        end: datetime,
        locations: list[str] | None = None,
    ) -> dict[str, list[HistoryPoint]]:
        """Every reading of `metric` between start and end (inclusive),
        oldest first, keyed by location. Defaults to every location with
        data in that window; pass `locations` to narrow it."""
        query = (
            "SELECT l.location AS location, me.mdatatime AS taken_at, me.value AS value "
            "FROM metering me "
            "JOIN metric m ON m.metricid = me.metric_metricid "
            "JOIN sensor s ON s.sensorid = me.sensor_sensorid "
            "JOIN location l ON l.locid = s.location_locid "
            "WHERE m.metric = %s AND me.mdatatime BETWEEN %s AND %s"
        )
        args: list = [metric, start, end]
        if locations:
            placeholders = ", ".join(["%s"] * len(locations))
            query += f" AND l.location IN ({placeholders})"
            args.extend(locations)
        query += " ORDER BY l.location ASC, me.mdatatime ASC"

        history: dict[str, list[HistoryPoint]] = {}
        for row in self._query(query, tuple(args)):
            history.setdefault(row["location"], []).append(
                HistoryPoint(taken_at=row["taken_at"], value=row["value"])
            )
        return history

    def _query(self, query: str, args: tuple) -> list[dict]:
        connection = self._connect()
        with connection.cursor() as cursor:
            cursor.execute(query, args)
            return cursor.fetchall()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def demo():
    config = Config.from_env()
    storage = MetricStorage(config)
    try:
        location = config.weather_location_name
        print("-- get_current() (all locations) --")
        for loc, metrics in storage.get_current().items():
            print(f"  {loc}:")
            for metric, reading in metrics.items():
                print(f"    {metric}: {reading.value} ({reading.taken_at}, {reading.sensor_name})")

        print("-- get_stats('temperature', 24h) (all locations) --")
        for loc, stats in storage.get_stats("temperature", timedelta(hours=24)).items():
            print(f"  {loc}: {stats}")

        print(f"-- get_history('temperature', last 1h, locations=[{location!r}]) --")
        end = datetime.now()
        start = end - timedelta(hours=1)
        history = storage.get_history("temperature", start, end, locations=[location])
        for point in history.get(location, []):
            print(f"  {point.taken_at}: {point.value}")
    finally:
        storage.close()


if __name__ == "__main__":
    demo()
