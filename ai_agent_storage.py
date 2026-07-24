"""
Storage layer the AI agent reasons over.

MetricStorage wraps the metering data collected by Collector.py (see
db/weather.sql for the schema) behind three read-only functions, so the
rest of the agent never writes SQL directly:

    get_current([locations], [metrics])       -> latest value per location/metric
    get_stats(metric, period, [locations])    -> min/max/avg/count per location
    get_stats_last_hours(metric, hours, [locations])   -> get_stats, last N hours
    get_stats_last_days(metric, days, [locations])     -> get_stats, last N days
    get_history(metric, start, end, [locations]) -> raw readings per location
    get_history_last_hours(metric, hours, [locations]) -> get_history, last N hours
    get_history_last_days(metric, days, [locations])   -> get_history, last N days

format_current/format_stats/format_history render those results as compact
text, meant to be dropped straight into an LLM prompt.

outside_locations/inside_locations are location names split by the
location.outside flag, loaded lazily and cached; refresh_locations()
reloads them from the `location` table.

pymysql is blocking, same as WeatherDb.py - call these through
asyncio.to_thread from async code. One MetricStorage is safe to share
between those threads: a pymysql connection carries one request at a time
and interleaving two corrupts the wire protocol, so queries are
serialised internally.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

import pymysql

from Config import Config

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


def _fmt_dt(taken_at: datetime) -> str:
    return taken_at.strftime("%Y-%m-%d %H:%M:%S")


def format_current(result: dict[str, dict[str, MetricValue]]) -> str:
    """Render get_current()'s result as compact text for an LLM prompt."""
    if not result:
        return "No current data."
    lines = []
    for location, metrics in result.items():
        readings = ", ".join(
            f"{metric}={value.value:g} ({_fmt_dt(value.taken_at)})"
            for metric, value in metrics.items()
        )
        lines.append(f"{location}: {readings}")
    return "\n".join(lines)


def format_stats(result: dict[str, MetricStats]) -> str:
    """Render get_stats()'s result as compact text for an LLM prompt."""
    if not result:
        return "No stats data."
    lines = [
        f"{location}: {stats.metric} min={stats.min:g} max={stats.max:g} "
        f"avg={stats.avg:.2f} (n={stats.count})"
        for location, stats in result.items()
    ]
    return "\n".join(lines)


def format_history(result: dict[str, list[HistoryPoint]]) -> str:
    """Render get_history()'s result as compact text for an LLM prompt."""
    if not result:
        return "No history data."
    lines = []
    for location, points in result.items():
        if not points:
            continue
        readings = ", ".join(f"{_fmt_dt(p.taken_at)}={p.value:g}" for p in points)
        lines.append(f"{location}: {readings}")
    return "\n".join(lines)


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
        self._outside_locations: list[str] | None = None
        self._inside_locations: list[str] | None = None
        # Guards the single connection. Callers reach this from several
        # threads at once (ai_agent_server.py runs each request in one), and
        # two queries sharing a pymysql connection interleave their packets
        # and leave the parser reading another query's bytes.
        self._lock = threading.Lock()

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
            user=self._config.db_user_readonly,
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

    @property
    def outside_locations(self) -> list[str]:
        """Location names with outside=1, loaded lazily and cached.
        Call refresh_locations() to reload."""
        if self._outside_locations is None:
            self.refresh_locations()
        return list(self._outside_locations)

    @property
    def inside_locations(self) -> list[str]:
        """Location names with outside=0/NULL, loaded lazily and cached.
        Call refresh_locations() to reload."""
        if self._inside_locations is None:
            self.refresh_locations()
        return list(self._inside_locations)

    def refresh_locations(self) -> None:
        """Reload outside_locations/inside_locations from the `location`
        table - call this if locations may have been added/changed since
        the lists were last loaded."""
        outside: list[str] = []
        inside: list[str] = []
        for row in self._query("SELECT location, outside FROM location", ()):
            (outside if row["outside"] else inside).append(row["location"])
        self._outside_locations = outside
        self._inside_locations = inside

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

    def get_stats_last_hours(
        self, metric: str, hours: int, locations: list[str] | None = None
    ) -> dict[str, MetricStats]:
        """get_stats over the last `hours`, ending now."""
        return self.get_stats(metric, timedelta(hours=hours), locations)

    def get_stats_last_days(
        self, metric: str, days: int, locations: list[str] | None = None
    ) -> dict[str, MetricStats]:
        """get_stats over the last `days`, ending now."""
        return self.get_stats(metric, timedelta(days=days), locations)

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

    def get_history_last_hours(
        self, metric: str, hours: int, locations: list[str] | None = None
    ) -> dict[str, list[HistoryPoint]]:
        """get_history over the last `hours`, ending now."""
        end = datetime.now()
        return self.get_history(metric, end - timedelta(hours=hours), end, locations)

    def get_history_last_days(
        self, metric: str, days: int, locations: list[str] | None = None
    ) -> dict[str, list[HistoryPoint]]:
        """get_history over the last `days`, ending now."""
        end = datetime.now()
        return self.get_history(metric, end - timedelta(days=days), end, locations)

    def _query(self, query: str, args: tuple) -> list[dict]:
        # Held across execute and fetch, not just execute: the result set is
        # read off the same socket, so releasing early would let another
        # thread's query overwrite the bytes this one is still reading.
        with self._lock:
            connection = self._connect()
            with connection.cursor() as cursor:
                cursor.execute(query, args)
                return cursor.fetchall()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
