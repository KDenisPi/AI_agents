"""
ClickHouse read layer for the AI agent - the ClickHouse counterpart of
ai_agent_storage.MetricStorage, and a drop-in for it: same public methods,
same return shapes, same MetricValue/MetricStats/HistoryPoint results (those
dataclasses and the format_* renderers are DB-agnostic, so they are imported
from ai_agent_storage rather than duplicated).

Two schema differences from the MariaDB layer shape the queries (see
db/clickhouse/weather_clickhouse.sql):

  * There is no single `metering` table. Readings are split by metric into
    metering_weather (everything except 'power') and metering_power ('power').
    Each query is routed to the table(s) its metrics live in and the results
    merged - a metric belongs to exactly one table, so the merge never
    conflicts.

  * These queries read the *raw* fact tables only, the same way the MariaDB
    layer only ever read the un-archived `metering` table (the monthly
    archive deleted older rows into metering_history, which MetricStorage
    never queried). metering_weather carries a ~10-day TTL, so weather data
    older than that is not visible here - it lives only in the hourly rollup
    (metering_weather_hourly). Reading through to that rollup for longer
    windows is a possible future addition; power is retained at full
    resolution indefinitely, so it has no such horizon.

Timestamp handling matches MetricStorage exactly: mdatatime is treated as
UTC and converted to config.local_timezone for display (toTimeZone here,
CONVERT_TZ there); window bounds are compared against stored UTC as-is.

Unlike pymysql, clickhouse-connect's client pools its connections and is
safe to query from several threads at once (ai_agent_server.py runs each
request in its own thread), so there is no per-query lock - only the lazy
client creation is guarded.
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import clickhouse_connect

from Config import Config

# DB-agnostic result types and renderers live in the MariaDB module; reuse
# them so both layers return identical objects. Re-exported below for callers
# that would rather import everything storage-related from one module.
from ai_agent_storage import (  # noqa: F401  (re-exported)
    HistoryPoint,
    MetricStats,
    MetricValue,
    format_current,
    format_history,
    format_stats,
    format_today_outside,
)

logger = logging.getLogger("ai-agent")

# Fact tables and the metric that lives in the power one; every other metric
# lives in the weather table. See db/clickhouse/weather_clickhouse.sql.
WEATHER_TABLE = "metering_weather"
POWER_TABLE = "metering_power"
POWER_METRIC = "power"


def _plan_tables(metrics: list[str] | None) -> list[tuple[str, list[str] | None]]:
    """Map the requested metrics onto the fact tables that hold them.

    Returns [(table, metrics_for_that_table), ...]. `metrics=None` means
    "every metric", which becomes both tables with no metric filter. A
    metrics list is split so power goes to metering_power and the rest to
    metering_weather; a table with no requested metrics is left out."""
    if metrics is None:
        return [(WEATHER_TABLE, None), (POWER_TABLE, None)]
    weather = [m for m in metrics if m != POWER_METRIC]
    power = [m for m in metrics if m == POWER_METRIC]
    plans: list[tuple[str, list[str] | None]] = []
    if weather:
        plans.append((WEATHER_TABLE, weather))
    if power:
        plans.append((POWER_TABLE, power))
    return plans


def _filters(metric_filter: list[str] | None, location_filter: list[str] | None) -> str:
    """Build the shared metric/location WHERE fragment (leading ' AND ' each),
    using has(array, col) so the values bind as array parameters."""
    fragment = ""
    if metric_filter:
        fragment += " AND has({metrics:Array(String)}, m.metric)"
    if location_filter:
        fragment += " AND has({locs:Array(String)}, l.location)"
    return fragment


class MetricStorageClickhouse:
    """
    Read-only access to the ClickHouse `weather` database.

    Usage:
        storage = MetricStorageClickhouse(config)
        storage.get_current(["Weather station"])
        storage.close()
    """

    def __init__(self, config: Config):
        self._config = config
        self._client = None
        self._outside_locations: list[str] | None = None
        self._inside_locations: list[str] | None = None
        # Guards lazy client creation only; queries themselves are concurrent.
        self._lock = threading.Lock()

    def _connect(self):
        """Live client, created on first use. clickhouse-connect manages an
        HTTP connection pool, so a dropped connection heals on the next
        request without a ping/reconnect dance."""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            self._client = clickhouse_connect.get_client(
                host=self._config.ch_host,
                port=self._config.ch_port,
                username=self._config.ch_user_readonly,
                password=self._config.ch_password_readonly,
                database=self._config.ch_database,
                secure=self._config.ch_secure,
                connect_timeout=10,
                # 0 = no client-side row cap; a history window can easily
                # exceed clickhouse-connect's default limit otherwise.
                query_limit=0,
            )
            logger.info(
                "Connected to ClickHouse %s:%s/%s",
                self._config.ch_host, self._config.ch_port, self._config.ch_database,
            )
            return self._client

    def _query(self, query: str, parameters: dict) -> list[dict]:
        logger.info("Query: %s | params=%s", query, parameters)
        client = self._connect()
        return list(client.query(query, parameters=parameters).named_results())

    def get_current(
        self, locations: list[str] | None = None, metrics: list[str] | None = None
    ) -> dict[str, dict[str, MetricValue]]:
        """Latest value of every metric reported at each location, keyed by
        location then metric. Defaults to every location; pass `locations`
        or `metrics` to narrow it. If several sensors at a location report
        the same metric, the most recent wins (argMax on mdatatime)."""
        current: dict[str, dict[str, MetricValue]] = {}
        for table, metric_filter in _plan_tables(metrics):
            query = (
                "SELECT l.location AS location, m.metric AS metric, "
                "argMax(me.value, me.mdatatime) AS value, "
                "toTimeZone(max(me.mdatatime), {tz:String}) AS taken_at, "
                "argMax(s.name, me.mdatatime) AS sensor_name "
                f"FROM {table} me "
                "JOIN metric m ON m.metricid = me.metric_metricid "
                "JOIN sensor s ON s.sensorid = me.sensor_sensorid "
                "JOIN location l ON l.locid = s.location_locid "
                "WHERE 1" + _filters(metric_filter, locations) +
                " GROUP BY l.location, m.metric"
            )
            parameters = {"tz": self._config.local_timezone}
            if metric_filter:
                parameters["metrics"] = metric_filter
            if locations:
                parameters["locs"] = locations
            for row in self._query(query, parameters):
                by_metric = current.setdefault(row["location"], {})
                metric = row["metric"]
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
        """Location names with outside=0, loaded lazily and cached.
        Call refresh_locations() to reload."""
        if self._inside_locations is None:
            self.refresh_locations()
        return list(self._inside_locations)

    def refresh_locations(self) -> None:
        """Reload outside_locations/inside_locations from the `location`
        table - call this if locations may have been added/changed since the
        lists were last loaded."""
        outside: list[str] = []
        inside: list[str] = []
        for row in self._query("SELECT location, outside FROM location", {}):
            (outside if row["outside"] else inside).append(row["location"])
        self._outside_locations = outside
        self._inside_locations = inside

    def get_stats(
        self,
        metric: str | list[str],
        period: timedelta,
        locations: list[str] | None = None,
        by_sensor: bool = False,
    ) -> dict[str, MetricStats] | dict[str, dict[str, MetricStats]]:
        """Min/max/avg/count over the last `period`, ending now. Pass a
        single metric name for the classic shape, keyed by location; pass a
        list to pull several metrics at once, keyed by location then metric,
        same shape as get_current()'s. Defaults to every location with data
        in that window; pass `locations` to narrow it (this still filters by
        location even when by_sensor=True). A location/metric with no
        readings in the window is simply absent from the result.

        by_sensor=True keys the result by sensor name instead of location -
        see get_history() for why (several distinctly-named sensors, e.g.
        power meters, can share one location, which would otherwise blend
        their stats into one bucket)."""
        metrics = [metric] if isinstance(metric, str) else metric
        since = datetime.now() - period
        key_col = "s.name" if by_sensor else "l.location"

        by_location: dict[str, dict[str, MetricStats]] = {}
        for table, metric_filter in _plan_tables(metrics):
            query = (
                f"SELECT {key_col} AS grp, m.metric AS metric, "
                "min(me.value) AS min, max(me.value) AS max, "
                "avg(me.value) AS avg, count() AS count "
                f"FROM {table} me "
                "JOIN metric m ON m.metricid = me.metric_metricid "
                "JOIN sensor s ON s.sensorid = me.sensor_sensorid "
                "JOIN location l ON l.locid = s.location_locid "
                "WHERE me.mdatatime >= {since:DateTime}"
                + _filters(metric_filter, locations) +
                f" GROUP BY {key_col}, m.metric"
            )
            parameters: dict = {"since": since}
            if metric_filter:
                parameters["metrics"] = metric_filter
            if locations:
                parameters["locs"] = locations
            for row in self._query(query, parameters):
                if not row["count"]:
                    continue
                by_location.setdefault(row["grp"], {})[row["metric"]] = MetricStats(
                    metric=row["metric"],
                    min=row["min"],
                    max=row["max"],
                    avg=row["avg"],
                    count=row["count"],
                )

        if isinstance(metric, str):
            return {
                location: per_metric[metric]
                for location, per_metric in by_location.items()
                if metric in per_metric
            }
        return by_location

    def get_stats_last_hours(
        self,
        metric: str | list[str],
        hours: int,
        locations: list[str] | None = None,
        by_sensor: bool = False,
    ) -> dict[str, MetricStats] | dict[str, dict[str, MetricStats]]:
        """get_stats over the last `hours`, ending now."""
        return self.get_stats(metric, timedelta(hours=hours), locations, by_sensor)

    def get_stats_last_days(
        self, metric: str | list[str], days: int, locations: list[str] | None = None
    ) -> dict[str, MetricStats] | dict[str, dict[str, MetricStats]]:
        """get_stats over the last `days`, ending now."""
        return self.get_stats(metric, timedelta(days=days), locations)

    def get_stats_outside_last_hours(
        self, metric: str | list[str], hours: int
    ) -> dict[str, MetricStats] | dict[str, dict[str, MetricStats]]:
        """get_stats over the last `hours`, ending now, outside_locations only."""
        return self.get_stats(metric, timedelta(hours=hours), self.outside_locations)

    def get_history(
        self,
        metric: str | list[str],
        start: datetime,
        end: datetime,
        locations: list[str] | None = None,
        by_sensor: bool = False,
    ) -> dict[str, list[HistoryPoint]] | dict[str, dict[str, list[HistoryPoint]]]:
        """Every reading between start and end (inclusive), oldest first.
        Pass a single metric name for the classic shape, keyed by location;
        pass a list to pull several metrics at once, keyed by location then
        metric, same shape as get_current()'s. Defaults to every location
        with data in that window; pass `locations` to narrow it (this still
        filters by location even when by_sensor=True).

        by_sensor=True keys the result by sensor name instead of location.
        Several distinctly-named sensors can share one location (e.g.
        several power meters in the same room), so location grouping blends
        their readings into a single series - by_sensor keeps them apart.
        Used for power history, where that blending would otherwise merge
        unrelated devices into one graph line."""
        metrics = [metric] if isinstance(metric, str) else metric
        key_col = "s.name" if by_sensor else "l.location"

        single = isinstance(metric, str)
        history: dict[str, list[HistoryPoint]] = {}
        history_multi: dict[str, dict[str, list[HistoryPoint]]] = {}
        for table, metric_filter in _plan_tables(metrics):
            query = (
                f"SELECT {key_col} AS grp, m.metric AS metric, "
                "toTimeZone(me.mdatatime, {tz:String}) AS taken_at, "
                "me.value AS value "
                f"FROM {table} me "
                "JOIN metric m ON m.metricid = me.metric_metricid "
                "JOIN sensor s ON s.sensorid = me.sensor_sensorid "
                "JOIN location l ON l.locid = s.location_locid "
                "WHERE me.mdatatime BETWEEN {start:DateTime} AND {end:DateTime}"
                + _filters(metric_filter, locations) +
                " ORDER BY grp ASC, m.metric ASC, me.mdatatime ASC"
            )
            parameters: dict = {"tz": self._config.local_timezone, "start": start, "end": end}
            if metric_filter:
                parameters["metrics"] = metric_filter
            if locations:
                parameters["locs"] = locations
            for row in self._query(query, parameters):
                point = HistoryPoint(taken_at=row["taken_at"], value=row["value"])
                if single:
                    history.setdefault(row["grp"], []).append(point)
                else:
                    history_multi.setdefault(row["grp"], {}).setdefault(
                        row["metric"], []
                    ).append(point)
        return history if single else history_multi

    def get_history_last_hours(
        self,
        metric: str | list[str],
        hours: int,
        locations: list[str] | None = None,
        by_sensor: bool = False,
    ) -> dict[str, list[HistoryPoint]] | dict[str, dict[str, list[HistoryPoint]]]:
        """get_history over the last `hours`, ending now."""
        end = datetime.now()
        return self.get_history(metric, end - timedelta(hours=hours), end, locations, by_sensor)

    def get_history_outside_last_hours(
        self, metric: str | list[str], hours: int
    ) -> dict[str, list[HistoryPoint]] | dict[str, dict[str, list[HistoryPoint]]]:
        """get_history over the last `hours`, ending now, outside_locations only."""
        end = datetime.now()
        return self.get_history(metric, end - timedelta(hours=hours), end, self.outside_locations)

    def get_history_last_days(
        self, metric: str | list[str], days: int, locations: list[str] | None = None
    ) -> dict[str, list[HistoryPoint]] | dict[str, dict[str, list[HistoryPoint]]]:
        """get_history over the last `days`, ending now."""
        end = datetime.now()
        return self.get_history(metric, end - timedelta(days=days), end, locations)

    def get_history_outside_last_days(
        self, metric: str | list[str], days: int
    ) -> dict[str, list[HistoryPoint]] | dict[str, dict[str, list[HistoryPoint]]]:
        """get_history over the last `days`, ending now, outside_locations only."""
        end = datetime.now()
        return self.get_history(metric, end - timedelta(days=days), end, self.outside_locations)

    def today_outside(self) -> dict[str, dict[str, list[HistoryPoint]]]:
        """Temperature and humidity readings for today, from 08:00 up to
        21:00 or now (whichever is earlier), from outside_locations only.
        Keyed by location then metric. Before 08:00 the window is empty.

        "Today" and 08:00/21:00 are anchored to config.local_timezone, then
        converted to UTC to match stored mdatatime - identical to
        MetricStorage.today_outside()."""
        tz = ZoneInfo(self._config.local_timezone)
        now_local = datetime.now(tz)
        start_local = now_local.replace(hour=8, minute=0, second=0, microsecond=0)
        end_local = min(now_local, now_local.replace(hour=21, minute=0, second=0, microsecond=0))
        start = start_local.astimezone(timezone.utc).replace(tzinfo=None)
        end = end_local.astimezone(timezone.utc).replace(tzinfo=None)
        return self.get_history(["temperature", "humidity"], start, end, self.outside_locations)

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
