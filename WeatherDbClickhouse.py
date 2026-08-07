"""
ClickHouse sink for collected readings - see
db/clickhouse/weather_clickhouse.sql for the schema.

The ClickHouse counterpart of WeatherDb. Two differences from the MariaDB
version drive its shape:

  * Readings are routed by metric into two fact tables. The 'power' metric
    goes to metering_power (full resolution, kept forever); everything else
    goes to metering_weather (raw kept ~10 days). Rollups and retention are
    handled by ClickHouse itself - the metering_*_hourly/_daily materialized
    views and the tables' TTLs - so there is no archive_metering() here, the
    way there is in WeatherDb.

  * ClickHouse has no INSERT IGNORE and MergeTree does not deduplicate on
    insert. location/sensor ids already present are tracked in an in-memory
    set seeded from the tables on connect, and only unseen ids are inserted,
    so the dimension tables stay free of duplicate rows across restarts. Raw
    metering rows are *not* deduplicated - a mid-interval restart can insert
    a duplicate reading, which is accepted as a rare, negligible event (it
    only nudges an hourly/daily average by one sample).

clickhouse-connect's client is blocking, so the collector calls save()
through asyncio.to_thread.
"""

import logging
from datetime import datetime

import clickhouse_connect
from clickhouse_connect.driver.exceptions import ClickHouseError

from Config import Config
from Reading import Reading

logger = logging.getLogger("weather-db-clickhouse")

# Column widths mirror db/weather.sql, so names look the same in either
# store - ClickHouse String has no length limit of its own to enforce.
LOCATION_NAME_MAX = 60
SENSOR_NAME_MAX = 45

# Fact tables and their shared column list. Which table a reading lands in
# is decided per reading by Reading.is_power (see save()).
WEATHER_TABLE = "metering_weather"
POWER_TABLE = "metering_power"
METERING_COLUMNS = ["mdatatime", "value", "sensor_sensorid", "metric_metricid"]

# Dimension tables. sensor_type is ClickHouse-only ('weather' or 'power');
# it comes from the reading's metric, not from MariaDB.
LOCATION_TABLE = "location"
LOCATION_COLUMNS = ["locid", "location", "outside"]
SENSOR_TABLE = "sensor"
SENSOR_COLUMNS = ["sensorid", "name", "location_locid", "sensor_type"]


def _location_row(reading: Reading) -> list:
    return [
        reading.location_id,
        reading.location_name[:LOCATION_NAME_MAX],
        1 if reading.outside else 0,
    ]


def _sensor_row(reading: Reading) -> list:
    return [
        reading.sensor_id,
        reading.sensor_name[:SENSOR_NAME_MAX],
        reading.location_id,
        "power" if reading.is_power else "weather",
    ]


def _metering_row(reading: Reading) -> list:
    return [reading.taken_at, reading.value, reading.sensor_id, reading.metric_id]


def _sql_literal(value) -> str:
    """Format a single value for an INSERT log preview - readable, not meant
    to be executed."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, datetime):
        return "'" + value.strftime("%Y-%m-%d %H:%M:%S") + "'"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _render_insert(table: str, columns: list[str], row: list) -> str:
    """Render one INSERT ... VALUES statement for dry-run logging."""
    cols = ", ".join(columns)
    vals = ", ".join(_sql_literal(v) for v in row)
    return f"INSERT INTO {table} ({cols}) VALUES ({vals})"


class WeatherDbClickhouse:
    """
    Usage:
        db = WeatherDbClickhouse(config)
        saved = db.save(readings)
        db.close()

    save() raises clickhouse_connect ClickHouseError if ClickHouse is
    unreachable or rejects the write - the collector catches it and moves on
    to the next cycle.
    """

    def __init__(self, config: Config):
        self._config = config
        self._client = None
        self._known_locations: set[int] = set()
        self._known_sensors: set[int] = set()

    def _connect(self):
        """Live client, created on first use. clickhouse-connect manages its
        own HTTP connection pool, so there is no ping/reconnect dance as in
        the pymysql version - a dropped connection is re-established on the
        next request by the pool itself."""
        if self._client is not None:
            return self._client

        self._client = clickhouse_connect.get_client(
            host=self._config.ch_host,
            port=self._config.ch_port,
            username=self._config.ch_user,
            password=self._config.ch_password,
            database=self._config.ch_database,
            secure=self._config.ch_secure,
            connect_timeout=10,
        )
        self._seed_known(self._client)
        logger.info(
            "Connected to ClickHouse %s:%s/%s (%d known location(s), %d known sensor(s))",
            self._config.ch_host, self._config.ch_port, self._config.ch_database,
            len(self._known_locations), len(self._known_sensors),
        )
        return self._client

    def _seed_known(self, client) -> None:
        """Prime the id caches from what is already in the dimension tables,
        so an unchanged sensor is never re-inserted after a restart."""
        self._known_locations = {
            row[0] for row in client.query(f"SELECT DISTINCT locid FROM {LOCATION_TABLE}").result_rows
        }
        self._known_sensors = {
            row[0] for row in client.query(f"SELECT DISTINCT sensorid FROM {SENSOR_TABLE}").result_rows
        }

    def save(self, readings: list[Reading]) -> int:
        """Register any unseen locations/sensors, then insert the meterings,
        each into metering_weather or metering_power by metric.

        Returns the number of metering rows inserted - ClickHouse accepts
        every row (no unique index), so this is just how many were sent.
        Raises ClickHouseError if the write fails."""
        if not readings:
            return 0

        client = self._connect()
        try:
            self._register(client, readings)

            weather = [_metering_row(r) for r in readings if not r.is_power]
            power = [_metering_row(r) for r in readings if r.is_power]
            if weather:
                client.insert(WEATHER_TABLE, weather, column_names=METERING_COLUMNS)
            if power:
                client.insert(POWER_TABLE, power, column_names=METERING_COLUMNS)
            return len(weather) + len(power)
        except ClickHouseError:
            # A failed write may have been the one that would have registered
            # a new id - drop the caches so the next attempt re-checks.
            self._known_locations.clear()
            self._known_sensors.clear()
            raise

    def _register(self, client, readings: list[Reading]) -> None:
        """Insert location and sensor rows for ids not seen before. Only ever
        inserts: a sensor renamed or moved in Hubitat keeps whatever the table
        already says, so hand-edits survive (same policy as WeatherDb)."""
        new_locations = []
        new_sensors = []
        added_locations: set[int] = set()
        added_sensors: set[int] = set()
        for reading in readings:
            if (
                reading.location_id not in self._known_locations
                and reading.location_id not in added_locations
            ):
                new_locations.append(_location_row(reading))
                added_locations.add(reading.location_id)
            if (
                reading.sensor_id not in self._known_sensors
                and reading.sensor_id not in added_sensors
            ):
                new_sensors.append(_sensor_row(reading))
                added_sensors.add(reading.sensor_id)

        if new_locations:
            client.insert(LOCATION_TABLE, new_locations, column_names=LOCATION_COLUMNS)
            self._known_locations |= added_locations
        if new_sensors:
            client.insert(SENSOR_TABLE, new_sensors, column_names=SENSOR_COLUMNS)
            self._known_sensors |= added_sensors

    def preview(self, readings: list[Reading]) -> list[str]:
        """Render the INSERT statements save() would run, values inlined,
        without opening a connection - for dry-run logging only.

        Each distinct location and sensor appears once, followed by one
        metering row per reading, into whichever fact table its metric picks."""
        statements: list[str] = []
        seen_locations: set[int] = set()
        seen_sensors: set[int] = set()
        for reading in readings:
            if reading.location_id not in seen_locations:
                statements.append(_render_insert(LOCATION_TABLE, LOCATION_COLUMNS, _location_row(reading)))
                seen_locations.add(reading.location_id)
            if reading.sensor_id not in seen_sensors:
                statements.append(_render_insert(SENSOR_TABLE, SENSOR_COLUMNS, _sensor_row(reading)))
                seen_sensors.add(reading.sensor_id)
        for reading in readings:
            table = POWER_TABLE if reading.is_power else WEATHER_TABLE
            statements.append(_render_insert(table, METERING_COLUMNS, _metering_row(reading)))
        return statements

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
