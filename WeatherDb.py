"""
MariaDB sink for collected readings - see db/weather.sql for the schema.

Every write is idempotent: locations and sensors are auto-registered the
first time they appear, and metering rows lean on the metering_uq unique
index so re-running a cycle can never duplicate a measurement.

pymysql is blocking, so the collector calls save() through asyncio.to_thread.
"""

import logging

import pymysql

from Config import Config
from Reading import Reading

logger = logging.getLogger("weather-db")

# Column widths from db/weather.sql - names are truncated rather than
# letting STRICT_TRANS_TABLES reject the whole insert.
LOCATION_NAME_MAX = 60
SENSOR_NAME_MAX = 45

# The three statements a cycle runs. Kept as constants so save() executes
# them and preview() (dry-run) renders the exact same SQL.
LOCATION_INSERT = (
    "INSERT IGNORE INTO location (locid, location, outside) VALUES (%s, %s, %s)"
)
SENSOR_INSERT = (
    "INSERT IGNORE INTO sensor (sensorid, name, location_locid) VALUES (%s, %s, %s)"
)
METERING_INSERT = (
    "INSERT IGNORE INTO metering "
    "(mdatatime, value, sensor_sensorid, metric_metricid) VALUES (%s, %s, %s, %s)"
)


def _location_args(reading: Reading) -> tuple:
    return (
        reading.location_id,
        reading.location_name[:LOCATION_NAME_MAX],
        1 if reading.outside else 0,
    )


def _sensor_args(reading: Reading) -> tuple:
    return (
        reading.sensor_id,
        reading.sensor_name[:SENSOR_NAME_MAX],
        reading.location_id,
    )


def _metering_args(reading: Reading) -> tuple:
    return (reading.taken_at, reading.value, reading.sensor_id, reading.metric_id)


def _sql_literal(value) -> str:
    """Format a single value for an SQL log preview - readable, not meant
    to be executed."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _render(query: str, args: tuple) -> str:
    """Inline args into a %s-placeholder statement for logging."""
    rendered = query
    for arg in args:
        rendered = rendered.replace("%s", _sql_literal(arg), 1)
    return rendered


class WeatherDb:
    """
    Usage:
        db = WeatherDb(config)
        saved = db.save(readings)
        db.close()
    """

    def __init__(self, config: Config):
        self._config = config
        self._connection: pymysql.Connection | None = None
        self._known_locations: set[int] = set()
        self._known_sensors: set[int] = set()

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
            user=self._config.db_user,
            password=self._config.db_password,
            database=self._config.db_name,
            charset="utf8mb4",
            connect_timeout=10,
        )
        # A fresh connection may mean a different server state - the caches
        # of already-registered rows are no longer trustworthy.
        self._known_locations.clear()
        self._known_sensors.clear()
        logger.info("Connected to MariaDB %s:%s/%s",
                    self._config.db_host, self._config.db_port, self._config.db_name)
        return self._connection

    def save(self, readings: list[Reading]) -> int:
        """Register any unseen sensors, then insert the meterings.

        Returns the number of metering rows actually stored - rows already
        present for that sensor/metric/timestamp are skipped, not an error.
        Raises pymysql.MySQLError if the database is unreachable."""
        if not readings:
            return 0

        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                self._register(cursor, readings)
                stored = cursor.executemany(
                    METERING_INSERT,
                    [_metering_args(r) for r in readings],
                )
            connection.commit()
            return stored
        except pymysql.MySQLError:
            connection.rollback()
            # The caches describe rows that may have been rolled back with it.
            self._known_locations.clear()
            self._known_sensors.clear()
            raise

    def _register(self, cursor, readings: list[Reading]) -> None:
        """Insert location and sensor rows for ids not seen before.

        Only ever inserts: a sensor renamed or moved to another room in
        Hubitat keeps whatever this table already says, so hand-edits to
        location/sensor survive."""
        for reading in readings:
            if reading.location_id not in self._known_locations:
                cursor.execute(LOCATION_INSERT, _location_args(reading))
                self._known_locations.add(reading.location_id)

            if reading.sensor_id not in self._known_sensors:
                cursor.execute(SENSOR_INSERT, _sensor_args(reading))
                self._known_sensors.add(reading.sensor_id)

    def preview(self, readings: list[Reading]) -> list[str]:
        """Render the INSERT statements save() would run, values inlined,
        without opening a DB connection - for dry-run logging only.

        Each distinct location and sensor appears once (INSERT IGNORE makes
        them no-ops when the row already exists), followed by one metering
        row per reading."""
        statements: list[str] = []
        seen_locations: set[int] = set()
        seen_sensors: set[int] = set()
        for reading in readings:
            if reading.location_id not in seen_locations:
                statements.append(_render(LOCATION_INSERT, _location_args(reading)))
                seen_locations.add(reading.location_id)
            if reading.sensor_id not in seen_sensors:
                statements.append(_render(SENSOR_INSERT, _sensor_args(reading)))
                seen_sensors.add(reading.sensor_id)
        for reading in readings:
            statements.append(_render(METERING_INSERT, _metering_args(reading)))
        return statements

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
