#!/usr/bin/env bash
# Import weather data (previously exported by export_from_mariadb.sh) into ClickHouse.
# Assumes the weather_clickhouse.sql schema has already been applied.
#
# Run this from the host, with TSV files from export_from_mariadb.sh in ./mariadb_export/

set -euo pipefail

# ---- connection settings: edit these ----
CH_CONTAINER="iot-clickhouse-server"   # docker container name
CH_USER="weather"                       # or "default" if you haven't set up app users yet
CH_PASSWORD="CHANGE_ME"
CH_DB="weather"

INDIR="./mariadb_export"

ch_insert () {
    local table="$1"
    local file="$2"
    echo "Importing into $table from $file ..."
    docker exec -i "$CH_CONTAINER" clickhouse-client \
        --user "$CH_USER" --password "$CH_PASSWORD" \
        --query "INSERT INTO $CH_DB.$table FORMAT TabSeparated" \
        < "$file"
}

# Order matters only in the sense that dimension tables are logically
# "parents" — ClickHouse won't reject out-of-order inserts (no FK checks),
# but importing them first keeps things sane if you query mid-import.

ch_insert location "$INDIR/location.tsv"
ch_insert sensor "$INDIR/sensor.tsv"
ch_insert metric "$INDIR/metric.tsv"

# Raw fact tables. Inserting into these also fires the materialized views
# (mv_metering_weather_hourly / mv_metering_power_daily), so the rollup
# tables get populated automatically as part of this same import — no
# separate backfill step needed for them.
ch_insert metering_weather "$INDIR/metering_weather.tsv"
ch_insert metering_power "$INDIR/metering_power.tsv"

# Legacy hourly archive: imported as plain values, not merged into the
# AggregatingMergeTree state column (see notes in chat). Requires a
# plain destination table — create it first if it doesn't exist yet:
#
#   CREATE TABLE IF NOT EXISTS weather.metering_history_legacy
#   (
#       mdatatime       DateTime,
#       value           Float32,
#       sample_count    UInt32,
#       sensor_sensorid UInt32,
#       metric_metricid UInt32
#   )
#   ENGINE = MergeTree
#   PARTITION BY toYYYYMM(mdatatime)
#   ORDER BY (metric_metricid, sensor_sensorid, mdatatime);
#
if [ -s "$INDIR/metering_history.tsv" ]; then
    ch_insert metering_history_legacy "$INDIR/metering_history.tsv"
else
    echo "metering_history.tsv is empty, skipping."
fi

echo "Import complete."

# ---- after import: fill in sensor_type for power sensors ----
# sensor.sensor_type defaulted to 'weather' for every row on export
# (MariaDB has no such column). Update the power sensors manually, e.g.:
#
#   docker exec -it iot-clickhouse-server clickhouse-client \
#     --user "$CH_USER" --password "$CH_PASSWORD" --query \
#     "ALTER TABLE weather.sensor UPDATE sensor_type = 'power' WHERE sensorid IN (<ids here>)"
#
# Note ALTER ... UPDATE is an async mutation in ClickHouse; check progress via
# system.mutations if it doesn't seem to apply immediately.
