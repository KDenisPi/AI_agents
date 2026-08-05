#!/usr/bin/env bash
# Export weather data from MariaDB into TSV files for ClickHouse import.
# Run this against your MariaDB/MySQL server (adjust connection settings below).

set -euo pipefail

# ---- connection settings: edit these ----
MYSQL_HOST="localhost"
MYSQL_PORT="3306"
MYSQL_USER="weather"
MYSQL_PASSWORD="CHANGE_ME"
MYSQL_DB="weather"

OUTDIR="./mariadb_export"
mkdir -p "$OUTDIR"

MYSQL="mysql -h$MYSQL_HOST -P$MYSQL_PORT -u$MYSQL_USER -p$MYSQL_PASSWORD -N -B $MYSQL_DB"
# -N: no column headers, -B: tab-separated batch output
# (client-side redirection avoids needing FILE privilege / server-side INTO OUTFILE)

echo "Exporting location..."
$MYSQL -e "SELECT locid, location, IFNULL(outside, 0) FROM location" > "$OUTDIR/location.tsv"

echo "Exporting sensor..."
# No power sensors yet — sensor_type hardcoded to 'weather' for all rows.
# When you add a power sensor later, switch this to the CASE version
# (see git history / just ask) to flag specific sensorids as 'power'.
$MYSQL -e "SELECT sensorid, name, location_locid, 'weather'
           FROM sensor" > "$OUTDIR/sensor.tsv"

echo "Exporting metric..."
$MYSQL -e "SELECT metricid, metric FROM metric" > "$OUTDIR/metric.tsv"

echo "Exporting metering (weather metrics, metricid != 8)..."
$MYSQL -e "SELECT mdatatime, value, sensor_sensorid, metric_metricid
           FROM metering
           WHERE metric_metricid <> 8
           ORDER BY mdatatime" > "$OUTDIR/metering_weather.tsv"

echo "Exporting metering (power metric, metricid = 8)..."
$MYSQL -e "SELECT mdatatime, value, sensor_sensorid, metric_metricid
           FROM metering
           WHERE metric_metricid = 8
           ORDER BY mdatatime" > "$OUTDIR/metering_power.tsv"

echo "Exporting metering_history (legacy hourly archive, if any rows exist)..."
$MYSQL -e "SELECT mdatatime, value, sample_count, sensor_sensorid, metric_metricid
           FROM metering_history
           ORDER BY mdatatime" > "$OUTDIR/metering_history.tsv"

echo "Done. Files written to $OUTDIR/"
wc -l "$OUTDIR"/*.tsv
