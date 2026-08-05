-- ClickHouse schema for the `weather` IoT sensor DB
-- Adapted from MariaDB weather.sql
-- Run with: clickhouse-client --multiquery < weather_clickhouse.sql

CREATE DATABASE IF NOT EXISTS weather;

-- -----------------------------------------------------
-- Dimension tables (small, rarely-changing reference data)
-- No FOREIGN KEY support in ClickHouse — integrity is
-- enforced at the application layer.
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS weather.location
(
    locid    UInt32,
    location String,
    outside  UInt8 DEFAULT 0        -- 1 = outside, 0 = inside
)
ENGINE = MergeTree
ORDER BY locid;

CREATE TABLE IF NOT EXISTS weather.sensor
(
    sensorid       UInt32,
    name           String,
    location_locid UInt32,
    sensor_type    LowCardinality(String) DEFAULT 'weather'   -- 'weather' or 'power' — drives which fact table it lands in
)
ENGINE = MergeTree
ORDER BY sensorid;

CREATE TABLE IF NOT EXISTS weather.metric
(
    metricid UInt32,
    metric   LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY metricid;

INSERT INTO weather.metric (metricid, metric) VALUES
(1, 'temperature'),
(2, 'humidity'),
(3, 'CO2'),
(4, 'TVOC'),
(5, 'battery'),
(6, 'pressure'),
(7, 'illuminance'),
(8, 'power');

-- =======================================================
-- WEATHER SENSORS: 10-min sampling
-- Raw kept ~10 days, hourly rollup kept forever
-- =======================================================

CREATE TABLE IF NOT EXISTS weather.metering_weather
(
    mdatatime       DateTime CODEC(DoubleDelta, ZSTD),
    value           Float32  CODEC(Gorilla, ZSTD),
    sensor_sensorid UInt32,
    metric_metricid UInt32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(mdatatime)
ORDER BY (metric_metricid, sensor_sensorid, mdatatime)
TTL mdatatime + INTERVAL 10 DAY;   -- raw resolution only needed for "today ... last 3-10 days"

CREATE TABLE IF NOT EXISTS weather.metering_weather_hourly
(
    hour            DateTime,
    sensor_sensorid UInt32,
    metric_metricid UInt32,
    value_avg       AggregateFunction(avg, Float32),
    sample_count    AggregateFunction(count, Float32)
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(hour)
ORDER BY (metric_metricid, sensor_sensorid, hour);
-- no TTL: this is the permanent long-term history

CREATE MATERIALIZED VIEW IF NOT EXISTS weather.mv_metering_weather_hourly
TO weather.metering_weather_hourly
AS
SELECT
    toStartOfHour(mdatatime) AS hour,
    sensor_sensorid,
    metric_metricid,
    avgState(value)          AS value_avg,
    countState(value)        AS sample_count
FROM weather.metering_weather
GROUP BY hour, sensor_sensorid, metric_metricid;

-- -----------------------------------------------------
-- Convenience view: transparently picks recent raw data
-- or older hourly-averaged data depending on the query range,
-- so your app can query one "logical" table either way.
-- -----------------------------------------------------
-- (Simplest to implement this branching logic in the app layer:
--  query metering_weather for date ranges within the last 10 days,
--  metering_weather_hourly (with avgMerge/countMerge) for anything older.)

-- =======================================================
-- POWER SENSORS: 30-sec sampling, kept forever at full resolution
-- =======================================================

CREATE TABLE IF NOT EXISTS weather.metering_power
(
    mdatatime       DateTime CODEC(DoubleDelta, ZSTD),
    value           Float32  CODEC(Gorilla, ZSTD),
    sensor_sensorid UInt32,
    metric_metricid UInt32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(mdatatime)
ORDER BY (sensor_sensorid, mdatatime);
-- no TTL: raw resolution retained indefinitely per requirement

-- Optional: daily rollup purely to speed up long-range dashboard
-- queries (e.g. "power over the last 2 years"). Raw data in
-- metering_power is untouched and still fully queryable.
CREATE TABLE IF NOT EXISTS weather.metering_power_daily
(
    day             Date,
    sensor_sensorid UInt32,
    metric_metricid UInt32,
    value_avg       AggregateFunction(avg, Float32),
    value_min       AggregateFunction(min, Float32),
    value_max       AggregateFunction(max, Float32),
    sample_count    AggregateFunction(count, Float32)
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(day)
ORDER BY (sensor_sensorid, metric_metricid, day);

CREATE MATERIALIZED VIEW IF NOT EXISTS weather.mv_metering_power_daily
TO weather.metering_power_daily
AS
SELECT
    toDate(mdatatime)  AS day,
    sensor_sensorid,
    metric_metricid,
    avgState(value)    AS value_avg,
    minState(value)    AS value_min,
    maxState(value)    AS value_max,
    countState(value)  AS sample_count
FROM weather.metering_power
GROUP BY day, sensor_sensorid, metric_metricid;

-- Example queries:
--
-- Recent weather (raw):
-- SELECT * FROM weather.metering_weather
-- WHERE metric_metricid = 1 AND mdatatime >= now() - INTERVAL 3 DAY;
--
-- Older weather (hourly avg):
-- SELECT hour, avgMerge(value_avg) AS avg_value
-- FROM weather.metering_weather_hourly
-- WHERE metric_metricid = 1 AND hour >= '2025-01-01'
-- GROUP BY hour ORDER BY hour;
--
-- Power, full precision, any range:
-- SELECT * FROM weather.metering_power
-- WHERE sensor_sensorid = 10 AND mdatatime >= now() - INTERVAL 1 HOUR;
--
-- Power, long-range dashboard (fast, uses rollup):
-- SELECT day, avgMerge(value_avg) AS avg_power
-- FROM weather.metering_power_daily
-- WHERE sensor_sensorid = 10
-- GROUP BY day ORDER BY day;
