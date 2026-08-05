-- ClickHouse user/role setup for the `weather` database
-- Analogous to db/weather_users.sql in the MariaDB project.
-- Fill in real passwords before running; keep this file out of git,
-- same as its MariaDB counterpart.
--
-- Run as the `default` superuser:
--   docker exec -it iot-clickhouse-server clickhouse-client --multiquery < weather_users_clickhouse.sql

-- -----------------------------------------------------
-- Roles (define permissions once, attach to users)
-- -----------------------------------------------------

CREATE ROLE IF NOT EXISTS weather_readwrite;
GRANT SELECT, INSERT ON weather.* TO weather_readwrite;

CREATE ROLE IF NOT EXISTS weather_readonly;
GRANT SELECT ON weather.* TO weather_readonly;

-- -----------------------------------------------------
-- Users
-- -----------------------------------------------------

CREATE USER IF NOT EXISTS weather
    IDENTIFIED WITH sha256_password BY '<CHANGE_ME_READWRITE_PASSWORD>';
GRANT weather_readwrite TO weather;
SET DEFAULT ROLE weather_readwrite TO weather;

CREATE USER IF NOT EXISTS weather_read
    IDENTIFIED WITH sha256_password BY '<CHANGE_ME_READONLY_PASSWORD>';
GRANT weather_readonly TO weather_read;
SET DEFAULT ROLE weather_readonly TO weather_read;

-- weather_admin: only create this if you want a distinct admin
-- identity separate from `default` (e.g. for audit trails or
-- multiple admins). Otherwise skip it and use `default` for
-- schema/DDL work.
--
-- CREATE USER IF NOT EXISTS weather_admin
--     IDENTIFIED WITH sha256_password BY '<CHANGE_ME_ADMIN_PASSWORD>';
-- GRANT ALL ON weather.* TO weather_admin WITH GRANT OPTION;

-- -----------------------------------------------------
-- Secure the built-in superuser while you're at it
-- -----------------------------------------------------
-- ALTER USER default IDENTIFIED WITH sha256_password BY '<CHANGE_ME_DEFAULT_PASSWORD>';
