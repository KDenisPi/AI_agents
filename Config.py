"""
Collector configuration, read from the environment (optionally seeded from
a .env file next to this module). See collector.env.example for the full
list of variables and their defaults.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    # --- MariaDB ---
    db_host: str = "192.168.1.11"
    db_port: int = 3306
    db_user: str = "weather"
    db_password: str = "Dk94404!"
    db_name: str = "weather"

    # --- Hubitat Maker API ---
    hubitat_ip: str = "192.168.1.214"
    hubitat_app_id: str = "7"
    hubitat_token: str = "3b72adce-e0b3-43ae-8d1e-549bf82355d5"

    # --- Weather station, reached through its MCP server (SSE endpoint) ---
    weather_mcp_url: str = "http://localhost:8000/weather/sse"
    weather_sensor_id: int = 1000
    weather_sensor_name: str = "WeatherStation"
    weather_location_id: int = 100
    weather_location_name: str = "Weather station"
    weather_location_outside: bool = False

    # --- Scheduling ---
    interval_seconds: int = 600

    # Fallback for Hubitat devices that are not assigned to any room.
    default_location_id: int = 0
    default_location_name: str = "Unassigned"

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(Path(__file__).with_name(".env"))
        defaults = cls()

        def text(name: str, default: str) -> str:
            return os.getenv(name, default)

        def number(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw else default

        def flag(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            return raw.lower() in ("1", "true", "yes") if raw else default

        return cls(
            db_host=text("DB_HOST", defaults.db_host),
            db_port=number("DB_PORT", defaults.db_port),
            db_user=text("DB_USER", defaults.db_user),
            db_password=text("DB_PASSWORD", defaults.db_password),
            db_name=text("DB_NAME", defaults.db_name),
            hubitat_ip=text("HUBITAT_IP", defaults.hubitat_ip),
            hubitat_app_id=text("HUBITAT_APP_ID", defaults.hubitat_app_id),
            hubitat_token=text("HUBITAT_TOKEN", defaults.hubitat_token),
            weather_mcp_url=text("WEATHER_MCP_URL", defaults.weather_mcp_url),
            weather_sensor_id=number("WEATHER_SENSOR_ID", defaults.weather_sensor_id),
            weather_sensor_name=text("WEATHER_SENSOR_NAME", defaults.weather_sensor_name),
            weather_location_id=number("WEATHER_LOCATION_ID", defaults.weather_location_id),
            weather_location_name=text("WEATHER_LOCATION_NAME", defaults.weather_location_name),
            weather_location_outside=flag(
                "WEATHER_LOCATION_OUTSIDE", defaults.weather_location_outside
            ),
            interval_seconds=number("INTERVAL_SECONDS", defaults.interval_seconds),
            default_location_id=number("DEFAULT_LOCATION_ID", defaults.default_location_id),
            default_location_name=text("DEFAULT_LOCATION_NAME", defaults.default_location_name),
        )
