"""
Collector configuration, read from the environment (optionally seeded from
a .env file next to this module). See collector.env.example for the full
list of variables and their defaults.
"""

import logging
import os
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
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
    # Read-only credentials for ai_agent.py's MetricStorage - it never writes.
    db_user_reasonly: str = "weather_read"
    db_password_readonly: str = "Dk94404!"

    # --- Hubitat Maker API ---
    hubitat_ip: str = "192.168.1.214"
    hubitat_app_id: str = "7"
    hubitat_token: str = "3b72adce-e0b3-43ae-8d1e-549bf82355d5"

    # --- Weather station, reached through its MCP server (SSE endpoint) ---
    weather_mcp_url: str = "http://192.168.1.57:8000/weather/sse"
    # The physical weather station's own HTTP API, queried by the MCP server
    # (McpWeather.py) itself - not to be confused with weather_mcp_url above.
    weather_station_url: str = "http://192.168.1.7:8080/api/status"
    weather_sensor_id: int = 1000
    weather_sensor_name: str = "WeatherStation"
    weather_location_id: int = 100
    weather_location_name: str = "Weather station"
    weather_location_outside: bool = False

    # --- Ollama ---
    ollama_url: str = "http://192.168.1.57:11434"
    pllama_model_1: str = "llama3.1"
    pllama_model_2: str = "qwen3.6"
    pllama_model_3: str = "llama3.2"

    # --- Scheduling ---
    interval_seconds: int = 600

    # When True, collect from HTTP/MCP as usual but write nothing - the
    # prepared SQL INSERTs are logged instead. Overridden by --dry-run.
    dry_run: bool = False

    # Fallback for Hubitat devices that are not assigned to any room.
    default_location_id: int = 0
    default_location_name: str = "Unassigned"

    # --- Logging ---
    # One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_level: str = "INFO"
    # Rotated every 24h from process start. Empty disables file logging -
    # console output only.
    log_file: str = "logs/app.log"
    # Rotated log files to keep before the oldest is deleted. 0 keeps all.
    log_backup_count: int = 7

    def configure_logging(self) -> None:
        """Set up the root logger from log_level/log_file. Call this once,
        from each entry point, right after Config.from_env() - library
        modules should only do logging.getLogger(__name__), never
        logging.basicConfig().
        """
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(
                TimedRotatingFileHandler(
                    log_path,
                    when="D",
                    interval=1,
                    backupCount=self.log_backup_count,
                )
            )

        for handler in handlers:
            handler.setFormatter(formatter)

        logging.basicConfig(level=self.log_level.upper(), handlers=handlers)

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
            db_user_reasonly=text("DB_USER_READONLY", defaults.db_user_reasonly),
            db_password_readonly=text("DB_PASSWORD_READONLY", defaults.db_password_readonly),
            hubitat_ip=text("HUBITAT_IP", defaults.hubitat_ip),
            hubitat_app_id=text("HUBITAT_APP_ID", defaults.hubitat_app_id),
            hubitat_token=text("HUBITAT_TOKEN", defaults.hubitat_token),
            weather_mcp_url=text("WEATHER_MCP_URL", defaults.weather_mcp_url),
            weather_station_url=text("WEATHER_STATION_URL", defaults.weather_station_url),
            weather_sensor_id=number("WEATHER_SENSOR_ID", defaults.weather_sensor_id),
            weather_sensor_name=text("WEATHER_SENSOR_NAME", defaults.weather_sensor_name),
            weather_location_id=number("WEATHER_LOCATION_ID", defaults.weather_location_id),
            weather_location_name=text("WEATHER_LOCATION_NAME", defaults.weather_location_name),
            weather_location_outside=flag(
                "WEATHER_LOCATION_OUTSIDE", defaults.weather_location_outside
            ),
            ollama_url=text("OLLAMA_URL", defaults.ollama_url),
            pllama_model_1=text("OLLAMA_MODEL_1", defaults.pllama_model_1),
            pllama_model_2=text("OLLAMA_MODEL_2", defaults.pllama_model_2),
            pllama_model_3=text("OLLAMA_MODEL_3", defaults.pllama_model_3),
            interval_seconds=number("INTERVAL_SECONDS", defaults.interval_seconds),
            dry_run=flag("DRY_RUN", defaults.dry_run),
            default_location_id=number("DEFAULT_LOCATION_ID", defaults.default_location_id),
            default_location_name=text("DEFAULT_LOCATION_NAME", defaults.default_location_name),
            log_level=text("LOG_LEVEL", defaults.log_level),
            log_file=text("LOG_FILE", defaults.log_file),
            log_backup_count=number("LOG_BACKUP_COUNT", defaults.log_backup_count),
        )
