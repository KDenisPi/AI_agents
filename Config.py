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
    # No secret defaults in source - real values live in .env (gitignored).
    db_password: str = ""
    db_name: str = "weather"
    # Read-only credentials for ai_agent.py's MetricStorage - it never writes.
    db_user_readonly: str = "weather_read"
    db_password_readonly: str = ""

    # --- ClickHouse (schema from db/clickhouse/weather_clickhouse.sql) ---
    # Used by the ClickHouse collector (WeatherDbClickhouse /
    # CollectorClickhouse) only - the MariaDB fields above still drive the
    # legacy Collector.py and ai_agent's MetricStorage. ch_port is the HTTP
    # interface (8123), which clickhouse-connect speaks; ch_secure=True
    # switches it to HTTPS.
    ch_host: str = "192.168.1.57"
    ch_port: int = 8123
    ch_user: str = "weather"
    # No secret defaults in source - real value lives in .env (gitignored).
    ch_password: str = ""
    ch_database: str = "weather"
    ch_secure: bool = False
    # Read-only credentials for ai_agent's MetricStorageClickhouse - it never
    # writes, same split as db_user_readonly on the MariaDB side.
    ch_user_readonly: str = "weather_read"
    ch_password_readonly: str = ""

    # --- Hubitat Maker API ---
    hubitat_ip: str = "192.168.1.214"
    hubitat_app_id: str = "7"
    hubitat_token: str = ""

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
    # Per-model overrides, for a model served by something other than the
    # shared Ollama host above (e.g. llama-server on its own port). Empty
    # (the default) falls back to ollama_url - see __post_init__.
    ollama_url_1: str = ""
    ollama_url_2: str = ""
    ollama_url_text_to_voice: str = ""
    # Keep the tag - a bare name resolves to <name>:latest, which the server
    # does not have for llama3.1.
    ollama_model_1: str = "llama3.1:8b"
    ollama_model_2: str = "qwen3.6"
    # Speech synthesis (text_to_voice.py). Must be an Orpheus-style model
    # that emits SNAC audio tokens - a general chat model returns prose and
    # produces no audio at all.
    ollama_model_text_to_voice: str = "sematre/orpheus:ft-en-3b-q4_k_m" #"sematre/orpheus:en-3b"
    # One of text_to_voice.VOICES.
    ollama_voice: str = "tara"
    # Where synthesized .wav files are written.
    voice_output_dir: str = "voice_output"
    # How long those files are kept. They are only useful until the client
    # has fetched them, and at roughly 1.5 MB per 30s answer the directory
    # would otherwise grow without limit. 0 keeps everything.
    voice_retention_hours: int = 24
    # How many chunks of one answer TextToVoice.synthesize() sends to Ollama
    # at once - independent calls, so concurrency here shortens multi-chunk
    # answers instead of paying each chunk's generation time in series.
    voice_max_workers: int = 4
    # Rough token budget (~4 chars/token) for the verbatim history
    # OllamaClient.chat() sends before folding older turns into a summary,
    # and how many of the most recent messages stay verbatim once it does.
    # Mirrors OllamaClient's own DEFAULT_MAX_HISTORY_TOKENS/
    # DEFAULT_KEEP_RECENT_MESSAGES - tune to the model's actual num_ctx.
    ollama_max_history_tokens: int = 3000
    ollama_keep_recent_messages: int = 10

    # --- Metric graphs (ai_server_graph.py) ---
    # Where rendered .png graphs are written.
    graph_output_dir: str = "graph_output"
    # How long those files are kept - same reasoning as voice_retention_hours.
    # 0 keeps everything.
    graph_retention_hours: int = 24

    # --- AI agent HTTP API (ai_agent_server.py) ---
    ai_agent_host: str = "0.0.0.0"
    ai_agent_port: int = 8100
    # Where accepted requests get their answer POSTed back to once they're
    # done - see ai_agent_server.py's module docstring for the contract.
    ai_client_callback_url: str = "http://127.0.0.1:9100/api/response"
    ai_client_callback_timeout: int = 10
    # Base URL used to build the audio_url in callbacks. Empty (the default)
    # takes it from the address the client used to reach the server, which
    # needs no configuration and is right on a normal LAN - ai_agent_host is
    # a bind address ("0.0.0.0") and cannot be used for this. Set it to pin
    # the link where that address is wrong or untrusted: behind a proxy
    # terminating on a different name, or where the Host header shouldn't be
    # echoed into a URL the callback receiver will fetch.
    ai_agent_public_url: str = ""

    # --- Scheduling ---
    interval_seconds: int = 600
    # The ClickHouse collector samples the 'power' metric on its own faster
    # cadence, into metering_power - see CollectorClickhouse. Ignored by the
    # legacy MariaDB Collector.py, which has a single interval.
    power_interval_seconds: int = 10

    # IANA zone name used only to decide what "today" means (today_outside()'s
    # 08:00-21:00 window) - metering.mdatatime stays UTC, as collected (see
    # Collector._tick_timestamp). Needed because the process computing "today"
    # may itself be running in a container with no local timezone, e.g. UTC.
    local_timezone: str = "America/Los_Angeles"

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

    def __post_init__(self) -> None:
        # Resolve per-model URL overrides now, so every other reader
        # (constructed directly or via from_env()) can just use
        # ollama_url_1/_2/_text_to_voice without re-checking for "".
        self.ollama_url_1 = self.ollama_url_1 or self.ollama_url
        self.ollama_url_2 = self.ollama_url_2 or self.ollama_url
        self.ollama_url_text_to_voice = self.ollama_url_text_to_voice or self.ollama_url

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
            db_user_readonly=text("DB_USER_READONLY", defaults.db_user_readonly),
            db_password_readonly=text("DB_PASSWORD_READONLY", defaults.db_password_readonly),
            ch_host=text("CH_HOST", defaults.ch_host),
            ch_port=number("CH_PORT", defaults.ch_port),
            ch_user=text("CH_USER", defaults.ch_user),
            ch_password=text("CH_PASSWORD", defaults.ch_password),
            ch_database=text("CH_DATABASE", defaults.ch_database),
            ch_secure=flag("CH_SECURE", defaults.ch_secure),
            ch_user_readonly=text("CH_USER_READONLY", defaults.ch_user_readonly),
            ch_password_readonly=text(
                "CH_PASSWORD_READONLY", defaults.ch_password_readonly
            ),
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
            # "" (not defaults.ollama_url_1, which __post_init__ already
            # resolved against the class default) so an unset override still
            # falls back to the OLLAMA_URL value read just above, not the
            # hardcoded class default.
            ollama_url_1=text("OLLAMA_URL_1", ""),
            ollama_url_2=text("OLLAMA_URL_2", ""),
            ollama_url_text_to_voice=text("OLLAMA_URL_TEXT_TO_VOICE", ""),
            ollama_model_1=text("OLLAMA_MODEL_1", defaults.ollama_model_1),
            ollama_model_2=text("OLLAMA_MODEL_2", defaults.ollama_model_2),
            ollama_model_text_to_voice=text(
                "OLLAMA_MODEL_TEXT_TO_VOICE", defaults.ollama_model_text_to_voice
            ),
            ollama_voice=text("OLLAMA_VOICE", defaults.ollama_voice),
            voice_output_dir=text("VOICE_OUTPUT_DIR", defaults.voice_output_dir),
            voice_retention_hours=number(
                "VOICE_RETENTION_HOURS", defaults.voice_retention_hours
            ),
            voice_max_workers=number("VOICE_MAX_WORKERS", defaults.voice_max_workers),
            ollama_max_history_tokens=number(
                "OLLAMA_MAX_HISTORY_TOKENS", defaults.ollama_max_history_tokens
            ),
            ollama_keep_recent_messages=number(
                "OLLAMA_KEEP_RECENT_MESSAGES", defaults.ollama_keep_recent_messages
            ),
            graph_output_dir=text("GRAPH_OUTPUT_DIR", defaults.graph_output_dir),
            graph_retention_hours=number(
                "GRAPH_RETENTION_HOURS", defaults.graph_retention_hours
            ),
            ai_agent_host=text("AI_AGENT_HOST", defaults.ai_agent_host),
            ai_agent_port=number("AI_AGENT_PORT", defaults.ai_agent_port),
            ai_client_callback_url=text(
                "AI_CLIENT_CALLBACK_URL", defaults.ai_client_callback_url
            ),
            ai_client_callback_timeout=number(
                "AI_CLIENT_CALLBACK_TIMEOUT", defaults.ai_client_callback_timeout
            ),
            ai_agent_public_url=text("AI_AGENT_PUBLIC_URL", defaults.ai_agent_public_url),
            interval_seconds=number("INTERVAL_SECONDS", defaults.interval_seconds),
            power_interval_seconds=number(
                "POWER_INTERVAL_SECONDS", defaults.power_interval_seconds
            ),
            local_timezone=text("LOCAL_TIMEZONE", defaults.local_timezone),
            dry_run=flag("DRY_RUN", defaults.dry_run),
            default_location_id=number("DEFAULT_LOCATION_ID", defaults.default_location_id),
            default_location_name=text("DEFAULT_LOCATION_NAME", defaults.default_location_name),
            log_level=text("LOG_LEVEL", defaults.log_level),
            log_file=text("LOG_FILE", defaults.log_file),
            log_backup_count=number("LOG_BACKUP_COUNT", defaults.log_backup_count),
        )
