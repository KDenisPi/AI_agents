"""
Local weather station as a reading source, reached through its MCP server
(McpWeather, served by McpMultiHttpServer) over the SSE transport.

Cycles are ten minutes apart, so there is no session worth keeping alive
between them - each collect() opens a persistent client, calls weather_info,
and closes it. That also means a restarted MCP server heals by itself on the
next cycle.
"""

import json
import logging
from datetime import datetime

from Config import Config
from McpClientPersistent import MCPClientPersistent
from Reading import Reading
from Source import Source, to_float

logger = logging.getLogger("weather-mcp-source")

WEATHER_TOOL = "weather_info"

# Station measurement key -> canonical metric name. 'altitude' is left out
# on purpose: the station derives it from pressure, it is not a measurement.
MEASUREMENT_METRICS = {
    "temperature": "temperature",
    "humidity": "humidity",
    "pressure": "pressure",
    "luximity": "illuminance",
    "co2": "CO2",
    "tvoc": "TVOC",
}


class WeatherMcpSource(Source):
    """The whole station is one sensor row, at a location of its own -
    its ids come from config since MCP exposes no device identity."""

    def __init__(self, config: Config):
        self._config = config

    @property
    def name(self) -> str:
        return "weather-mcp"

    async def collect(self, taken_at: datetime) -> list[Reading]:
        payload = await self._call_weather_tool()
        if payload is None:
            return []

        measurement = payload.get("measurement", {})
        if not measurement:
            logger.warning("weather_info returned no measurement block")
            return []

        readings = []
        for key, metric in MEASUREMENT_METRICS.items():
            value = to_float(measurement.get(key))
            if value is None:
                continue
            readings.append(
                Reading(
                    sensor_id=self._config.weather_sensor_id,
                    sensor_name=self._config.weather_sensor_name,
                    location_id=self._config.weather_location_id,
                    location_name=self._config.weather_location_name,
                    outside=self._config.weather_location_outside,
                    metric=metric,
                    value=value,
                    taken_at=taken_at,
                )
            )
        return readings

    async def _call_weather_tool(self) -> dict | None:
        """Decoded station JSON, or None if the server or the tool failed."""
        try:
            async with MCPClientPersistent(self._config.weather_mcp_url) as client:
                text = await client.call_tool(WEATHER_TOOL, {})
        except Exception as e:
            # Covers connection failures and tool-reported errors, which
            # call_tool raises as RuntimeError.
            logger.error("MCP call to %s failed: %s", self._config.weather_mcp_url, e)
            return None

        try:
            return json.loads(text)
        except ValueError:
            # The tool hands back whatever the station's HTTP endpoint said;
            # a non-JSON body means the station itself is unhappy.
            logger.error("%s returned non-JSON payload: %.200s", WEATHER_TOOL, text)
            return None
