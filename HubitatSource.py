"""
Hubitat Elevation as a reading source, over the Maker API (HTTP).

One devices/all call returns every subscribed device with its current
attributes, so a cycle costs a single request no matter how many sensors
are paired.
"""

import asyncio
import logging
from datetime import datetime

from Config import Config
from HubitatClient import HubitatClient
from Reading import Reading
from Source import Source, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hubitat-source")

# Hubitat attribute name -> canonical metric name. Attributes not listed
# here are ignored, which also filters the dataType/values keys that
# devices/all mixes into the attribute map.
ATTRIBUTE_METRICS = {
    "temperature": "temperature",
    "humidity": "humidity",
    "battery": "battery",
    "carbonDioxide": "CO2",
    "pressure": "pressure",
    "illuminance": "illuminance",
}


class HubitatSource(Source):
    """Every device the Maker API instance exposes becomes a sensor row,
    keyed by its Hubitat device id."""

    def __init__(self, config: Config):
        self._config = config
        self._client = HubitatClient(
            config.hubitat_ip,
            app_id=config.hubitat_app_id,
            access_token=config.hubitat_token,
        )

    @property
    def name(self) -> str:
        return "hubitat"

    async def collect(self, taken_at: datetime) -> list[Reading]:
        # HubitatClient uses blocking requests - keep it off the event loop.
        devices = await asyncio.to_thread(self._client.list_devices_all)
        readings = []
        for device in devices:
            readings.extend(self._device_readings(device, taken_at))
        return readings

    def _device_readings(self, device: dict, taken_at: datetime) -> list[Reading]:
        sensor_id = to_float(device.get("id"))
        if sensor_id is None:
            # sensor.sensorid is an INT; a hub with non-numeric device ids
            # would need a different id scheme than "use Hubitat's".
            logger.warning("Skipping device with non-numeric id: %r", device.get("id"))
            return []

        name = device.get("label") or device.get("name") or f"device-{int(sensor_id)}"
        room = device.get("room")
        room_id = device.get("roomId")
        if room_id is None:
            room = self._config.default_location_name
            room_id = self._config.default_location_id

        readings = []
        for attribute, metric in ATTRIBUTE_METRICS.items():
            value = to_float(device.get("attributes", {}).get(attribute))
            if value is None:
                continue
            readings.append(
                Reading(
                    sensor_id=int(sensor_id),
                    sensor_name=name,
                    location_id=int(room_id),
                    location_name=room,
                    outside=_is_outside(room),
                    metric=metric,
                    value=value,
                    taken_at=taken_at,
                )
            )
        return readings


def _is_outside(room: str) -> bool:
    """Best guess at the location.outside flag from the room name. Only
    used when the location row is first created, so correcting it by hand
    afterwards sticks."""
    return any(word in room.lower() for word in ("outside", "outdoor"))
