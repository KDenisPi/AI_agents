"""
The unit of data every source produces, plus the canonical metric names.

A source's job is to turn whatever its upstream speaks into Readings; the
database layer knows nothing else. Metric names here must match the seed
rows in db/weather.sql - METRIC_IDS is the mapping between the two.
"""

from dataclasses import dataclass
from datetime import datetime

METRIC_IDS = {
    "temperature": 1,
    "humidity": 2,
    "CO2": 3,
    "TVOC": 4,
    "battery": 5,
    "pressure": 6,
    "illuminance": 7,
}


@dataclass(frozen=True)
class Reading:
    """One measurement, carrying enough context to auto-register its sensor.

    location/sensor fields are only used the first time a given id is seen -
    later cycles insert the metering row alone."""

    sensor_id: int
    sensor_name: str
    location_id: int
    location_name: str
    outside: bool
    metric: str
    value: float
    taken_at: datetime

    @property
    def metric_id(self) -> int:
        return METRIC_IDS[self.metric]
