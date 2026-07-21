"""
Common shape for everything the collector polls.

A source owns its transport and its upstream's quirks, and hands back a
flat list of Readings. It must not raise for an unreachable upstream -
one dead source should cost that cycle its readings, not the whole cycle.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from Reading import Reading


class Source(ABC):
    """Poll one upstream for the current values of everything it exposes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label used in log lines."""

    @abstractmethod
    async def collect(self, taken_at: datetime) -> list[Reading]:
        """Readings as of now, all stamped taken_at.

        Empty list if the upstream is unreachable or has nothing to report."""


def to_float(value) -> float | None:
    """Upstreams report numbers as strings, nulls, and occasionally junk -
    None means 'not a usable measurement', and the caller skips it."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
