"""
Client for a Hubitat Elevation hub's Maker API app - the built-in local
REST integration for reading device state and sending commands, no cloud
required.

Endpoint shape: http://{hub_ip}/apps/api/{app_id}/{path}?access_token={token}
"""

import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hubitat")


class HubitatClient:
    """
    Usage:
        hub = HubitatClient("192.168.1.50", app_id="1", access_token="xxxx")
        devices = hub.list_devices()
        device = hub.get_device(device_id)
        is_on = hub.get_attribute(device_id, "switch")
        hub.send_command(device_id, "on")
        hub.send_command(device_id, "setLevel", 50)
    """

    def __init__(self, hub_ip: str, app_id: str, access_token: str, timeout: float = 10):
        self._base_url = f"http://{hub_ip}/apps/api/{app_id}"
        self._access_token = access_token
        self.timeout = timeout

    def _get(self, path: str):
        """Returns the decoded JSON body, or None if the hub is unreachable
        or answers with an error."""
        url = f"{self._base_url}/{path}"
        try:
            response = requests.get(
                url,
                params={"access_token": self._access_token},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Request to %s failed: %s", url, e)
            return None
        except ValueError as e:
            logger.error("Bad JSON from %s: %s", url, e)
            return None

    def list_devices(self) -> list[dict]:
        """All devices known to this Maker API instance (id, name, label, type).

        Empty list if the hub could not be reached."""
        return self._get("devices") or []

    def get_device(self, device_id: str) -> dict:
        """Full device info: current attributes, capabilities, and available commands.

        Empty dict if the hub could not be reached."""
        return self._get(f"devices/{device_id}") or {}

    def get_attribute(self, device_id: str, attribute: str):
        """Current value of a single attribute (e.g. 'switch', 'motion', 'temperature').

        None if the device could not be read or has no such attribute."""
        device = self.get_device(device_id)
        if not device:
            return None
        for attr in device.get("attributes", []):
            if attr.get("name") == attribute:
                return attr.get("currentValue")
        logger.error("Device %s has no attribute '%s'", device_id, attribute)
        return None

    def send_command(self, device_id: str, command: str, value=None):
        """Send a command to a device, optionally with a secondary value (e.g. setLevel, 50).

        None if the hub could not be reached."""
        path = f"devices/{device_id}/{command}"
        if value is not None:
            path += f"/{value}"
        return self._get(path)


def demo():
    hub = HubitatClient("192.168.1.242", app_id="1", access_token="3b72adce-e0b3-43ae-8d1e-549bf82355d5")

    print("Devices:", hub.list_devices())

    #device_id = "1622"
    #print("Full device info:", hub.get_device(device_id))
    #print("Switch state:", hub.get_attribute(device_id, "switch"))

    #hub.send_command(device_id, "on")
    #hub.send_command(device_id, "setLevel", 50)


if __name__ == "__main__":
    demo()
