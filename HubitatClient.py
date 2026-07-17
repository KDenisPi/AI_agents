"""
Client for a Hubitat Elevation hub's Maker API app - the built-in local
REST integration for reading device state and sending commands, no cloud
required.

Endpoint shape: http://{hub_ip}/apps/api/{app_id}/{path}?access_token={token}
"""

import requests


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
        response = requests.get(
            f"{self._base_url}/{path}",
            params={"access_token": self._access_token},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_devices(self) -> list[dict]:
        """All devices known to this Maker API instance (id, name, label, type)."""
        return self._get("devices")

    def get_device(self, device_id: str) -> dict:
        """Full device info: current attributes, capabilities, and available commands."""
        return self._get(f"devices/{device_id}")

    def get_attribute(self, device_id: str, attribute: str):
        """Current value of a single attribute (e.g. 'switch', 'motion', 'temperature')."""
        device = self.get_device(device_id)
        for attr in device.get("attributes", []):
            if attr.get("name") == attribute:
                return attr.get("currentValue")
        raise KeyError(f"Device {device_id} has no attribute '{attribute}'")

    def send_command(self, device_id: str, command: str, value=None):
        """Send a command to a device, optionally with a secondary value (e.g. setLevel, 50)."""
        path = f"devices/{device_id}/{command}"
        if value is not None:
            path += f"/{value}"
        return self._get(path)


def demo():
    hub = HubitatClient("192.168.1.50", app_id="1", access_token="<your-token>")

    print("Devices:", hub.list_devices())

    device_id = "1622"
    print("Full device info:", hub.get_device(device_id))
    print("Switch state:", hub.get_attribute(device_id, "switch"))

    hub.send_command(device_id, "on")
    hub.send_command(device_id, "setLevel", 50)


if __name__ == "__main__":
    demo()
