"""
Client for a Hubitat Elevation hub's Maker API app - the built-in local
REST integration for reading device state and sending commands, no cloud
required.

Endpoint shape: http://{hub_ip}/apps/api/{app_id}/{path}?access_token={token}

Device endpoints

Get all devices
http://192.168.1.214/apps/api/7/devices?access_token={token}
Get all devices (full details)
http://192.168.1.214/apps/api/7/devices/all?access_token={token}
Get device info (replace [Device ID] with actual subscribed device id)
http://192.168.1.214/apps/api/7/devices/[Device ID]?access_token={token}
Get device data section from Device Info tab (replace [Device ID] with actual subscribed device id)
http://192.168.1.214/apps/api/7/devices/deviceData/[Device ID]?access_token={token}
Get device event history (replace [Device ID] with actual subscribed device id)
http://192.168.1.214/apps/api/7/devices/[Device ID]/events?access_token={token}
Get device commands (replace [Device ID] with actual subscribed device id)
http://192.168.1.214/apps/api/7/devices/[Device ID]/commands?access_token={token}
Get device capabilities (replace [Device ID] with actual subscribed device id)
http://192.168.1.214/apps/api/7/devices/[Device ID]/capabilities?access_token={token}
Get device attribute (replace [Device ID] with actual subscribed device id and [Attribute] with a supported device attribute)
http://192.168.1.214/apps/api/7/devices/[Device ID]/attribute/[Attribute]?access_token={token}
Send device command (replace [Device ID] with actual subscribed device id and [Command] with a supported command.  Supports optional [Secondary value])
http://192.168.1.214/apps/api/7/devices/[Device ID]/[Command]/[Secondary value]?access_token={token}
Set device label (replace [Device ID] with actual subscribed device id and [Label] with new label)
http://192.168.1.214/apps/api/7/devices/[Device ID]/setLabel?label=[Label]&access_token={token}
Set device driver (replace [Device ID] with actual subscribed device id and [Driver Namespace]/[Driver Name] with actual names)
http://192.168.1.214/apps/api/7/devices/[Device ID]/setDriver?namespace=[Driver Namespace]&name=[Driver Name]&access_token={token}
Delete device (replace [Device ID] with actual subscribed device id)
http://192.168.1.214/apps/api/7/devices/[Device ID]/deleteDevice?access_token={token}

Hub variable endpoints

Get a list of hub variables
http://192.168.1.214/apps/api/7/hubvariables?access_token={token}
Get hub variable (replace [Variable Name] with actual variable name)
http://192.168.1.214/apps/api/7/hubvariables/[Variable Name]?access_token={token}
Set hub variable (replace [Variable Name] with actual variable name and [Value] with value to set)
http://192.168.1.214/apps/api/7/hubvariables/[Variable Name]/[Value]?access_token={token}

Mode endpoints

Get modes list
http://192.168.1.214/apps/api/7/modes?access_token={token}
Set mode (replace [Mode ID] with actual mode id)
http://192.168.1.214/apps/api/7/modes/[Mode ID]?access_token={token}

Hubitat Safety Monitor endpoints

Get HSM status
http://192.168.1.214/apps/api/7/hsm?access_token={token}
Set HSM status (replace [HSM Status] with an actual value)
http://192.168.1.214/apps/api/7/hsm/[HSM Status]?access_token={token}

Room endpoints

Get room list
http://192.168.1.214/apps/api/7/rooms?access_token={token}
Get room details (replace [Room ID] with actual room id)
http://192.168.1.214/apps/api/7/room/select/[Room ID]?access_token={token}
Insert room (requires name and optional deviceIds parameters, ex. ?name=Kitchen&deviceIds=123,456)
http://192.168.1.214/apps/api/7/room/insert?name=[Room Name]&deviceIds=[Device ID list]&access_token={token}
Update room (requires id, name, and deviceIds parameters, ex. ?id=789&name=New%20Name&deviceIds=123,456)
http://192.168.1.214/apps/api/7/room/update/[Room ID]?name=[Room Name]&deviceIds=[Device ID list]&access_token={token}
Delete room (replace [Room ID] with actual room id)
http://192.168.1.214/apps/api/7/room/delete/[Room ID]?access_token={token}

Other/miscellaneous endpoints

Send POST URL (replace [URL] with actual URL to send POST to (URL encoded))
http://192.168.1.214/apps/api/7/postURL/[URL]?access_token=r


"""

import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hubitat")


class HubitatClient:
    """
    Usage:
        hub = HubitatClient("192.168.1.50", app_id="7", access_token="xxxx")
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

    def list_devices_all(self) -> list[dict]:
        """Get all devices (full details).

        Empty list if the hub could not be reached."""
        return self._get("devices/all") or []

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
    hub = HubitatClient("192.168.1.214", app_id="7", access_token="3b72adce-e0b3-43ae-8d1e-549bf82355d5")

    devices = hub.list_devices()
    print("Devices:", devices)
    devices_all = hub.list_devices_all()
    print("Devices All info:", devices_all)

    if len(devices) > 0:
        device_id = devices[0]["id"]
        print("Full device info:", hub.get_device(device_id))
        #print("Switch state:", hub.get_attribute(device_id, "switch"))

        #hub.send_command(device_id, "on")
        #hub.send_command(device_id, "setLevel", 50)


if __name__ == "__main__":
    demo()
