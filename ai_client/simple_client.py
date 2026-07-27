"""
Stand-in for the real ai_client, until it exists - just enough to exercise
ai_agent_server.py's accept-then-callback contract by hand: prints a
numbered menu of the agent's endpoints, fires a GET for whichever one is
picked, and listens for the POST /api/response callback that follows,
printing whatever comes back.

Run:
    python ai_client/simple_client.py
    python ai_client/simple_client.py --agent-url http://192.168.1.57:8100
    python ai_client/simple_client.py --voice          # spoken answer too
    python ai_client/simple_client.py --voice leo      # in a named voice
    python ai_client/simple_client.py --graph          # rendered graphs too

With --voice the callback carries an audio_url alongside the text - a link
to a WAV on the agent, which is pruned after the agent's retention window,
so fetch it rather than keeping the link. --graph does the same for a
graphs field (one link per metric), and only applies to endpoints that
support it.
"""

import argparse
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

logger = logging.getLogger("ai-client")

# One id per process is enough to tell "which client" apart in a shared log
# - request_id itself only needs to be unique per client, via the timestamp.
_CLIENT_ID = uuid.uuid4().hex[:8]


def make_request_id() -> str:
    return f"{int(time.time() * 1000)}-{_CLIENT_ID}"


class CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/response":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body.decode(errors="replace")}
        print(f"\n<< POST /api/response: {json.dumps(payload, indent=2)}")

        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet - the payload above is all we care about here


def _get(agent_url: str, path: str, params: dict[str, str]) -> None:
    query = "&".join(f"{k}={v}" if v else k for k, v in params.items())
    try:
        response = requests.get(f"{agent_url}{path}", params=params, timeout=5)
        print(f">> GET {path}?{query} -> {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f">> GET {path}?{query} -> failed: {e}")


def request_current(agent_url: str, voice: str | None = None) -> None:
    params = {"request_id": make_request_id()}
    if voice is not None:
        # "" sends a bare "voice=", which the server reads the same as the
        # "&voice" flag its docstring documents: present, so speak the
        # answer, with no particular voice named.
        params["voice"] = voice
    _get(agent_url, "/api/current", params)


def request_outside_today(agent_url: str, voice: str | None = None, graph: bool = False) -> None:
    params = {"request_id": make_request_id()}
    if voice is not None:
        params["voice"] = voice
    if graph:
        params["graph"] = ""
    _get(agent_url, "/api/outside_today", params)


# Menu shown by main() - label plus the call it sends, in selection order.
# Add a tuple here for any new ai_agent_server.py endpoint.
def _menu(args: argparse.Namespace) -> list[tuple[str, callable]]:
    return [
        (
            "Current summary (/api/current)",
            lambda: request_current(args.agent_url, args.voice)
        ),
        (
            "Outside today, 08:00-21:00 (/api/outside_today)",
            lambda: request_outside_today(args.agent_url, args.voice, args.graph),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal test client for ai_agent_server.py")
    parser.add_argument("--host", default="127.0.0.1", help="host to listen on for callbacks")
    parser.add_argument("--port", type=int, default=9100, help="port to listen on for callbacks")
    parser.add_argument(
        "--agent-url", default="http://127.0.0.1:8100", help="ai_agent_server.py base URL"
    )
    # Optional value, mirroring the server's own "&voice[=<name>]": bare
    # --voice asks for audio in the configured voice, --voice leo names one.
    # Absent, no voice parameter is sent at all and the answer is text only.
    parser.add_argument(
        "--voice",
        nargs="?",
        const="",
        default=None,
        metavar="NAME",
        help="also ask for a spoken answer; optionally name a voice, e.g. --voice leo",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also ask for a rendered graph per metric, where the endpoint supports it",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    server = HTTPServer((args.host, args.port), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Listening for callbacks on http://{args.host}:{args.port}/api/response")
    print(f"Sending requests to {args.agent_url}")
    if args.voice is not None:
        named = f" ({args.voice})" if args.voice else ""
        print(f"Asking for spoken answers{named} - audio_url will be in the callback")
    if args.graph:
        print("Asking for graphs where supported - graphs will be in the callback")

    options = _menu(args)

    def print_menu() -> None:
        print("\nChoose a request to send:")
        for i, (label, _) in enumerate(options, start=1):
            print(f"  {i}. {label}")
        print("  q. quit")

    try:
        while True:
            print_menu()
            choice = input("> ").strip().lower()
            if choice in ("q", "quit"):
                break
            if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
                print(f"Not a valid option - enter a number 1-{len(options)}, or q.")
                continue
            options[int(choice) - 1][1]()
    except (KeyboardInterrupt, EOFError):
        print("\nStopping")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
