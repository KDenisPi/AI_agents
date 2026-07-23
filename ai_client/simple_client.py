"""
Stand-in for the real ai_client, until it exists - just enough to exercise
ai_agent_server.py's accept-then-callback contract by hand: fires
GET /api/current at the agent, and listens for the POST /api/response
callback that follows, printing whatever comes back.

Run:
    python ai_client/simple_client.py
    python ai_client/simple_client.py --agent-url http://192.168.1.57:8100
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


def request_current(agent_url: str) -> None:
    request_id = make_request_id()
    params = {"request_id": request_id}
    try:
        response = requests.get(f"{agent_url}/api/current", params=params, timeout=5)
        print(f">> GET /api/current?request_id={request_id} -> {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f">> GET /api/current?request_id={request_id} -> failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal test client for ai_agent_server.py")
    parser.add_argument("--host", default="127.0.0.1", help="host to listen on for callbacks")
    parser.add_argument("--port", type=int, default=9100, help="port to listen on for callbacks")
    parser.add_argument(
        "--agent-url", default="http://127.0.0.1:8100", help="ai_agent_server.py base URL"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    server = HTTPServer((args.host, args.port), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Listening for callbacks on http://{args.host}:{args.port}/api/response")
    print(f"Sending requests to {args.agent_url}\n")

    try:
        while True:
            input("Press Enter to send GET /api/current (Ctrl+C to quit)... ")
            request_current(args.agent_url)
    except (KeyboardInterrupt, EOFError):
        print("\nStopping")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
