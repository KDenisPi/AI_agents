#!/usr/bin/env bash
# End-to-end smoke test for McpMultiHttpServer.py (weather MCP server over SSE).
#
# Starts the server, opens an SSE session, and drives the MCP JSON-RPC
# handshake (initialize -> notifications/initialized -> tools/list ->
# tools/call weather_info) purely with curl. Prints PASS/FAIL per step.
#
# Usage:
#   ./test_mcp_weather_server.sh [venv_path]
#
# venv_path defaults to ../venv3.12 (relative to this script) if present,
# otherwise falls back to whatever "python3" resolves to on PATH.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-/home/deniskudia/sources/lunal/venv3.12}"
PYTHON="$VENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

PORT=8000
HOST="http://localhost:$PORT"
WORKDIR="$(mktemp -d)"
SERVER_LOG="$WORKDIR/server.log"
SSE_LOG="$WORKDIR/sse_stream.log"
RUN_LOG="$SCRIPT_DIR/test_mcp_weather_server.log"
FAIL=0

# Save the full run output (PASS/FAIL summary + full server responses) to a
# log file, in addition to printing it to the screen.
exec > >(tee "$RUN_LOG") 2>&1

echo "$SERVER_LOG"

cleanup() {
  [[ -n "${SSE_PID:-}" ]] && kill "$SSE_PID" 2>/dev/null
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null
  # Only wait on the processes we started ourselves - a bare `wait` would
  # also block on the `tee` job from the `exec > >(tee ...)` redirect above,
  # which never exits until our own stdout closes (deadlock on script exit).
  [[ -n "${SSE_PID:-}" ]] && wait "$SSE_PID" 2>/dev/null
  [[ -n "${SERVER_PID:-}" ]] && wait "$SERVER_PID" 2>/dev/null
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

check() {
  local desc="$1" got="$2" want="$3"
  if [[ "$got" == *"$want"* ]]; then
    echo "PASS: $desc"
  else
    echo "FAIL: $desc (expected to contain: $want, got: $got)"
    FAIL=1
  fi
}

echo "== Starting server (pid will be logged) =="
(cd "$SCRIPT_DIR" && "$PYTHON" McpMultiHttpServer.py) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 20); do
  curl -sf -o /dev/null "$HOST/weather/sse" --max-time 1 && break
  sleep 0.5
done

echo "== Opening SSE session =="
curl -sN "$HOST/weather/sse" > "$SSE_LOG" 2>&1 &
SSE_PID=$!
sleep 1

ENDPOINT=$(grep "^data:" "$SSE_LOG" | head -1 | sed 's/^data: //' | tr -d '\r')
if [[ -z "$ENDPOINT" ]]; then
  echo "FAIL: did not receive an 'endpoint' event from the server"
  echo "--- server log ---"; cat "$SERVER_LOG"
  exit 1
fi
BASE="$HOST$ENDPOINT"
echo "Using message endpoint: $BASE"

post() {
  curl -s -w "\n%{http_code}" -X POST "$BASE" -H "Content-Type: application/json" -d "$1"
}

resp="$(post '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke-test","version":"0.0.1"}}}')"
check "initialize accepted" "$resp" "202"

post '{"jsonrpc":"2.0","method":"notifications/initialized"}' > /dev/null

resp="$(post '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')"
check "tools/list accepted" "$resp" "202"

resp="$(post '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"weather_info","arguments":{}}}')"
check "tools/call accepted" "$resp" "202"

sleep 1
STREAM="$(tr -d '\r' < "$SSE_LOG")"
check "initialize result on SSE stream" "$STREAM" '"id":1,"result":{"protocolVersion"'
check "tools/list result on SSE stream" "$STREAM" '"id":2,"result":{"tools":[{"name":"weather_info"'
check "tools/call result on SSE stream" "$STREAM" '"id":3,"result":{"content":[{"type":"text","text":"Weather info:'

echo
echo "== Full messages received from server (SSE stream) =="
awk '/^event: message$/{getline; sub(/^data: /, ""); print}' <<< "$STREAM" | \
while IFS= read -r line; do
  jq . <<< "$line" 2>/dev/null || echo "$line"
  echo
done

echo "== Simulating client disconnect =="
kill "$SSE_PID" 2>/dev/null
SSE_PID=""
sleep 1
if grep -q "Traceback" "$SERVER_LOG"; then
  echo "FAIL: server logged a traceback after client disconnect"
  FAIL=1
else
  echo "PASS: no server-side traceback after client disconnect"
fi

echo
if [[ "$FAIL" -eq 0 ]]; then
  echo "ALL CHECKS PASSED"
else
  echo "SOME CHECKS FAILED"
  echo "--- server log ---"
  cat "$SERVER_LOG"
fi

cp "$SERVER_LOG" "./server.log"
echo "Full run output saved to: $RUN_LOG"
exit "$FAIL"
