#!/usr/bin/env bash
#
# start_all.sh - bring up everything this project needs after a server
# restart:
#   1. sanity-check the docker daemon
#   2. launch the llama.cpp server (Orpheus TTS model) detached
#   3. warm up the Ollama models used interactively
#   4. (re)create the 4 docker containers used by run_steps.txt
#
# Models are started first and the app containers last, since ai-server /
# mcp-server talk to Ollama and the llama-server on startup.
#
# Idempotent: safe to re-run. Anything already running is left alone.
#
# All paths are resolved relative to this script's own location, so it can be
# invoked from anywhere (cron, a systemd unit, an interactive shell, ...).

set -uo pipefail

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/denis/projects/llama.cpp}"
LLAMA_MODEL="${LLAMA_MODEL:-/opt/models/gguf/orpheus-3b-0.1-ft-q4_k_m.gguf}"
LLAMA_PORT="${LLAMA_PORT:-1234}"
LLAMA_PIDFILE="$LLAMA_CPP_DIR/logs/llama.pid"
LLAMA_LOG="$LLAMA_CPP_DIR/logs/llama.log"

OLLAMA_MODELS=("llama3.1:8b" "qwen3.6:latest")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$PROJECT_DIR/logs/clickhouse" "$PROJECT_DIR/data/clickhouse" "$LLAMA_CPP_DIR/logs"

# ---------------------------------------------------------------------------
# 1. Docker daemon
# ---------------------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
    log "docker daemon not reachable, trying 'sudo systemctl start docker' (may prompt for a password)"
    sudo systemctl start docker
    for _ in $(seq 1 15); do
        docker info >/dev/null 2>&1 && break
        sleep 1
    done
fi
if ! docker info >/dev/null 2>&1; then
    log "ERROR: docker daemon still not available, aborting"
    exit 1
fi

container_state() {
    # prints: running | stopped | absent
    local name="$1" status
    status="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)" || { echo absent; return; }
    [[ "$status" == "running" ]] && echo running || echo stopped
}

wait_for_http() {
    local url="$1" timeout="${2:-30}"
    for ((i = 0; i < timeout; i++)); do
        curl -fsS "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# Mirrors run_steps.txt: drop a stopped container before recreating it.
start_container() {
    local name="$1"
    shift
    case "$(container_state "$name")" in
        running) log "container '$name' already running, skipping" ; return 0 ;;
        stopped) log "removing stopped container '$name'" ; docker rm "$name" >/dev/null ;;
    esac
    log "starting container '$name'"
    "$@" >/dev/null
}

# ---------------------------------------------------------------------------
# 2. llama.cpp server (Orpheus TTS model)
# ---------------------------------------------------------------------------
if [[ -f "$LLAMA_PIDFILE" ]] && kill -0 "$(cat "$LLAMA_PIDFILE")" 2>/dev/null; then
    log "llama-server already running (pid $(cat "$LLAMA_PIDFILE")), skipping"
elif ss -ltn 2>/dev/null | grep -q ":$LLAMA_PORT "; then
    log "something is already listening on port $LLAMA_PORT, skipping llama-server start"
else
    log "starting llama-server on port $LLAMA_PORT"
    (
        cd "$LLAMA_CPP_DIR" || exit 1
        setsid ./build/bin/llama-server \
            -m "$LLAMA_MODEL" \
            -c 8192 -ngl 99 --host 0.0.0.0 --port "$LLAMA_PORT" \
            -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
            -np 4 --no-kv-unified \
            >"$LLAMA_LOG" 2>&1 </dev/null &
        echo $! >"$LLAMA_PIDFILE"
        disown
    )
    sleep 2
    log "llama-server pid $(cat "$LLAMA_PIDFILE" 2>/dev/null || echo '?')"
fi

# ---------------------------------------------------------------------------
# 3. Ollama - make sure the service is up, then warm the models
# ---------------------------------------------------------------------------
if ! systemctl is-active --quiet ollama; then
    log "ollama service not active, trying 'sudo systemctl start ollama' (may prompt for a password)"
    sudo systemctl start ollama
fi
for _ in $(seq 1 30); do
    curl -fsS http://localhost:11434 >/dev/null 2>&1 && break
    sleep 1
done

for model in "${OLLAMA_MODELS[@]}"; do
    log "warming up ollama model '$model'"
    ollama run "$model" "" >/dev/null 2>&1 || log "WARNING: warmup failed for $model"
done

# ---------------------------------------------------------------------------
# 4. ClickHouse first, then the app containers that depend on it/the models
# ---------------------------------------------------------------------------
start_container iot-clickhouse-server \
    docker run -d --name=iot-clickhouse-server \
        -p 8123:8123 -p 9000:9000 --ulimit nofile=262144:262144 \
        -v "$PROJECT_DIR/logs/clickhouse:/var/log/clickhouse-server" \
        -v "$PROJECT_DIR/data/clickhouse:/var/lib/clickhouse" \
        clickhouse/clickhouse-server:latest

log "waiting for ClickHouse to accept connections..."
if wait_for_http "http://localhost:8123/ping" 60; then
    log "ClickHouse is up"
else
    log "WARNING: ClickHouse did not respond within 60s, continuing anyway"
fi

start_container mcp-server \
    docker run -v "$PROJECT_DIR:/app" -v "$PROJECT_DIR/logs:/app/logs:rw" \
        -u 1000:1000 -d -w /app --name mcp-server -p 8000:8000 \
        ai-agent:latest python -u McpMultiHttpServer.py

start_container data-collector \
    docker run -v "$PROJECT_DIR:/app" -v "$PROJECT_DIR/logs:/app/logs:rw" \
        -u 1000:1000 -d -w /app --name data-collector \
        ai-agent:latest python -u CollectorClickhouse.py

start_container ai-server \
    docker run -v "$PROJECT_DIR:/app" -v "$PROJECT_DIR/logs:/app/logs:rw" \
        -u 1000:1000 -w /app --name ai-server -d --gpus all -p 8100:8100 \
        ai-server-gpu:latest python -u ai_agent_server.py

log "start_all.sh complete"
