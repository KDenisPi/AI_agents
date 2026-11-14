#!/usr/bin/env bash
#
# stop_all.sh - cleanly shut down everything started by start_all.sh, in
# exact reverse order, before a server shutdown/restart:
#   ai-server -> data-collector -> mcp-server -> wiki-server -> clickhouse ->
#   ollama models -> llama-servers (stage 2 SQL, then TTS)
#
# Docker containers are stopped (not removed) - start_all.sh removes stopped
# containers itself before recreating them, matching the original
# run_steps.txt "docker rm" + "docker run" flow.
#
# All paths are resolved relative to this script's own location.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/denis/projects/llama.cpp}"
LLAMA_PIDFILE="$LLAMA_CPP_DIR/logs/llama.pid"
SQL_LLAMA_PORT="${SQL_LLAMA_PORT:-1235}"
SQL_LLAMA_PIDFILE="${SQL_LLAMA_PIDFILE:-$LLAMA_CPP_DIR/logs/llama-sql.pid}"

#OLLAMA_MODELS=("llama3.1-16k:latest" "qwen3.6:latest")
OLLAMA_MODELS=("llama3.1-16k:latest")
# Exact reverse of start_all.sh's start order:
#   models -> clickhouse -> wiki-server -> mcp-server -> data-collector -> ai-server
# so: ai-server -> data-collector -> mcp-server -> wiki-server -> clickhouse -> models
#
# App containers run our own Python code, so they're stopped with SIGINT
# (== KeyboardInterrupt) to let their try/except cleanup run. ClickHouse is
# a stock image that shuts down cleanly on docker stop's default SIGTERM.
APP_CONTAINERS=(ai-server data-collector mcp-server wiki-server)
DB_CONTAINERS=(iot-clickhouse-server)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

stop_container() {
    local name="$1" signal="$2"
    local state
    state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)"
    if [[ -z "$state" ]]; then
        log "container '$name' does not exist"
    elif [[ "$state" == "running" ]]; then
        log "stopping container '$name' (signal $signal)"
        docker stop --signal "$signal" -t 30 "$name" >/dev/null
    else
        log "container '$name' already stopped"
    fi
}

# ---------------------------------------------------------------------------
# 1. Docker containers, reverse of start order
# ---------------------------------------------------------------------------
if docker info >/dev/null 2>&1; then
    for name in "${APP_CONTAINERS[@]}"; do
        stop_container "$name" SIGINT
    done
    for name in "${DB_CONTAINERS[@]}"; do
        stop_container "$name" SIGTERM
    done
else
    log "docker daemon not reachable, skipping container shutdown"
fi

# ---------------------------------------------------------------------------
# 2. Unload Ollama models to free GPU memory. The ollama service itself is a
#    systemd unit and is stopped by the OS shutdown sequence - not touched here.
# ---------------------------------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    running_models="$(ollama ps 2>/dev/null | awk 'NR>1{print $1}')"
    for model in "${OLLAMA_MODELS[@]}"; do
        if grep -qx "$model" <<<"$running_models"; then
            log "unloading ollama model '$model'"
            ollama stop "$model" >/dev/null 2>&1
        fi
    done
fi

# ---------------------------------------------------------------------------
# 3a. The wiki's stage 2 llama-server (started after the TTS one, so stopped
#     before it). Same pidfile-then-port fallback as the TTS block below.
# ---------------------------------------------------------------------------
sql_pid=""
if [[ -f "$SQL_LLAMA_PIDFILE" ]] && kill -0 "$(cat "$SQL_LLAMA_PIDFILE")" 2>/dev/null; then
    sql_pid="$(cat "$SQL_LLAMA_PIDFILE")"
else
    # no pidfile, or a stale one: whoever is listening on the port
    sql_pid="$(fuser "$SQL_LLAMA_PORT"/tcp 2>/dev/null | xargs)"
fi
if [[ -n "$sql_pid" ]]; then
    log "stopping stage 2 llama-server (pid $sql_pid)"
    kill -TERM $sql_pid 2>/dev/null
    for _ in $(seq 1 10); do
        kill -0 $sql_pid 2>/dev/null || break
        sleep 1
    done
    if kill -0 $sql_pid 2>/dev/null; then
        log "stage 2 llama-server did not exit in time, sending SIGKILL"
        kill -KILL $sql_pid 2>/dev/null
    fi
else
    log "stage 2 llama-server not running"
fi
rm -f "$SQL_LLAMA_PIDFILE"

# ---------------------------------------------------------------------------
# 3. llama.cpp server (started first, stopped last)
# ---------------------------------------------------------------------------
if [[ -f "$LLAMA_PIDFILE" ]]; then
    pid="$(cat "$LLAMA_PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
        log "stopping llama-server (pid $pid)"
        kill -TERM "$pid" 2>/dev/null
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            log "llama-server did not exit in time, sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null
        fi
    else
        log "llama-server pidfile is stale (no process at pid $pid)"
    fi
    rm -f "$LLAMA_PIDFILE"
else
    # Fallback if the pidfile is missing: find whoever is listening on the port.
    pid="$(fuser 1234/tcp 2>/dev/null)"
    if [[ -n "${pid:-}" ]]; then
        log "stopping llama-server found on port 1234 (pid $pid)"
        kill -TERM $pid 2>/dev/null
    else
        log "llama-server not running"
    fi
fi

log "stop_all.sh complete"
