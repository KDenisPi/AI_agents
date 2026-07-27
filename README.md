# AI agents

A home sensor-monitoring system: a collector polls Hubitat and a local
weather station on a fixed interval and stores every reading in MariaDB, and
an AI agent layer (backed by Ollama) reasons over that data and answers
requests through an async HTTP API.

## Components

### Collector (`Collector.py`)
The write-side entry point. On a fixed interval (`INTERVAL_SECONDS`) it
polls every `Source` concurrently, writes what comes back through
`WeatherDb`, and once a calendar month rolls over, archives the previous
month's raw readings into hourly averages so `metering` doesn't grow
unbounded. Runs as the `weather-collector` systemd service; `--once` and
`--dry-run` cover a single manual cycle, `--archive-data` runs just the
archiving step.

### Sources (`Source.py`, `Reading.py`, `HubitatSource.py`, `HubitatClient.py`, `WeatherMcpSource.py`)
`Source` is the common shape every upstream is adapted to: `collect()`
returns a flat list of `Reading`s and never raises, so one dead upstream
only costs that cycle's readings from it.

- `HubitatSource` talks to a Hubitat Elevation hub through `HubitatClient`,
  over the hub's local Maker API (plain HTTP, no cloud) - one
  `devices/all` call per cycle covers every paired sensor.
- `WeatherMcpSource` gets the local weather station's readings by calling
  an MCP tool (`weather_info`) rather than reaching the station directly -
  see the MCP server below.

### Weather station MCP server (`McpMultiHttpServer.py`, `McpWeather.py`, `HttpClient.py`)
A small standalone process exposing the physical weather station as an MCP
tool server, so any MCP-speaking client (the collector today, an
Ollama-driven agent later) can query it without knowing the station's own
HTTP API. `McpMultiHttpServer.py` hosts one or more MCP servers behind a
single Starlette/uvicorn port, routed by path (`McpWeatherServer` lives
under `/weather`); `McpWeather.py` implements the `weather_info` tool by
fetching the station's own HTTP status endpoint through `HttpClient.py`
(aiohttp).

### Database (`db/weather.sql`, `db/weather_users.sql.example`, `WeatherDb.py`)
MariaDB schema: `location`, `sensor`, `metric`, `metering` (raw readings),
and `metering_history` (hourly-averaged archive). Two DB users keep write
and read paths separate - `weather` (SELECT/INSERT/UPDATE/DELETE, used by
`WeatherDb`/`Collector.py`) and `weather_read` (SELECT-only, used by the AI
agent's storage layer) - so a bug on the read side can't corrupt data.
User creation lives in the gitignored `db/weather_users.sql`, copied from
the committed `.example`.

### AI agent (`ai_agent.py`, `ai_agent_storage.py`, `OllamaClient.py`)
The reasoning layer. `MetricStorage` (`ai_agent_storage.py`) wraps the
`weather` schema behind read-only queries (`get_current`, `get_stats`,
`get_history`, `today_outside`, ...) formatted as compact text for a
prompt; `get_stats`/`get_history` take either one metric or a list, so
several metrics come back from one query instead of one round trip each.
`OllamaClient` wraps one model + one conversation talking to an Ollama
server over its HTTP `/api/chat` endpoint, with disk-persisted history and
automatic sliding-window summarization once a conversation grows past a
configured token budget. `AiAgent` ties the storage layer, two chat
models, `TextToVoice` (speech synthesis, see below) and `MetricGrapher`
together - a small model for cheap/frequent calls (`summarize_current`,
`summarize_current_battery`, `summarize_outside_for_today`,
`summarize_history_outside_last_hours`), a large one for anything needing
more reasoning (`transalate_eng_ru`), and `say()` for turning any answer
into a `.wav`.

### Metric graphs (`ai_server_graph.py`)
Renders a `MetricStorage` result as a PNG, one per metric, into
`graph_output_dir` - `MetricGrapher.plot_metrics()` sniffs which of
`get_stats`/`get_history`/`today_outside`'s shapes it was given and picks
a line or bar chart accordingly. `get_current`-shaped data (a single
reading, nothing to plot a trend from) is rejected. Used by
`AiAgent.summarize_outside_for_today`/`summarize_history_outside_last_hours`
when called with `graph=True`.

### AI agent HTTP API (`ai_agent_server.py`)
Exposes `AiAgent` over HTTP for a client that shouldn't have to hold a
connection open through a model call. A `GET` is answered immediately -
`200` once accepted, `503` if Ollama isn't reachable, `500` if something
else stopped it from starting - and the actual answer is delivered later as
a `POST` back to the client, once the model call finishes. Two endpoints
today, both following that shape: `GET /api/current` (current readings)
and `GET /api/outside_today` (today's outside temperature+humidity,
08:00-21:00 or now). Either takes an optional `&voice[=<name>]` for a
spoken answer; `/api/outside_today` also takes `&graph` for a rendered PNG
per metric. Synthesized `.wav`s and rendered `.png`s are served back over
HTTP from `/audio` and `/graphs`, and linked in the callback as
`audio_url`/`graphs`. Runs as the `ai-agent-server` systemd service, on
its own port and its own unit so it restarts independently of the
collector.

### AI client (`ai_client/simple_client.py`)
A throwaway stand-in for the real client, until one exists: listens for the
`POST /api/response` callback and prints it, and prints a numbered menu of
the agent's endpoints so any of them can be fired on demand, with
`--voice`/`--graph` opting into those extras where the endpoint supports
them. Exists to exercise the API's accept-then-callback contract end to
end.

### Configuration (`Config.py`, `collector.env.example`)
Every component reads a single `Config`, built from the environment
(optionally seeded from a gitignored `.env`) via `Config.from_env()`.
`collector.env.example` documents every variable and its default - DB and
Hubitat credentials have no default and must come from `.env`.

### Deployment (`weather-collector.service`, `ai-agent-server.service`, `docker/`)
Each long-running process gets its own systemd unit, so a crash in one
doesn't take down the other. `docker/` holds a Dockerfile and unpinned
`requirements.txt` for containerized runs.

## Communication diagram

```mermaid
flowchart LR
    subgraph Hardware["Physical devices"]
        Hub["Hubitat Elevation hub<br/>(Maker API)"]
        Station["Weather station<br/>(onboard HTTP API)"]
    end

    subgraph CollectorProc["Collector.py — weather-collector.service"]
        direction TB
        HubSrc["HubitatSource"]
        WxSrc["WeatherMcpSource"]
        WDb["WeatherDb"]
        HubSrc --> WDb
        WxSrc --> WDb
    end

    subgraph McpProc["McpMultiHttpServer.py — :8000"]
        McpWx["McpWeatherServer<br/>tool: weather_info"]
    end

    DB[("MariaDB<br/>weather schema")]

    subgraph AgentProc["ai_agent_server.py — :8100 — ai-agent-server.service"]
        direction TB
        ApiCurrent["GET /api/current"]
        ApiToday["GET /api/outside_today"]
        Agent["AiAgent"]
        Store["MetricStorage<br/>(read-only)"]
        Grapher["MetricGrapher<br/>(ai_server_graph.py)"]
        Voice["TextToVoice"]
        Audio["/audio (static)"]
        Graphs["/graphs (static)"]
        ApiCurrent --> Agent
        ApiToday --> Agent
        Agent --> Store
        Agent -. "graph=True" .-> Grapher
        Agent -. "voice" .-> Voice
        Grapher --> Graphs
        Voice --> Audio
    end

    Ollama["Ollama server<br/>(LLM inference + TTS)"]
    Client["ai_client / simple_client.py<br/>(stand-in client)"]

    Hub -- "HTTP GET /devices/all\n(poll, every INTERVAL_SECONDS)" --> HubSrc
    WxSrc -- "MCP call_tool(weather_info)\nover SSE" --> McpWx
    McpWx -- "HTTP GET /api/status" --> Station
    WDb -- "MySQL wire protocol (write user)\nINSERT metering / archive" --> DB

    Store -- "MySQL wire protocol (read-only user)\nSELECT" --> DB
    Agent -- "HTTP POST /api/chat" --> Ollama
    Voice -- "HTTP POST /api/generate\n(Orpheus TTS)" --> Ollama

    Client -- "HTTP GET /api/current?...&voice" --> ApiCurrent
    Client -- "HTTP GET /api/outside_today?...&voice&graph" --> ApiToday
    AgentProc -. "HTTP POST /api/response\n(async callback: text, audio_url, graphs)" .-> Client
    Client -- "HTTP GET /audio/*.wav" --> Audio
    Client -- "HTTP GET /graphs/*.png" --> Graphs
```

**Protocols in play:**
- **HTTP/REST** - Hubitat's Maker API, the weather station's own status
  endpoint, Ollama's `/api/chat` and `/api/generate`, the agent's own
  client-facing API (the `GET` request and the async `POST` callback), and
  the static `/audio`/`/graphs` mounts the callback's links point at.
- **MCP over SSE** - between `WeatherMcpSource` and `McpMultiHttpServer`,
  so the collector never talks to the weather station's HTTP API directly.
- **MySQL wire protocol** - `pymysql` connections from `WeatherDb` (write)
  and `MetricStorage` (read-only) to MariaDB.
