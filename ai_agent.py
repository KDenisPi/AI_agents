"""
AI agent - ties the storage layer (ai_agent_storage_clickhouse.py, the
ClickHouse reader) to the chat models built from Config's Ollama settings.
model_small/model_large go through
OllamaClient, which speaks the OpenAI-compatible /v1/chat/completions API
and so can point at Ollama or a llama.cpp server; model_text_to_voice
still talks to Ollama specifically (see text_to_voice.py).

AiAgent exposes model_small (cheap/frequent calls), model_large (anything
needing more reasoning) and model_text_to_voice (speech synthesis), plus
summarize_current(), which uses model_small to turn get_current() into a
plain-language summary, and say(), which speaks text into a .wav.

Each run starts a new conversation unless AiAgent is given a session_id,
which resumes the one saved under that name.
"""

from pathlib import Path

from Config import Config
from OllamaClient import OllamaClient, session_id_for
from ai_agent_storage import format_current, format_history, format_stats, format_today_outside
from ai_agent_storage_clickhouse import MetricStorageClickhouse
from ai_server_graph import GraphError, MetricGrapher
from text_to_voice import TextToVoice


class AiAgent:
    """
    Ties the storage layer to two chat models on the same host - a small
    one for cheap/frequent calls, a large one for anything that needs more
    reasoning.

    Usage:
        agent = AiAgent(config)
        agent.storage.get_current(["Weather station"])
        agent.model_small.chat("...")
        agent.close()

        # Same conversation across runs - both models pick up where they
        # left off, instead of starting fresh each process:
        agent = AiAgent(config, session_id="kitchen")
    """
    prompt_template_summarize_current = "Summarize these current sensor readings in 2-3 short sentences total. \
	Do not mention the specific date or time each reading was taken - these are current conditions, so there is no need to say when. No notes or closing remarks.\n"

    # The all-normal reply is a fixed phrase decided in code (see
    # summarize_current_battery), never by the model - the small model kept
    # narrating it ("There is no other device...") instead of emitting it
    # verbatim. So the prompt only covers the listing case, which is all the
    # model is ever asked to do here.
    battery_low_threshold = 50
    prompt_battery_all_normal = "Battery levels for all devices are normal."
    prompt_template_battery_status = "Report battery status of these devices. Output only the answer itself - " \
        "no preamble, reasoning, or explanation. List only devices below 50% as " \
        "'DeviceName: NN%'. Do not mention devices at or above 50% individually.\n"

    prompt_template_translate_en_ru = "Translate these sentences from English to Russian:\n"
    # {hours} filled in per call. Fed min/max/avg/count, not raw readings -
    # see summarize_history_outside_last_hours - so it's told as much, and
    # given the same "no notes or closing remarks" guard as the today
    # template below: a small model handed a wall of numbers with no such
    # guard tends to append a chatty offer to help instead of stopping.
    prompt_template_history_outside_last_hours = "Summarize outside sensor min/max/avg readings for the last {hours} hours in exactly 2 short sentences: \
	one for temperature (range and average only), one for humidity (range and average only). \
	Do not add any other sentence - no interpretation, commentary, or trend descriptions beyond those numbers. \
	Do not mention how many readings were taken, sampling intervals, or anything about how the data was collected. No notes or closing remarks.\n"
    prompt_template_outside_for_today = "Summarize outside sensor readings for today in exactly 2 short sentences: \
	one for temperature (state its min, its max, and the time given next to the max, exactly as given below - do not compute, estimate, or restate them as a different number), one for humidity (min and max only). \
	Do not add any other sentence - no interpretation, commentary, or notes about the data beyond those numbers. \
	Do not mention how many readings were taken, sampling intervals, or anything about how the data was collected. No notes or closing remarks.\n"
    # {hours} filled in per call. Fed min/max/avg/count per device (sensor
    # name, not location - several power sensors can share one location, see
    # summarize_history_power_last_hours), same as
    # prompt_template_history_outside_last_hours above - told so, and given
    # the same "no notes or closing remarks" guard, plus an explicit steer
    # toward "typical" and "peak" since that's the pair of numbers a power
    # question actually wants (avg and max), not every field in MetricStats.
    # The "only devices listed below" / "exact names" lines guard against a
    # failure mode seen with the small model: given a single blended-device
    # data line but asked for "each device", it invented plausible-looking
    # extra devices (e.g. "Light Bulb", "Computer") instead of reporting
    # just what the data actually contained.
    prompt_template_history_power_last_hours = "Summarize power usage readings for the last {hours} hours in 2-3 short sentences total. \
	Report one line per device, using the exact device name as given below - do not rename, translate, or invent devices. Only cover devices listed below; if only one is listed, report only that one. For each device, report its typical (average) usage and its peak. Do not mention how many readings were taken, sampling intervals, or anything about how the data was collected. No notes or closing remarks.\n"
    prompt_no_data = "No current data available."

    def __init__(self, config: Config, session_id: str | None = None):
        """`session_id` resumes a named conversation, surviving process
        restarts. Default (None) is a fresh session per process - note that
        resuming re-sends every earlier exchange, so readings from previous
        runs stay in the prompt and it grows with each one. Call
        model_small.reset() when the history stops being worth carrying.
        """
        self.storage = MetricStorageClickhouse(config)
        # One id per model, not one per agent: both clients would otherwise
        # write the same history file. The role suffix keeps them apart even
        # when ollama_model_1 and ollama_model_2 name the same model.
        self.model_small = OllamaClient(
            config.ollama_url_1,
            config.ollama_model_1,
            session_id=session_id_for(config.ollama_model_1, f"{session_id}-small")
            if session_id
            else None,
            max_history_tokens=config.ollama_max_history_tokens,
            keep_recent_messages=config.ollama_keep_recent_messages,
        )
        self.model_large = OllamaClient(
            config.ollama_url_2,
            config.ollama_model_2,
            session_id=session_id_for(config.ollama_model_2, f"{session_id}-large")
            if session_id
            else None,
            max_history_tokens=config.ollama_max_history_tokens,
            keep_recent_messages=config.ollama_keep_recent_messages,
        )
        # Not an OllamaClient: speech synthesis needs a raw, untemplated
        # prompt and has no conversation to keep. No session_id for the
        # same reason. See text_to_voice.py.
        self.model_text_to_voice = TextToVoice(
            config.ollama_url_text_to_voice,
            config.ollama_model_text_to_voice,
            voice=config.ollama_voice,
            output_dir=config.voice_output_dir,
            retention_hours=config.voice_retention_hours,
            max_workers=config.voice_max_workers,
            backend=config.ollama_backend_text_to_voice,
        )
        self._grapher = MetricGrapher(config.graph_output_dir, config.graph_retention_hours)

    def say(self, text: str, path: str | None = None, voice: str | None = None) -> Path:
        """Speak `text` into a .wav and return where it was written.
        Defaults to a timestamped file under Config.voice_output_dir, in
        Config.ollama_voice unless `voice` names another one."""
        return self.model_text_to_voice.synthesize(text, path, voice)

    def summarize_current(
        self, locations: list[str] | None = None, metrics: list[str] | None = None
    ) -> str:
        """Ask model_small for a plain-language summary of get_current()."""
        current = self.storage.get_current(locations, metrics or ['temperature', 'humidity'])
        if not current:
            return self.prompt_no_data
        prompt = (self.prompt_template_summarize_current + format_current(current))
        return self.model_small.chat_once(prompt)

    def summarize_current_battery(self) -> str:
        """Summarize battery status of get_current(). When no device is
        below battery_low_threshold, return the fixed all-normal phrase
        without a model call - the small model narrated it instead of
        emitting it verbatim. The model is asked only to list the devices
        that are actually low."""
        current = self.storage.get_current([], ['battery'])
        if not current:
            return self.prompt_no_data
        if not any(
            metrics['battery'].value < self.battery_low_threshold
            for metrics in current.values()
            if 'battery' in metrics
        ):
            return self.prompt_battery_all_normal
        prompt = (self.prompt_template_battery_status + format_current(current))
        return self.model_small.chat_once(prompt)

    def summarize_history_outside_last_hours(
        self, hours: int, metrics: list[str] | None = None, graph: bool = False
    ) -> tuple[str, dict[str, Path]]:
        """Ask model_small for a plain-language summary of
        get_stats_outside_last_hours() - min/max/avg/count, not every raw
        reading. At the collector's 10-minute interval, 24 hours is ~150
        readings per metric; handed that as a raw list (the previous
        approach, via get_history + format_history) a small model tends to
        just echo it back as a table instead of summarizing it, and the
        prompt only grows with the window regardless. Stats are one line
        per metric no matter how many hours are requested. Returns
        (summary, graphs) - graphs is empty unless graph=True, in which
        case it holds one rendered PNG per metric
        (ai_server_graph.plot_metrics()), from a separate raw-history query
        only made when a graph is actually wanted."""
        metrics = metrics or ['temperature', 'humidity']
        stats = self.storage.get_stats_outside_last_hours(metrics, hours)
        if not stats:
            return self.prompt_no_data, {}

        graphs: dict[str, Path] = {}
        if graph:
            history = self.storage.get_history_outside_last_hours(metrics, hours)
            try:
                graphs = self._grapher.plot_metrics(history)
            except GraphError:
                pass  # nothing to plot - the text summary still goes out

        prompt = (
            self.prompt_template_history_outside_last_hours.format(hours=hours)
            + format_stats(stats)
        )
        return self.model_small.chat_once(prompt), graphs

    def summarize_history_power_last_hours(
        self, hours: int, graph: bool = False
    ) -> tuple[str, dict[str, Path]]:
        """Ask model_small for a plain-language summary of power usage over
        the last `hours` - min/max/avg/count per device, via
        get_stats_last_hours("power", hours, by_sensor=True). Unlike
        summarize_history_outside_last_hours, this is a single-metric call
        (no metrics list), so storage returns the classic key -> stats shape
        rather than nesting by metric. by_sensor=True keys that by sensor
        name rather than location, since several distinctly-named power
        sensors can share one location - location grouping would blend
        their readings into one bucket, which is both wrong for the
        text summary and what previously caused the model to invent
        devices that don't exist (see prompt_template_history_power_last_hours).
        Returns (summary, graphs) - graphs is empty unless graph=True, in
        which case it holds one PNG with one line per device, from the same
        by_sensor grouping via get_history_last_hours()."""
        stats = self.storage.get_stats_last_hours("power", hours, by_sensor=True)
        if not stats:
            return self.prompt_no_data, {}

        graphs: dict[str, Path] = {}
        if graph:
            history = self.storage.get_history_last_hours("power", hours, by_sensor=True)
            try:
                graphs = self._grapher.plot_metrics(history, metric="power")
            except GraphError:
                pass  # nothing to plot - the text summary still goes out

        prompt = (
            self.prompt_template_history_power_last_hours.format(hours=hours)
            + format_stats(stats)
        )
        return self.model_small.chat_once(prompt), graphs

    def summarize_outside_for_today(self, graph: bool = False) -> tuple[str, dict[str, Path]]:
        """Ask model_small for a plain-language summary of
        today_outside(). Returns (summary, graphs) - graphs is empty
        unless graph=True, in which case it holds one rendered PNG per
        metric (ai_server_graph.plot_metrics(), reusing the same history
        already fetched for the prompt rather than querying again)."""
        history = self.storage.today_outside()
        if not history:
            return self.prompt_no_data, {}

        graphs: dict[str, Path] = {}
        if graph:
            try:
                graphs = self._grapher.plot_metrics(history)
            except GraphError:
                pass  # nothing to plot - the text summary still goes out

        prompt = (self.prompt_template_outside_for_today + format_today_outside(history))
        return self.model_small.chat_once(prompt), graphs

    def transalate_eng_ru(self, message: str) -> str:
        """Ask model_large for message translation from Eng to Russian."""
        prompt = (self.prompt_template_translate_en_ru + message)
        #We will keep history of translations, so we use chat() instead of chat_once()
        return self.model_large.chat(prompt)

    def close(self) -> None:
        self.storage.close()

def demo():
    config = Config.from_env()
    if config.log_file == Config.log_file:  # LOG_FILE not overridden via env
        config.log_file = "logs/ai_agent.log"
    config.configure_logging()
    agent = AiAgent(config)
    try:
        location = config.weather_location_name
        print("-- get_current() (all locations) --")
        print(format_current(agent.storage.get_current(metrics=['temperature', 'humidity'])))

        print("-- inside locations --")
        print(str(agent.storage.inside_locations))

        print("-- outside locations --")
        print(str(agent.storage.outside_locations))

        #print("\n-- get_stats_last_hours('temperature', 24) (all locations) --")
        #print(format_stats(agent.storage.get_stats_last_hours("temperature", 24)))

        #print(f"\n-- get_history_last_hours('temperature', 1, locations=[{location!r}]) --")
        #print(format_history(agent.storage.get_history_last_hours("temperature", 1, locations=[location])))

        print("\n-- summarize_current() --")
        try:
            msg = agent.summarize_current(metrics=['temperature', 'humidity'])
            print(msg)
            print("\n-- translate_eng_ru() --")
            msg_ru = agent.transalate_eng_ru(msg)
            print(msg_ru)
        except Exception as e:
            print(f"  (model_small unreachable: {e})")

        print("\n-- summarize_current_battery() --")
        try:
            msg = agent.summarize_current_battery()
            print(msg)
            #print("\n-- translate_eng_ru() --")
            #msg_ru = agent.transalate_eng_ru(msg)
            #print(msg_ru)
        except Exception as e:
            print(f"  (model_small unreachable: {e})")

        print("\n-- summarize_outside_for_today() --")
        try:
            msg = agent.summarize_outside_for_today(graph=True)
            print(msg)
        except Exception as e:
            print(f"  (model_small unreachable: {e})")

    finally:
        agent.close()


def demo_pruning():
    """Exercise OllamaClient's sliding-window pruning end to end through a
    real AiAgent. Lowers model_large's budget well below Config's default
    (which would take dozens of exchanges to fill) so a handful of chat()
    turns is enough to force at least one prune, then checks that older
    turns actually got folded into a summary instead of just growing
    forever."""
    config = Config.from_env()
    agent = AiAgent(config)
    bot = agent.model_large
    bot.max_history_tokens = 100
    bot.keep_recent_messages = 4

    # Long enough on their own (~19 est. tokens/prompt, cumulative ~109 by
    # the last one) that the budget gets crossed from the prompts alone -
    # no need to depend on how verbose the model's replies turn out to be.
    prompts = [
        "My favorite color is teal, and I really love long walks on rainy afternoons.",
        "I have a dog named Biscuit, a very energetic three-year-old golden retriever.",
        "I live in Seattle, in a small apartment near the water with a nice view.",
        "I work as a software engineer and enjoy hiking on weekends when I can.",
        "What's a good weekend activity near where I live, given my dog and the weather?",
        "Remind me what my dog's name is, my favorite color, and where I live.",
    ]
    try:
        for prompt in prompts:
            reply = bot.chat(prompt)
            print(f"> {prompt}\n{reply}")
            print(f"  [history={len(bot.context)} msg(s), summary={'yes' if bot.summary else 'no'}]\n")

        print("Final summary:", bot.summary)
        assert bot.summary is not None, "expected the low budget to force at least one pruning round"
        print(
            "pruning check: OK - older turns were folded into a summary "
            f"instead of growing the verbatim history past {len(prompts) * 2} message(s)"
        )
    except AssertionError:
        raise
    except Exception as e:
        print(f"  (model unreachable: {e})")
    finally:
        agent.close()


if __name__ == "__main__":
    demo()
    print("\n" + "=" * 60)
    demo_pruning()
