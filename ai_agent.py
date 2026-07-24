"""
AI agent - ties the storage layer (ai_agent_storage.py) to the Ollama
models built from Config's Ollama settings.

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
from ai_agent_storage import MetricStorage, format_current, format_history, format_stats
from text_to_voice import TextToVoice


class AiAgent:
    """
    Ties the storage layer to two Ollama models on the same host - a small
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
    prompt_template_summarize_current = "Summarize these current sensor readings in a few plain sentences:\n"
    prompt_template_battery_status = "Summarize the battery status of these devices in a few plain sentences:\n"
    prompt_template_translate_en_ru = "Translate these sentences from English to Russian:\n"
    promp_no_data = "No current data available."

    def __init__(self, config: Config, session_id: str | None = None):
        """`session_id` resumes a named conversation, surviving process
        restarts. Default (None) is a fresh session per process - note that
        resuming re-sends every earlier exchange, so readings from previous
        runs stay in the prompt and it grows with each one. Call
        model_small.reset() when the history stops being worth carrying.
        """
        self.storage = MetricStorage(config)
        # One id per model, not one per agent: both clients would otherwise
        # write the same history file. The role suffix keeps them apart even
        # when ollama_model_1 and ollama_model_2 name the same model.
        self.model_small = OllamaClient(
            config.ollama_url,
            config.ollama_model_1,
            session_id=session_id_for(config.ollama_model_1, f"{session_id}-small")
            if session_id
            else None,
            max_history_tokens=config.ollama_max_history_tokens,
            keep_recent_messages=config.ollama_keep_recent_messages,
        )
        self.model_large = OllamaClient(
            config.ollama_url,
            config.ollama_model_2,
            session_id=session_id_for(config.ollama_model_2, f"{session_id}-large")
            if session_id
            else None,
            max_history_tokens=config.ollama_max_history_tokens,
            keep_recent_messages=config.ollama_keep_recent_messages,
        )
        # Not an OllamaClient: speech synthesis needs /api/generate with raw
        # prompts and has no conversation to keep. No session_id for the
        # same reason. See text_to_voice.py.
        self.model_text_to_voice = TextToVoice(
            config.ollama_url,
            config.ollama_model_text_to_voice,
            voice=config.ollama_voice,
            output_dir=config.voice_output_dir,
        )

    def say(self, text: str, path: str | None = None) -> Path:
        """Speak `text` into a .wav and return where it was written.
        Defaults to a timestamped file under Config.voice_output_dir."""
        return self.model_text_to_voice.synthesize(text, path)

    def summarize_current(
        self, locations: list[str] | None = None, metrics: list[str] | None = None
    ) -> str:
        """Ask model_small for a plain-language summary of get_current()."""
        current = self.storage.get_current(locations, metrics)
        if not current:
            return self.promp_no_data
        prompt = (self.prompt_template_summarize_current + format_current(current))
        return self.model_small.chat_once(prompt)

    def summarize_current_battery(self) -> str:
        """Ask model_small for a plain-language summary of get_current()."""
        current = self.storage.get_current([], ['battery'])
        if not current:
            return self.promp_no_data
        prompt = (self.prompt_template_battery_status + format_current(current))
        return self.model_small.chat_once(prompt)

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
            try:
                msg_ru = agent.transalate_eng_ru(msg)
                print(msg_ru)
            except Exception as e:
                print(f"  (model_large unreachable: {e})")
        except Exception as e:
            print(f"  (model_small unreachable: {e})")

        print("\n-- summarize_current_battery() --")
        try:
            msg = agent.summarize_current_battery()
            print(msg)
            print("\n-- translate_eng_ru() --")
            try:
                msg_ru = agent.transalate_eng_ru(msg)
                print(msg_ru)
            except Exception as e:
                print(f"  (model_large unreachable: {e})")
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
