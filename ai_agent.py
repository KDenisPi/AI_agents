"""
AI agent - ties the storage layer (ai_agent_storage.py) to two Ollama
models built from Config's Ollama settings.

AiAgent exposes model_small (cheap/frequent calls) and model_large
(anything needing more reasoning), plus summarize_current(), which uses
model_small to turn get_current() into a plain-language summary.

Each run starts a new conversation unless AiAgent is given a session_id,
which resumes the one saved under that name.
"""

from Config import Config
from OllamaClient import OllamaClient, session_id_for
from ai_agent_storage import MetricStorage, format_current, format_history, format_stats


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
        )
        self.model_large = OllamaClient(
            config.ollama_url,
            config.ollama_model_2,
            session_id=session_id_for(config.ollama_model_2, f"{session_id}-large")
            if session_id
            else None,
        )

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


if __name__ == "__main__":
    demo()
