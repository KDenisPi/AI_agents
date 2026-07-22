"""
AI agent - ties the storage layer (ai_agent_storage.py) to two Ollama
models built from Config's Ollama settings.

AiAgent exposes model_small (cheap/frequent calls) and model_large
(anything needing more reasoning), plus summarize_current(), which uses
model_small to turn get_current() into a plain-language summary.
"""

from Config import Config
from OllamaClient import OllamaClient
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
    """
    prompt_template_summarize_current = "Summarize these current sensor readings in a few plain sentences:\n"
    prompt_template_battery_status = "Summarize the battery status of these devices in a few plain sentences:\n"
    promp_no_data = "No current data available."

    def __init__(self, config: Config):
        self.storage = MetricStorage(config)
        self.model_small = OllamaClient(config.ollama_url, config.pllama_model_1)
        self.model_large = OllamaClient(config.ollama_url, config.pllama_model_2)

    def summarize_current(
        self, locations: list[str] | None = None, metrics: list[str] | None = None
    ) -> str:
        """Ask model_small for a plain-language summary of get_current()."""
        current = self.storage.get_current(locations, metrics)
        if not current:
            return self.promp_no_data
        prompt = (self.prompt_template_summarize_current + format_current(current))
        return self.model_small.chat(prompt)

    def summarize_current_battery(self) -> str:
        """Ask model_small for a plain-language summary of get_current()."""
        current = self.storage.get_current([], ['battery'])
        if not current:
            return self.promp_no_data
        prompt = (self.prompt_template_battery_status + format_current(current))
        return self.model_small.chat(prompt)


    def close(self) -> None:
        self.storage.close()


def demo():
    config = Config.from_env()
    agent = AiAgent(config)
    try:
        location = config.weather_location_name
        print("-- get_current() (all locations) --")
        print(format_current(agent.storage.get_current()))

        print("\n-- get_stats_last_hours('temperature', 24) (all locations) --")
        print(format_stats(agent.storage.get_stats_last_hours("temperature", 24)))

        print(f"\n-- get_history_last_hours('temperature', 1, locations=[{location!r}]) --")
        print(format_history(agent.storage.get_history_last_hours("temperature", 1, locations=[location])))

        print("\n-- summarize_current() --")
        try:
            print(agent.summarize_current())
        except Exception as e:
            print(f"  (model_small unreachable: {e})")
    finally:
        agent.close()


if __name__ == "__main__":
    demo()
