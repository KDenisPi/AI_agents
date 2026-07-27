"""
Renders MetricStorage results (ai_agent_storage.py) as PNG graphs, one per
metric - so an answer can point at an image the way say() points at a
generated .wav (text_to_voice.py).

MetricStorage's functions don't all return the same shape, so plot_metrics()
is the one entry point: it sniffs which shape it was given and dispatches
to the matching plot_* below.

    get_current(...)   -> location -> metric -> MetricValue         plot_current: bar, one per metric
    get_stats*(...)    -> location -> MetricStats                   plot_stats: bar (min/avg/max), one metric
    today_outside()    -> location -> metric -> list[HistoryPoint]  one plot_history per metric
    get_history*(...)  -> location -> list[HistoryPoint]            plot_history: line, needs metric= (see below)

get_history()-shaped results don't carry a metric name - HistoryPoint has
none, it's implied by whatever metric was passed to get_history() itself.
Pass plot_metrics(data, metric="temperature") for those, or call
plot_history() directly.

Uses matplotlib's Agg backend throughout - this runs on a headless server,
so there is never a display to render to.

Install:
    pip install matplotlib
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ai_agent_storage import HistoryPoint, MetricStats, MetricValue

logger = logging.getLogger("ai-agent")

DEFAULT_OUTPUT_DIR = Path("graph_output")
DEFAULT_RETENTION_HOURS = 24


def _safe(name: str) -> str:
    """Filename-safe form of a metric name, so it can't reach outside the
    output directory."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class GraphError(RuntimeError):
    """Nothing plottable was found in the data passed in."""


class MetricGrapher:
    """
    Usage:
        grapher = MetricGrapher()
        grapher.plot_metrics(storage.today_outside())
        grapher.plot_metrics(storage.get_history_last_hours("temperature", 24), metric="temperature")
    """

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        retention_hours: float = DEFAULT_RETENTION_HOURS,
    ):
        # Same reasoning as TextToVoice.retention_hours - these are only
        # useful until a client has fetched them, and would otherwise grow
        # the directory without limit. 0 or less keeps everything.
        self.retention_hours = retention_hours
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self, retention_hours: float | None = None) -> int:
        """Delete pngs in the output directory last modified more than
        retention_hours ago, returning how many were removed. Called before
        every render, so anything generating a graph prunes as it goes."""
        hours = self.retention_hours if retention_hours is None else retention_hours
        if hours <= 0:
            return 0

        cutoff = time.time() - hours * 3600
        removed = freed = 0
        # Non-recursive, *.png only - the reach stays exactly the files this
        # class writes, same as TextToVoice.cleanup().
        for png in sorted(self._output_dir.glob("*.png")):
            try:
                stat = png.stat()
                if not png.is_file() or stat.st_mtime >= cutoff:
                    continue
                png.unlink()
            except OSError as e:
                logger.warning("Could not remove %s: %s", png, e)
                continue
            removed += 1
            freed += stat.st_size

        if removed:
            logger.info(
                "Removed %d png file(s) older than %gh from %s, freeing %.1f MB",
                removed, hours, self._output_dir, freed / 1e6,
            )
        return removed

    def _path_for(self, metric: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self._output_dir / f"{_safe(metric)}-{stamp}.png"

    def _save(self, fig, metric: str, path: Path | None) -> Path:
        path = Path(path) if path else self._path_for(metric)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_history(
        self,
        history: dict[str, list[HistoryPoint]],
        metric: str = "value",
        path: str | Path | None = None,
    ) -> Path:
        """Line chart, one line per location, from a get_history()-shaped
        result. Raises GraphError if every location came back empty."""
        self.cleanup()
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = False
        for location, points in sorted(history.items()):
            if not points:
                continue
            ax.plot(
                [p.taken_at for p in points], [p.value for p in points],
                marker="o", markersize=3, label=location,
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            raise GraphError(f"No data to plot for {metric!r}")

        ax.set_title(metric)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        out = self._save(fig, metric, path)
        logger.info("plot_history(%s): %d location(s) -> %s", metric, len(history), out)
        return out

    def plot_stats(
        self, stats: dict[str, MetricStats], path: str | Path | None = None
    ) -> Path:
        """Bar chart of min/avg/max per location, from a get_stats()-shaped
        result. The metric name comes from the MetricStats values
        themselves, so it can't be split across more than one metric."""
        if not stats:
            raise GraphError("No data to plot")
        metric = next(iter(stats.values())).metric
        self.cleanup()

        locations = sorted(stats)
        x = range(len(locations))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - width for i in x], [stats[l].min for l in locations], width, label="min")
        ax.bar(list(x), [stats[l].avg for l in locations], width, label="avg")
        ax.bar([i + width for i in x], [stats[l].max for l in locations], width, label="max")
        ax.set_xticks(list(x))
        ax.set_xticklabels(locations, rotation=30, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        out = self._save(fig, metric, path)
        logger.info("plot_stats(%s): %d location(s) -> %s", metric, len(stats), out)
        return out

    def plot_current(self, current: dict[str, dict[str, MetricValue]]) -> dict[str, Path]:
        """One bar chart per metric (latest value per location), from a
        get_current()-shaped result."""
        by_metric: dict[str, dict[str, MetricValue]] = {}
        for location, metrics in current.items():
            for metric, value in metrics.items():
                by_metric.setdefault(metric, {})[location] = value
        if not by_metric:
            raise GraphError("No data to plot")

        paths: dict[str, Path] = {}
        for metric, per_location in by_metric.items():
            self.cleanup()
            locations = sorted(per_location)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(locations, [per_location[l].value for l in locations])
            ax.set_xticks(range(len(locations)))
            ax.set_xticklabels(locations, rotation=30, ha="right")
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(True, axis="y", alpha=0.3)
            paths[metric] = self._save(fig, metric, None)
        logger.info("plot_current(): %d metric(s) -> %s", len(paths), list(paths))
        return paths

    def plot_metrics(self, data: dict, metric: str | None = None) -> dict[str, Path]:
        """One entry point for any MetricStorage result: sniffs which of
        the shapes documented at the top of this module `data` is, and
        renders one PNG per metric. `metric` only matters for a
        get_history()-shaped result, where it can't be inferred - see the
        module docstring. Raises GraphError if data is empty or a shape
        this can't handle."""
        if not data:
            raise GraphError("No data to plot")
        sample = next(iter(data.values()))

        if isinstance(sample, MetricStats):
            out = self.plot_stats(data)
            return {sample.metric: out}

        if isinstance(sample, list):
            name = metric or "value"
            return {name: self.plot_history(data, metric=name)}

        if isinstance(sample, dict):
            inner = next(iter(sample.values()), None)
            if inner is None or isinstance(inner, MetricValue):
                return self.plot_current(data)
            if isinstance(inner, list):
                # today_outside()-shaped: location -> metric -> list[HistoryPoint]
                by_metric: dict[str, dict[str, list[HistoryPoint]]] = {}
                for location, metrics in data.items():
                    for m, points in metrics.items():
                        by_metric.setdefault(m, {})[location] = points
                paths: dict[str, Path] = {}
                for m, history in by_metric.items():
                    try:
                        paths[m] = self.plot_history(history, metric=m)
                    except GraphError:
                        continue  # that metric had no data anywhere - skip it, not fatal
                if not paths:
                    raise GraphError("No data to plot")
                return paths

        raise GraphError(f"Don't know how to plot {type(sample).__name__} values")


def demo():
    from Config import Config
    from ai_agent_storage import MetricStorage

    config = Config.from_env()
    config.configure_logging()

    storage = MetricStorage(config)
    grapher = MetricGrapher(config.graph_output_dir, config.graph_retention_hours)
    try:
        print("-- plot_metrics(today_outside()) --")
        print(grapher.plot_metrics(storage.today_outside()))

        print("-- plot_metrics(get_current(metrics=['temperature', 'humidity'])) --")
        print(grapher.plot_metrics(storage.get_current(metrics=["temperature", "humidity"])))

        print("-- plot_metrics(get_stats_last_hours('temperature', 24)) --")
        print(grapher.plot_metrics(storage.get_stats_last_hours("temperature", 24)))

        print("-- plot_metrics(get_history_last_hours('temperature', 24), metric='temperature') --")
        print(grapher.plot_metrics(storage.get_history_last_hours("temperature", 24), metric="temperature"))
    except GraphError as e:
        print(f"  (nothing to plot: {e})")
    finally:
        storage.close()


if __name__ == "__main__":
    demo()
