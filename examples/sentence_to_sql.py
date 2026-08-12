"""
Two-stage Ollama pipeline: a sentence goes to one model with a context
(template) file to become a plain-English data request, then that request
goes to a second model with a different context file to become a DuckDB
SQL query, which is then run against a local DuckDB file.

Schema-agnostic - point --schema at whatever DDL/description file matches
the database you actually want SQL for; the default is just a small
placeholder so this runs out of the box.

duckdb isn't installed in every environment this repo runs in - if it's
missing, that step is logged and skipped instead of failing the run.

Run:
    python examples/sentence_to_sql.py --sentence "Total quantity ordered per product last month"
    python examples/sentence_to_sql.py --sentence "..." --schema /path/to/your_schema.txt
    python examples/sentence_to_sql.py --sentence "..." --context1 my_stage1.txt --context2 my_stage2.txt
    python examples/sentence_to_sql.py --sentence "..." --db my.duckdb

Every model call's prompt and reply are logged here; OllamaClient's own
@_timed logging (via the "ollama" logger) adds elapsed time and token
counts for each call to the same log.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Repo has no package layout - reach into the parent dir for OllamaClient.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from OllamaClient import OllamaClient  # noqa: E402 - after sys.path fixup

logger = logging.getLogger("examples.sentence_to_sql")

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_CONTEXT1 = EXAMPLES_DIR / "sentence_to_sql_stage1_context.txt"
DEFAULT_CONTEXT2 = EXAMPLES_DIR / "sentence_to_sql_stage2_context.txt"
DEFAULT_SCHEMA = EXAMPLES_DIR / "sentence_to_sql_schema.txt"


def _fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key.upper() + "}}", value)
    return template


def _strip_sql_fence(text: str) -> str:
    """Models sometimes wrap SQL in a ```sql ... ``` fence despite being
    asked not to - strip it so what follows is plain SQL."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def run_stage(client: OllamaClient, stage_name: str, prompt: str) -> str:
    logger.info("%s prompt:\n%s", stage_name, prompt)
    reply = client.chat_once(prompt)
    logger.info("%s reply:\n%s", stage_name, reply)
    return reply


def run_sql(db_path: str, sql: str) -> None:
    """Execute `sql` against a local DuckDB file, if the duckdb package is
    installed - its absence is logged and treated as a normal, expected
    outcome here rather than an error."""
    try:
        import duckdb
    except ImportError:
        logger.warning("duckdb package not installed - skipping execution of:\n%s", sql)
        return

    started = time.perf_counter()
    try:
        connection = duckdb.connect(db_path)
        try:
            rows = connection.execute(sql).fetchall()
            columns = [d[0] for d in connection.description]
        finally:
            connection.close()
    except Exception as e:
        logger.warning("duckdb(%s) failed in %.2fs: %s", db_path, time.perf_counter() - started, e)
        return

    logger.info(
        "duckdb(%s) ok in %.2fs - %d row(s), columns=%s",
        db_path, time.perf_counter() - started, len(rows), columns,
    )
    for row in rows:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentence -> intent -> SQL, two-stage Ollama pipeline, run against local DuckDB"
    )
    parser.add_argument(
        "--sentence",
        default="Total quantity ordered per product last month",
        help="natural-language question to turn into SQL",
    )
    parser.add_argument("--context1", default=str(DEFAULT_CONTEXT1), help="stage 1 context/template file")
    parser.add_argument("--context2", default=str(DEFAULT_CONTEXT2), help="stage 2 context/template file")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="DDL/description file for stage 2")
    parser.add_argument("--url", default="http://192.168.1.57:11434", help="Ollama server URL")
    parser.add_argument("--model1", default="llama3.1:8b", help="model for stage 1 (sentence -> intent)")
    parser.add_argument("--model2", default="qwen3.6", help="model for stage 2 (intent -> SQL)")
    parser.add_argument("--db", default="sentence_to_sql.duckdb", help="local DuckDB file to run the SQL against")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    context1 = Path(args.context1).read_text()
    context2 = Path(args.context2).read_text()
    schema = Path(args.schema).read_text()

    prompt1 = _fill(context1, sentence=args.sentence)
    client1 = OllamaClient(args.url, args.model1)
    intent = run_stage(client1, "stage1", prompt1)

    prompt2 = _fill(context2, answer=intent, schema=schema)
    client2 = OllamaClient(args.url, args.model2)
    sql = _strip_sql_fence(run_stage(client2, "stage2", prompt2))

    print("\nGenerated SQL:\n" + sql)
    run_sql(args.db, sql)


if __name__ == "__main__":
    main()
