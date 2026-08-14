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
counts for each call to the same log. Stage 2 (intent -> SQL) additionally
appends a (prompt, completion, execution outcome) record per run to
--training-log, meant as raw material for later fine-tuning that model.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repo has no package layout - reach into the parent dir for OllamaClient.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from OllamaClient import OllamaClient  # noqa: E402 - after sys.path fixup

logger = logging.getLogger("examples.sentence_to_sql")

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_CONTEXT1 = EXAMPLES_DIR / "sentence_to_sql_stage1_context.txt"
DEFAULT_CONTEXT2 = EXAMPLES_DIR / "sentence_to_sql_stage2_context.txt"
DEFAULT_SCHEMA = EXAMPLES_DIR / "duckdb_schema.sql"
DEFAULT_SQL_EXAMPLES = EXAMPLES_DIR / "queries.sql"



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


def _extract_intent_json(text: str) -> str:
    """Stage 1 is asked for JSON only, but smaller models sometimes wrap it in a
    prose preamble or a ```json fence. Pull out the first complete top-level
    {...} object (brace-matched, ignoring braces inside strings) so stage 2
    receives the intent alone rather than the surrounding chatter. If no balanced
    object is found, return the text stripped - stage 2 still sees something, and
    run_stage has already logged the raw reply for inspection."""
    start = text.find("{")
    if start == -1:
        return text.strip()
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text.strip()


def run_stage(client: OllamaClient, stage_name: str, prompt: str) -> str:
    logger.debug("%s prompt:\n%s", stage_name, prompt)
    reply = client.chat_once(prompt)
    logger.info("%s reply:\n%s", stage_name, reply)
    return reply


def run_sql(db_path: str, sql: str) -> dict:
    """Execute `sql` against a local DuckDB file, if the duckdb package is
    installed - its absence is logged and treated as a normal, expected
    outcome here rather than an error. Returns the outcome (status/error/
    row_count) so the caller can record it alongside the training example -
    "unavailable" means the SQL was never checked, not that it's wrong."""
    try:
        import duckdb
    except ImportError:
        logger.warning("duckdb package not installed - skipping execution of:\n%s", sql)
        return {"status": "unavailable", "error": None, "row_count": None}

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
        return {"status": "failed", "error": str(e), "row_count": None}

    logger.info(
        "duckdb(%s) ok in %.2fs - %d row(s), columns=%s",
        db_path, time.perf_counter() - started, len(rows), columns,
    )
    for row in rows:
        print(row)
    return {"status": "success", "error": None, "row_count": len(rows)}


def log_training_example(
    path: str, model: str, prompt: str, completion: str, sql: str, db_result: dict
) -> None:
    """Append one stage-2 (prompt, completion) pair to a JSONL file, for use
    later as fine-tuning data for the stage-2 model. Kept separate from the
    human-readable log: one self-contained record per line, with the exact
    prompt actually sent (schema and request already filled in) so it stays
    reproducible even after the templates change, plus whether the SQL it
    produced actually ran - a cheap signal for filtering good examples from
    bad ones once DuckDB is available to check against."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt": prompt,
        "completion": completion,
        "sql": sql,
        "db_status": db_result["status"],
        "db_error": db_result["error"],
        "row_count": db_result["row_count"],
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("logged stage2 training example to %s (db_status=%s)", path, db_result["status"])


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
    parser.add_argument("--sqlexamples", default=str(DEFAULT_SQL_EXAMPLES), help="SQL examples for stage 2")
    parser.add_argument("--url", default="http://192.168.1.57:11434", help="Ollama server URL")
    parser.add_argument("--model1", default="llama3.1:8b", help="model for stage 1 (sentence -> intent)")
    parser.add_argument("--model2", default="Qwen2.5-Coder", help="model for stage 2 (intent -> SQL)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature for both models; 0 = greedy/deterministic, "
        "which keeps intent extraction and SQL stable run-to-run",
    )
    parser.add_argument("--db", default="/home/denis/projects/wiki_data/run2/wiki.duckdb", help="local DuckDB file to run the SQL against")
    parser.add_argument(
        "--training-log",
        default="sentence_to_sql_stage2_training.jsonl",
        help="JSONL file to append stage 2 (prompt, completion, execution outcome) records to",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    context1 = Path(args.context1).read_text()
    context2 = Path(args.context2).read_text()
    schema = Path(args.schema).read_text()
    sqlexamples = Path(args.sqlexamples).read_text()

    options = {"temperature": args.temperature}
    prompt1 = _fill(context1, sentence=args.sentence)
    client1 = OllamaClient(args.url, args.model1, options=options)
    intent = _extract_intent_json(run_stage(client1, "stage1", prompt1))
    logger.info("stage1 intent (extracted):\n%s", intent)

    prompt2 = _fill(context2, answer=intent, schema=schema, examples=sqlexamples)
    client2 = OllamaClient(args.url, args.model2, options=options)
    completion = run_stage(client2, "stage2", prompt2)
    sql = _strip_sql_fence(completion)

    print("\nGenerated SQL:\n" + sql)
    db_result = run_sql(args.db, sql)
    log_training_example(args.training_log, args.model2, prompt2, completion, sql, db_result)


if __name__ == "__main__":
    main()
