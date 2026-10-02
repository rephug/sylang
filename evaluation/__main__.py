"""Command line entry point; explicit local assets only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import DEFAULT_SUITE, EvaluationError, comprehension_tasks, evaluate, score_answers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--fixture-manifest", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Measure exact round trips and optional local tokenizers")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--tokenizer-manifest", type=Path)
    run.add_argument("--repetitions", type=int, default=5)
    export = commands.add_parser("export-comprehension", help="Export prompts without invoking models")
    export.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score-comprehension", help="Score locally supplied JSONL responses")
    score.add_argument("--answers", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = evaluate(args.suite, args.fixture_manifest, args.tokenizer_manifest, args.repetitions)
        elif args.command == "export-comprehension":
            tasks, _ = comprehension_tasks(args.suite, args.fixture_manifest)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
            print(f"Wrote {len(tasks)} tasks to {args.output}; no model execution.")
            return 0
        else:
            result = score_answers(args.answers, args.suite, args.fixture_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}; no model execution.")
        if args.command == "run" and not all(row["semantic_roundtrip_exact"] for row in result["rows"]):
            return 1
        return 0
    except (EvaluationError, OSError, ValueError, ImportError) as error:
        parser.exit(2, f"evaluation: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
