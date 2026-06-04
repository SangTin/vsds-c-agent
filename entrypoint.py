import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from src.io_utils import discover_input, write_predictions
from src.loader import load_questions
from src.pipeline import answer_question


DEFAULT_CONFIG: dict[str, Any] = {"output": {"fallback_answer": "A"}}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bang C answer pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return DEFAULT_CONFIG
    try:
        import yaml
    except ImportError:
        return DEFAULT_CONFIG

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        return DEFAULT_CONFIG

    output = loaded.get("output")
    if not isinstance(output, dict):
        return DEFAULT_CONFIG
    fallback = output.get("fallback_answer")
    if not isinstance(fallback, str):
        return DEFAULT_CONFIG
    return {"output": {"fallback_answer": fallback}}


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    try:
        args = parse_args(argv)
        config = load_config(args.config)
        input_path = discover_input(args.data_dir)
        if args.verbose:
            print(f"Input: {input_path}")
            print(f"Config: {args.config if args.config.is_file() else 'defaults'}")

        questions = load_questions(input_path)
        print(f"Questions read: {len(questions)}")

        fallback = config["output"]["fallback_answer"]
        predictions = [answer_question(question, fallback) for question in questions]
        output_path = args.output_dir / "pred.csv"
        write_predictions(predictions, output_path)

        elapsed = time.perf_counter() - started
        print(f"Output: {output_path}")
        print(f"Elapsed seconds: {elapsed:.3f}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
