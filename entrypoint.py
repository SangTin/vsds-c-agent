import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from src.io_utils import discover_input, write_predictions
from src.llm import LLMAnswerer
from src.loader import load_questions
from src.pipeline import answer_question


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "path": "models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "n_gpu_layers": 20,
        "n_ctx": 8192,
        "seed": 42,
    },
    "output": {"fallback_answer": "A"},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    config_args, _ = config_parser.parse_known_args(argv)
    model_config = load_config(config_args.config)["model"]

    parser = argparse.ArgumentParser(description="Run the Bang C answer pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--n-gpu-layers", type=int, default=model_config["n_gpu_layers"]
    )
    parser.add_argument("--n-ctx", type=int, default=model_config["n_ctx"])
    parser.add_argument("--seed", type=int, default=model_config["seed"])
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

    model = loaded.get("model")
    output = loaded.get("output")
    model_defaults = DEFAULT_CONFIG["model"]
    output_defaults = DEFAULT_CONFIG["output"]
    if not isinstance(model, dict):
        model = {}
    if not isinstance(output, dict):
        output = {}

    path = model.get("path", model_defaults["path"])
    n_gpu_layers = model.get("n_gpu_layers", model_defaults["n_gpu_layers"])
    n_ctx = model.get("n_ctx", model_defaults["n_ctx"])
    seed = model.get("seed", model_defaults["seed"])
    fallback = output.get("fallback_answer", output_defaults["fallback_answer"])
    if not isinstance(path, str):
        path = model_defaults["path"]
    if not isinstance(n_gpu_layers, int):
        n_gpu_layers = model_defaults["n_gpu_layers"]
    if not isinstance(n_ctx, int):
        n_ctx = model_defaults["n_ctx"]
    if not isinstance(seed, int):
        seed = model_defaults["seed"]
    if not isinstance(fallback, str):
        fallback = output_defaults["fallback_answer"]

    return {
        "model": {
            "path": path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "seed": seed,
        },
        "output": {"fallback_answer": fallback},
    }


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    try:
        args = parse_args(argv)
        config = load_config(args.config)
        model_path = args.model_path or Path(config["model"]["path"])
        input_path = discover_input(args.data_dir)
        if args.verbose:
            print(f"Input: {input_path}")
            print(f"Config: {args.config if args.config.is_file() else 'defaults'}")
            print(f"Model path: {model_path}")
            print(f"n_gpu_layers: {args.n_gpu_layers}")
            print(f"n_ctx: {args.n_ctx}")
            print(f"seed: {args.seed}")

        questions = load_questions(input_path)
        print(f"Questions read: {len(questions)}")

        llm: LLMAnswerer | None = None
        if model_path.is_file():
            try:
                llm = LLMAnswerer(
                    model_path=model_path,
                    n_gpu_layers=args.n_gpu_layers,
                    n_ctx=args.n_ctx,
                    seed=args.seed,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print(
                    f"LLM disabled, falling back to stub: {exc}",
                    file=sys.stderr,
                )
        else:
            print(
                f"LLM disabled, falling back to stub: model file not found: {model_path}",
                file=sys.stderr,
            )

        fallback = config["output"]["fallback_answer"]
        predictions = [
            answer_question(question, fallback, llm) for question in questions
        ]
        output_path = args.output_dir / "pred.csv"
        write_predictions(predictions, output_path)

        elapsed = time.perf_counter() - started
        average = elapsed / len(questions) if questions else 0.0
        print(f"Output: {output_path}")
        print(f"Total elapsed seconds: {elapsed:.3f}")
        print(f"Average seconds per question: {average:.6f}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
