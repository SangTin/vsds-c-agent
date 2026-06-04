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
from src.rag.embedder import BGEEmbedder
from src.rag.retriever import FaissRetriever


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "path": "models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "n_gpu_layers": 20,
        "n_ctx": 8192,
        "seed": 42,
        "use_cot": False,
        "cot_max_tokens": 200,
    },
    "rag": {
        "enabled": False,
        "index_path": "data_kb/viwiki/index.faiss",
        "metadata_path": "data_kb/viwiki/metadata.jsonl",
        "top_k": 3,
        "device": "cpu",
        "model_name": "BAAI/bge-m3",
    },
    "output": {"fallback_answer": "A"},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_config(config_args.config)
    model_config = config["model"]

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
    parser.add_argument("--cot", action="store_true", default=None)
    parser.add_argument("--no-cot", action="store_false", dest="cot")
    parser.add_argument("--cot-max-tokens", type=int, default=None)
    parser.add_argument("--rag", action="store_true", default=None)
    parser.add_argument("--no-rag", action="store_false", dest="rag")
    parser.add_argument("--rag-index", type=Path, default=None)
    parser.add_argument("--rag-metadata", type=Path, default=None)
    parser.add_argument("--rag-top-k", type=int, default=None)
    parser.add_argument("--rag-device", choices=("cpu", "cuda"), default=None)
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
    rag = loaded.get("rag")
    output = loaded.get("output")
    model_defaults = DEFAULT_CONFIG["model"]
    rag_defaults = DEFAULT_CONFIG["rag"]
    output_defaults = DEFAULT_CONFIG["output"]
    if not isinstance(model, dict):
        model = {}
    if not isinstance(output, dict):
        output = {}
    if not isinstance(rag, dict):
        rag = {}

    path = model.get("path", model_defaults["path"])
    n_gpu_layers = model.get("n_gpu_layers", model_defaults["n_gpu_layers"])
    n_ctx = model.get("n_ctx", model_defaults["n_ctx"])
    seed = model.get("seed", model_defaults["seed"])
    use_cot = model.get("use_cot", model_defaults["use_cot"])
    cot_max_tokens = model.get("cot_max_tokens", model_defaults["cot_max_tokens"])
    fallback = output.get("fallback_answer", output_defaults["fallback_answer"])
    rag_enabled = rag.get("enabled", rag_defaults["enabled"])
    index_path = rag.get("index_path", rag_defaults["index_path"])
    metadata_path = rag.get("metadata_path", rag_defaults["metadata_path"])
    top_k = rag.get("top_k", rag_defaults["top_k"])
    device = rag.get("device", rag_defaults["device"])
    model_name = rag.get("model_name", rag_defaults["model_name"])
    if not isinstance(path, str):
        path = model_defaults["path"]
    if not isinstance(n_gpu_layers, int):
        n_gpu_layers = model_defaults["n_gpu_layers"]
    if not isinstance(n_ctx, int):
        n_ctx = model_defaults["n_ctx"]
    if not isinstance(seed, int):
        seed = model_defaults["seed"]
    if not isinstance(use_cot, bool):
        use_cot = model_defaults["use_cot"]
    if not isinstance(cot_max_tokens, int):
        cot_max_tokens = model_defaults["cot_max_tokens"]
    if not isinstance(fallback, str):
        fallback = output_defaults["fallback_answer"]
    if not isinstance(rag_enabled, bool):
        rag_enabled = rag_defaults["enabled"]
    if not isinstance(index_path, str):
        index_path = rag_defaults["index_path"]
    if not isinstance(metadata_path, str):
        metadata_path = rag_defaults["metadata_path"]
    if not isinstance(top_k, int) or top_k < 1:
        top_k = rag_defaults["top_k"]
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        device = rag_defaults["device"]
    if not isinstance(model_name, str):
        model_name = rag_defaults["model_name"]

    return {
        "model": {
            "path": path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "seed": seed,
            "use_cot": use_cot,
            "cot_max_tokens": cot_max_tokens,
        },
        "rag": {
            "enabled": rag_enabled,
            "index_path": index_path,
            "metadata_path": metadata_path,
            "top_k": top_k,
            "device": device,
            "model_name": model_name,
        },
        "output": {"fallback_answer": fallback},
    }


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    try:
        args = parse_args(argv)
        config = load_config(args.config)
        model_path = args.model_path or Path(config["model"]["path"])
        use_cot = config["model"]["use_cot"] if args.cot is None else args.cot
        cot_max_tokens = (
            config["model"]["cot_max_tokens"]
            if args.cot_max_tokens is None
            else args.cot_max_tokens
        )
        rag_enabled = config["rag"]["enabled"] if args.rag is None else args.rag
        rag_index = args.rag_index or Path(config["rag"]["index_path"])
        rag_metadata = args.rag_metadata or Path(config["rag"]["metadata_path"])
        rag_top_k = (
            config["rag"]["top_k"] if args.rag_top_k is None else args.rag_top_k
        )
        rag_device = args.rag_device or config["rag"]["device"]
        rag_model_name = config["rag"]["model_name"]
        input_path = discover_input(args.data_dir)
        if args.verbose:
            print(f"Input: {input_path}")
            print(f"Config: {args.config if args.config.is_file() else 'defaults'}")
            print(f"Model path: {model_path}")
            print(f"n_gpu_layers: {args.n_gpu_layers}")
            print(f"n_ctx: {args.n_ctx}")
            print(f"seed: {args.seed}")
            print(f"use_cot: {use_cot}")
            print(f"cot_max_tokens: {cot_max_tokens}")
            print(f"RAG requested: {rag_enabled}")
            print(f"RAG top_k: {rag_top_k}")
            print(f"RAG index path: {rag_index}")
            print(f"RAG embedder device: {rag_device}")

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
                    use_cot=use_cot,
                    cot_max_tokens=cot_max_tokens,
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

        retriever: FaissRetriever | None = None
        if rag_enabled:
            try:
                if not rag_index.is_file():
                    raise FileNotFoundError(f"index file not found: {rag_index}")
                if not rag_metadata.is_file():
                    raise FileNotFoundError(
                        f"metadata file not found: {rag_metadata}"
                    )
                embedder = BGEEmbedder(
                    model_name=rag_model_name,
                    device=rag_device,
                    verbose=args.verbose,
                )
                retriever = FaissRetriever(
                    index_path=rag_index,
                    metadata_path=rag_metadata,
                    embedder=embedder,
                    top_k=rag_top_k,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print(
                    f"RAG disabled, falling back to direct LLM: {exc}",
                    file=sys.stderr,
                )
        if args.verbose:
            print(f"RAG enabled: {retriever is not None}")

        fallback = config["output"]["fallback_answer"]
        predictions = [
            answer_question(question, fallback, llm, retriever)
            for question in questions
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
