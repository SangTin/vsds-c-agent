from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

from src.io_utils import discover_input, write_predictions
from src.llm import LLMAnswerer
from src.loader import load_questions
from src.pipeline import answer_question
from src.rag.embedder import BGEEmbedder
from src.rag.retriever import FaissRetriever
from src.router import is_math_question


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "path": "models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "n_gpu_layers": 20,
        "n_ctx": 8192,
        "seed": 42,
        "use_cot": False,
        "use_pot_ranking": False,
        "cot_max_tokens": 200,
        "use_cot_passage": False,
        "cot_passage_max_chars": 3500,
        "use_self_verify": False,
        "use_tools": False,
        "tool_timeout": 5.0,
    },
    "rag": {
        "enabled": False,
        "index_path": "data_kb/viwiki/index.faiss",
        "metadata_path": "data_kb/viwiki/metadata.jsonl",
        "top_k": 3,
        "device": "cpu",
        "model_name": "BAAI/bge-m3",
    },
    "legal_rag": {
        "enabled": False,
        "index_path": "data_kb/legal/index.faiss",
        "metadata_path": "data_kb/legal/metadata.jsonl",
        "top_k": 3,
        "min_score": 0.0,
        "device": "cpu",
        "model_name": "BAAI/bge-m3",
    },
    "polysci_rag": {
        "enabled": False,
        "index_path": "data_kb/polysci/index.faiss",
        "metadata_path": "data_kb/polysci/metadata.jsonl",
        "top_k": 3,
        "min_score": 0.0,
        "device": "cpu",
        "model_name": "BAAI/bge-m3",
    },
    "self_consistency": {"enabled": False},
    "output": {"fallback_answer": "A"},
}


def resolve_gpu_layers(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    env_value = os.environ.get("BANGC_N_GPU_LAYERS")
    if env_value is not None:
        return int(env_value)
    if os.path.exists("/dev/nvidia0") or shutil.which("nvidia-smi"):
        return 99
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_config(config_args.config)
    model_config = config["model"]
    legal_rag_config = config["legal_rag"]
    polysci_rag_config = config["polysci_rag"]

    parser = argparse.ArgumentParser(description="Run the Bang C answer pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=model_config["n_ctx"])
    parser.add_argument("--seed", type=int, default=model_config["seed"])
    parser.add_argument("--cot", action="store_true", default=None)
    parser.add_argument("--no-cot", action="store_false", dest="cot")
    parser.add_argument("--pot-ranking", action="store_true", default=None)
    parser.add_argument("--no-pot-ranking", action="store_false", dest="pot_ranking")
    parser.add_argument("--cot-max-tokens", type=int, default=None)
    parser.add_argument("--cot-passage", action="store_true", default=None)
    parser.add_argument("--no-cot-passage", action="store_false", dest="cot_passage")
    parser.add_argument("--cot-passage-max-chars", type=int, default=None)
    parser.add_argument("--self-verify", action="store_true", default=None)
    parser.add_argument(
        "--no-self-verify",
        action="store_false",
        dest="self_verify",
    )
    parser.add_argument("--tools", action="store_true", default=None)
    parser.add_argument("--no-tools", action="store_false", dest="tools")
    parser.add_argument(
        "--tool-timeout",
        type=float,
        default=model_config["tool_timeout"],
    )
    parser.add_argument("--rag", action="store_true", default=None)
    parser.add_argument("--no-rag", action="store_false", dest="rag")
    parser.add_argument("--rag-index", type=Path, default=None)
    parser.add_argument("--rag-metadata", type=Path, default=None)
    parser.add_argument("--rag-top-k", type=int, default=None)
    parser.add_argument("--rag-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--legal-rag", action="store_true", default=None)
    parser.add_argument("--no-legal-rag", action="store_false", dest="legal_rag")
    parser.add_argument(
        "--legal-index",
        type=Path,
        default=Path(legal_rag_config["index_path"]),
    )
    parser.add_argument(
        "--legal-metadata",
        type=Path,
        default=Path(legal_rag_config["metadata_path"]),
    )
    parser.add_argument("--legal-top-k", type=int, default=legal_rag_config["top_k"])
    parser.add_argument(
        "--legal-min-score",
        type=float,
        default=legal_rag_config["min_score"],
    )
    parser.add_argument(
        "--legal-device",
        choices=("cpu", "cuda"),
        default=legal_rag_config["device"],
    )
    parser.add_argument("--polysci-rag", action="store_true", default=None)
    parser.add_argument("--no-polysci-rag", action="store_false", dest="polysci_rag")
    parser.add_argument(
        "--polysci-index",
        type=Path,
        default=Path(polysci_rag_config["index_path"]),
    )
    parser.add_argument(
        "--polysci-metadata",
        type=Path,
        default=Path(polysci_rag_config["metadata_path"]),
    )
    parser.add_argument(
        "--polysci-top-k",
        type=int,
        default=polysci_rag_config["top_k"],
    )
    parser.add_argument(
        "--polysci-min-score",
        type=float,
        default=polysci_rag_config["min_score"],
    )
    parser.add_argument(
        "--polysci-device",
        choices=("cpu", "cuda"),
        default=polysci_rag_config["device"],
    )
    parser.add_argument("--self-consistency", action="store_true", default=None)
    parser.add_argument(
        "--no-self-consistency",
        action="store_false",
        dest="self_consistency",
    )
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
    legal_rag = loaded.get("legal_rag")
    polysci_rag = loaded.get("polysci_rag")
    self_consistency = loaded.get("self_consistency")
    output = loaded.get("output")
    model_defaults = DEFAULT_CONFIG["model"]
    rag_defaults = DEFAULT_CONFIG["rag"]
    legal_rag_defaults = DEFAULT_CONFIG["legal_rag"]
    polysci_rag_defaults = DEFAULT_CONFIG["polysci_rag"]
    self_consistency_defaults = DEFAULT_CONFIG["self_consistency"]
    output_defaults = DEFAULT_CONFIG["output"]
    if not isinstance(model, dict):
        model = {}
    if not isinstance(output, dict):
        output = {}
    if not isinstance(rag, dict):
        rag = {}
    if not isinstance(legal_rag, dict):
        legal_rag = {}
    if not isinstance(polysci_rag, dict):
        polysci_rag = {}
    if not isinstance(self_consistency, dict):
        self_consistency = {}

    path = model.get("path", model_defaults["path"])
    n_gpu_layers = model.get("n_gpu_layers", model_defaults["n_gpu_layers"])
    n_ctx = model.get("n_ctx", model_defaults["n_ctx"])
    seed = model.get("seed", model_defaults["seed"])
    use_cot = model.get("use_cot", model_defaults["use_cot"])
    use_pot_ranking = model.get(
        "use_pot_ranking",
        model_defaults["use_pot_ranking"],
    )
    cot_max_tokens = model.get("cot_max_tokens", model_defaults["cot_max_tokens"])
    use_cot_passage = model.get(
        "use_cot_passage",
        model_defaults["use_cot_passage"],
    )
    cot_passage_max_chars = model.get(
        "cot_passage_max_chars",
        model_defaults["cot_passage_max_chars"],
    )
    use_self_verify = model.get(
        "use_self_verify",
        model_defaults["use_self_verify"],
    )
    use_tools = model.get("use_tools", model_defaults["use_tools"])
    tool_timeout = model.get("tool_timeout", model_defaults["tool_timeout"])
    fallback = output.get("fallback_answer", output_defaults["fallback_answer"])
    rag_enabled = rag.get("enabled", rag_defaults["enabled"])
    index_path = rag.get("index_path", rag_defaults["index_path"])
    metadata_path = rag.get("metadata_path", rag_defaults["metadata_path"])
    top_k = rag.get("top_k", rag_defaults["top_k"])
    device = rag.get("device", rag_defaults["device"])
    model_name = rag.get("model_name", rag_defaults["model_name"])
    legal_rag_enabled = legal_rag.get("enabled", legal_rag_defaults["enabled"])
    legal_index_path = legal_rag.get(
        "index_path",
        legal_rag_defaults["index_path"],
    )
    legal_metadata_path = legal_rag.get(
        "metadata_path",
        legal_rag_defaults["metadata_path"],
    )
    legal_top_k = legal_rag.get("top_k", legal_rag_defaults["top_k"])
    legal_min_score = legal_rag.get(
        "min_score",
        legal_rag_defaults["min_score"],
    )
    legal_device = legal_rag.get("device", legal_rag_defaults["device"])
    legal_model_name = legal_rag.get("model_name", legal_rag_defaults["model_name"])
    polysci_rag_enabled = polysci_rag.get(
        "enabled",
        polysci_rag_defaults["enabled"],
    )
    polysci_index_path = polysci_rag.get(
        "index_path",
        polysci_rag_defaults["index_path"],
    )
    polysci_metadata_path = polysci_rag.get(
        "metadata_path",
        polysci_rag_defaults["metadata_path"],
    )
    polysci_top_k = polysci_rag.get("top_k", polysci_rag_defaults["top_k"])
    polysci_min_score = polysci_rag.get(
        "min_score",
        polysci_rag_defaults["min_score"],
    )
    polysci_device = polysci_rag.get("device", polysci_rag_defaults["device"])
    polysci_model_name = polysci_rag.get(
        "model_name",
        polysci_rag_defaults["model_name"],
    )
    self_consistency_enabled = self_consistency.get(
        "enabled",
        self_consistency_defaults["enabled"],
    )
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
    if not isinstance(use_pot_ranking, bool):
        use_pot_ranking = model_defaults["use_pot_ranking"]
    if not isinstance(cot_max_tokens, int):
        cot_max_tokens = model_defaults["cot_max_tokens"]
    if not isinstance(use_cot_passage, bool):
        use_cot_passage = model_defaults["use_cot_passage"]
    if not isinstance(cot_passage_max_chars, int) or cot_passage_max_chars < 1:
        cot_passage_max_chars = model_defaults["cot_passage_max_chars"]
    if not isinstance(use_self_verify, bool):
        use_self_verify = model_defaults["use_self_verify"]
    if not isinstance(use_tools, bool):
        use_tools = model_defaults["use_tools"]
    if not isinstance(tool_timeout, (int, float)) or tool_timeout <= 0:
        tool_timeout = model_defaults["tool_timeout"]
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
    if not isinstance(legal_rag_enabled, bool):
        legal_rag_enabled = legal_rag_defaults["enabled"]
    if not isinstance(legal_index_path, str):
        legal_index_path = legal_rag_defaults["index_path"]
    if not isinstance(legal_metadata_path, str):
        legal_metadata_path = legal_rag_defaults["metadata_path"]
    if not isinstance(legal_top_k, int) or legal_top_k < 1:
        legal_top_k = legal_rag_defaults["top_k"]
    if not isinstance(legal_min_score, (int, float)):
        legal_min_score = legal_rag_defaults["min_score"]
    if not isinstance(legal_device, str) or legal_device not in {"cpu", "cuda"}:
        legal_device = legal_rag_defaults["device"]
    if not isinstance(legal_model_name, str):
        legal_model_name = legal_rag_defaults["model_name"]
    if not isinstance(polysci_rag_enabled, bool):
        polysci_rag_enabled = polysci_rag_defaults["enabled"]
    if not isinstance(polysci_index_path, str):
        polysci_index_path = polysci_rag_defaults["index_path"]
    if not isinstance(polysci_metadata_path, str):
        polysci_metadata_path = polysci_rag_defaults["metadata_path"]
    if not isinstance(polysci_top_k, int) or polysci_top_k < 1:
        polysci_top_k = polysci_rag_defaults["top_k"]
    if not isinstance(polysci_min_score, (int, float)):
        polysci_min_score = polysci_rag_defaults["min_score"]
    if not isinstance(polysci_device, str) or polysci_device not in {"cpu", "cuda"}:
        polysci_device = polysci_rag_defaults["device"]
    if not isinstance(polysci_model_name, str):
        polysci_model_name = polysci_rag_defaults["model_name"]
    if not isinstance(self_consistency_enabled, bool):
        self_consistency_enabled = self_consistency_defaults["enabled"]

    return {
        "model": {
            "path": path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "seed": seed,
            "use_cot": use_cot,
            "use_pot_ranking": use_pot_ranking,
            "cot_max_tokens": cot_max_tokens,
            "use_cot_passage": use_cot_passage,
            "cot_passage_max_chars": cot_passage_max_chars,
            "use_self_verify": use_self_verify,
            "use_tools": use_tools,
            "tool_timeout": float(tool_timeout),
        },
        "rag": {
            "enabled": rag_enabled,
            "index_path": index_path,
            "metadata_path": metadata_path,
            "top_k": top_k,
            "device": device,
            "model_name": model_name,
        },
        "legal_rag": {
            "enabled": legal_rag_enabled,
            "index_path": legal_index_path,
            "metadata_path": legal_metadata_path,
            "top_k": legal_top_k,
            "min_score": float(legal_min_score),
            "device": legal_device,
            "model_name": legal_model_name,
        },
        "polysci_rag": {
            "enabled": polysci_rag_enabled,
            "index_path": polysci_index_path,
            "metadata_path": polysci_metadata_path,
            "top_k": polysci_top_k,
            "min_score": float(polysci_min_score),
            "device": polysci_device,
            "model_name": polysci_model_name,
        },
        "self_consistency": {"enabled": self_consistency_enabled},
        "output": {"fallback_answer": fallback},
    }


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    try:
        args = parse_args(argv)
        config = load_config(args.config)
        model_path = args.model_path or Path(config["model"]["path"])
        use_cot = config["model"]["use_cot"] if args.cot is None else args.cot
        use_pot_ranking = (
            args.pot_ranking
            if args.pot_ranking is not None
            else config["model"]["use_pot_ranking"]
        )
        cot_max_tokens = (
            config["model"]["cot_max_tokens"]
            if args.cot_max_tokens is None
            else args.cot_max_tokens
        )
        use_cot_passage = (
            config["model"]["use_cot_passage"]
            if args.cot_passage is None
            else args.cot_passage
        )
        cot_passage_max_chars = (
            config["model"]["cot_passage_max_chars"]
            if args.cot_passage_max_chars is None
            else args.cot_passage_max_chars
        )
        use_self_verify = (
            args.self_verify
            if args.self_verify is not None
            else config["model"]["use_self_verify"]
        )
        use_tools = config["model"]["use_tools"] if args.tools is None else args.tools
        tool_timeout = args.tool_timeout
        n_gpu_layers = resolve_gpu_layers(args.n_gpu_layers)
        rag_enabled = config["rag"]["enabled"] if args.rag is None else args.rag
        rag_index = args.rag_index or Path(config["rag"]["index_path"])
        rag_metadata = args.rag_metadata or Path(config["rag"]["metadata_path"])
        rag_top_k = (
            config["rag"]["top_k"] if args.rag_top_k is None else args.rag_top_k
        )
        rag_device = args.rag_device or config["rag"]["device"]
        rag_model_name = config["rag"]["model_name"]
        legal_rag_enabled = (
            config["legal_rag"]["enabled"]
            if args.legal_rag is None
            else args.legal_rag
        )
        legal_index = args.legal_index
        legal_metadata = args.legal_metadata
        legal_top_k = args.legal_top_k
        legal_min_score = args.legal_min_score
        legal_device = args.legal_device
        legal_model_name = config["legal_rag"]["model_name"]
        polysci_rag_enabled = (
            config["polysci_rag"]["enabled"]
            if args.polysci_rag is None
            else args.polysci_rag
        )
        polysci_index = args.polysci_index
        polysci_metadata = args.polysci_metadata
        polysci_top_k = args.polysci_top_k
        polysci_min_score = args.polysci_min_score
        polysci_device = args.polysci_device
        polysci_model_name = config["polysci_rag"]["model_name"]
        self_consistency_enabled = (
            args.self_consistency
            if args.self_consistency is not None
            else config["self_consistency"]["enabled"]
        )
        if self_consistency_enabled and use_self_verify:
            raise ValueError("--self-verify cannot be combined with --self-consistency")
        input_path = discover_input(args.data_dir)
        if args.verbose:
            print(f"Input: {input_path}")
            print(f"Config: {args.config if args.config.is_file() else 'defaults'}")
            print(f"Model path: {model_path}")
            print(f"n_gpu_layers: {n_gpu_layers}")
            print(f"n_ctx: {args.n_ctx}")
            print(f"seed: {args.seed}")
            print(f"use_cot: {use_cot}")
            print(f"use_pot_ranking: {use_pot_ranking}")
            print(f"cot_max_tokens: {cot_max_tokens}")
            print(f"use_cot_passage: {use_cot_passage}")
            print(f"cot_passage_max_chars: {cot_passage_max_chars}")
            print(f"use_self_verify: {use_self_verify}")
            print(f"use_tools: {use_tools}")
            print(f"tool_timeout: {tool_timeout}")
            print(f"RAG requested: {rag_enabled}")
            print(f"RAG top_k: {rag_top_k}")
            print(f"RAG index path: {rag_index}")
            print(f"RAG embedder device: {rag_device}")
            print(f"Legal RAG requested: {legal_rag_enabled}")
            print(f"Legal RAG top_k: {legal_top_k}")
            print(f"Legal RAG min_score: {legal_min_score}")
            print(f"Legal RAG index path: {legal_index}")
            print(f"Legal RAG embedder device: {legal_device}")
            print(f"Polysci RAG requested: {polysci_rag_enabled}")
            print(f"Polysci RAG top_k: {polysci_top_k}")
            print(f"Polysci RAG min_score: {polysci_min_score}")
            print(f"Polysci RAG index path: {polysci_index}")
            print(f"Polysci RAG embedder device: {polysci_device}")
            print(f"Self-consistency enabled: {self_consistency_enabled}")

        questions = load_questions(input_path)
        print(f"Questions read: {len(questions)}")

        llm: LLMAnswerer | None = None
        if model_path.is_file():
            try:
                llm = LLMAnswerer(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
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

        legal_retriever: FaissRetriever | None = None
        if legal_rag_enabled:
            try:
                if not legal_index.is_file():
                    raise FileNotFoundError(f"index file not found: {legal_index}")
                if not legal_metadata.is_file():
                    raise FileNotFoundError(
                        f"metadata file not found: {legal_metadata}"
                    )
                legal_embedder = BGEEmbedder(
                    model_name=legal_model_name,
                    device=legal_device,
                    verbose=args.verbose,
                )
                legal_retriever = FaissRetriever(
                    index_path=legal_index,
                    metadata_path=legal_metadata,
                    embedder=legal_embedder,
                    top_k=legal_top_k,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print(
                    f"Legal RAG disabled, falling back to direct LLM: {exc}",
                    file=sys.stderr,
                )
        if args.verbose:
            print(f"Legal RAG enabled: {legal_retriever is not None}")

        polysci_retriever: FaissRetriever | None = None
        if polysci_rag_enabled:
            try:
                if not polysci_index.is_file():
                    raise FileNotFoundError(f"index file not found: {polysci_index}")
                if not polysci_metadata.is_file():
                    raise FileNotFoundError(
                        f"metadata file not found: {polysci_metadata}"
                    )
                polysci_embedder = BGEEmbedder(
                    model_name=polysci_model_name,
                    device=polysci_device,
                    verbose=args.verbose,
                )
                polysci_retriever = FaissRetriever(
                    index_path=polysci_index,
                    metadata_path=polysci_metadata,
                    embedder=polysci_embedder,
                    top_k=polysci_top_k,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print(
                    f"Polysci RAG disabled, falling back to direct LLM: {exc}",
                    file=sys.stderr,
                )
        if args.verbose:
            print(f"Polysci RAG enabled: {polysci_retriever is not None}")

        tool_runner: Callable[[str], dict[str, Any]] | None = None
        if use_tools:
            from src.tools.sandbox import run_python

            tool_runner = lambda code: run_python(code, timeout=tool_timeout)

        fallback = config["output"]["fallback_answer"]
        predictions = []
        for question in questions:
            if (
                args.verbose
                and use_tools
                and llm is not None
                and is_math_question(question.question, question.choices)
            ):
                print(f"Tool path: {question.qid}", file=sys.stderr)
            predictions.append(
                answer_question(
                    question,
                    fallback,
                    llm,
                    retriever,
                    use_tools=use_tools,
                    run_python=tool_runner,
                    legal_retriever=legal_retriever,
                    legal_min_score=legal_min_score,
                    polysci_retriever=polysci_retriever,
                    polysci_min_score=polysci_min_score,
                    self_consistency=self_consistency_enabled,
                    use_pot_ranking=use_pot_ranking,
                    use_cot_passage=use_cot_passage,
                    cot_passage_max_chars=cot_passage_max_chars,
                    use_self_verify=use_self_verify,
                )
            )
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
