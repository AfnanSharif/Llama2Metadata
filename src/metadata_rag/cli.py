from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .index import create_index
from .providers import HeuristicProvider, Llama2Provider
from .service import MetadataStudio


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Generate structured metadata with retrieval context")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--provider", choices=["heuristic", "llama2"], default=os.getenv("METADATA_PROVIDER", "heuristic"))
    parser.add_argument("--index", choices=["auto", "faiss", "hash"], default=os.getenv("METADATA_INDEX_BACKEND", "auto"))
    parser.add_argument("--extractor", choices=["native", "tika", "textract"], default=os.getenv("METADATA_EXTRACTOR", "native"))
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default=os.getenv("LLAMA_QUANTIZATION", "4bit"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.getenv("LLAMA_DEVICE", "auto"))
    parser.add_argument("--allow-cpu", action="store_true", default=os.getenv("LLAMA_ALLOW_CPU", "false").lower() == "true")
    parser.add_argument("--gpu-memory-gib", type=int, default=int(os.getenv("LLAMA_GPU_MEMORY_GIB", "0")) or None)
    parser.add_argument("--cpu-memory-gib", type=int, default=int(os.getenv("LLAMA_CPU_MEMORY_GIB", "0")) or None)
    parser.add_argument("--ask", help="Run grounded provider Q&A after indexing")
    parser.add_argument("--qa-output", type=Path, help="Optional JSON destination for --ask")
    parser.add_argument("--output", type=Path, default=Path("artifacts/metadata.json"))
    parser.add_argument("--max-file-mb", type=int, default=int(os.getenv("METADATA_MAX_FILE_MB", "20")))
    args = parser.parse_args()
    if args.max_file_mb < 1:
        parser.error("--max-file-mb must be positive")
    provider = (
        Llama2Provider(
            os.getenv("LLAMA_MODEL_ID", "meta-llama/Llama-2-13b-chat-hf"),
            os.getenv("HUGGINGFACE_TOKEN"),
            quantization=args.quantization,
            device=args.device,
            allow_cpu=args.allow_cpu,
            gpu_memory_gib=args.gpu_memory_gib,
            cpu_memory_gib=args.cpu_memory_gib,
            offload_folder=os.getenv("LLAMA_OFFLOAD_DIR") or None,
        )
        if args.provider == "llama2"
        else HeuristicProvider()
    )
    studio = MetadataStudio(provider, create_index(args.index, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")))
    assets = studio.ingest(args.paths, extractor=args.extractor, max_bytes=args.max_file_mb * 1024 * 1024)
    payload = [studio.generate(asset).to_dict() for asset in assets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(payload)} metadata records to {args.output}")
    if args.ask:
        qa_payload = studio.ask(args.ask).to_dict()
        rendered = json.dumps(qa_payload, indent=2, ensure_ascii=False)
        if args.qa_output:
            args.qa_output.parent.mkdir(parents=True, exist_ok=True)
            args.qa_output.write_text(rendered, encoding="utf-8")
            print(f"Wrote grounded answer to {args.qa_output}")
        else:
            print(rendered)


if __name__ == "__main__":
    main()
