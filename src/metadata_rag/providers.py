from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .index import TOKENS
from .models import Asset, GeneratedMetadata, QAResult

STOPWORDS = {"about", "after", "also", "been", "before", "being", "between", "from", "have", "into", "more", "other", "that", "their", "there", "these", "they", "this", "using", "were", "what", "when", "where", "which", "with", "your", "return", "class", "self"}


@dataclass(frozen=True, slots=True)
class LlamaRuntimeConfig:
    model_id: str
    token: str | None = None
    quantization: str = "4bit"
    device: str = "auto"
    allow_cpu: bool = False
    gpu_memory_gib: int | None = None
    cpu_memory_gib: int | None = None
    offload_folder: str | None = None

    def __post_init__(self) -> None:
        if self.quantization not in {"4bit", "8bit", "none"}:
            raise ValueError("quantization must be 4bit, 8bit, or none")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.device == "cpu" and not self.allow_cpu:
            raise ValueError("CPU Llama 2 loading requires allow_cpu=True because 13B models need substantial RAM")
        if self.device == "cpu" and self.quantization != "none":
            raise ValueError("This safe runtime only enables bitsandbytes 4/8-bit loading on CUDA")
        if self.gpu_memory_gib is not None and self.gpu_memory_gib < 1:
            raise ValueError("gpu_memory_gib must be positive")
        if self.cpu_memory_gib is not None and self.cpu_memory_gib < 1:
            raise ValueError("cpu_memory_gib must be positive")


class MetadataProvider(Protocol):
    name: str
    def generate(self, asset: Asset, related_context: str = "") -> GeneratedMetadata: ...
    def answer(self, question: str, evidence: list[Asset]) -> QAResult: ...


class HeuristicProvider:
    name = "heuristic"

    def generate(self, asset: Asset, related_context: str = "") -> GeneratedMetadata:
        words = TOKENS.findall(asset.text)
        counts = Counter(word.lower() for word in words if len(word) > 3 and word.lower() not in STOPWORDS)
        keywords = [word for word, _ in counts.most_common(10)]
        sentences = re.split(r"(?<=[.!?])\s+|\n+", asset.text.strip())
        summary = " ".join(part.strip() for part in sentences if part.strip())[:480]
        if len(asset.text) > len(summary):
            summary = summary.rstrip() + "…"
        entities = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2}\b", asset.text)))[:10]
        suffix = "." + asset.name.rsplit(".", 1)[-1].lower() if "." in asset.name else ""
        content_type = "source-code" if suffix in {".py", ".js", ".ts", ".java", ".go", ".rs", ".sql"} else "document"
        warnings = [] if summary else ["No extractable text was found"]
        return GeneratedMetadata(
            asset_id=asset.id,
            title=asset.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip().title(),
            summary=summary or "No textual summary is available.",
            keywords=keywords,
            content_type=content_type,
            language="en" if words else "unknown",
            entities=entities,
            confidence=min(0.95, 0.45 + len(words) / 1000),
            generator=self.name,
            warnings=warnings,
        )

    def answer(self, question: str, evidence: list[Asset]) -> QAResult:
        query_terms = {word.lower() for word in TOKENS.findall(question) if word.lower() not in STOPWORDS}
        candidates: list[tuple[int, str, Asset]] = []
        for asset in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", asset.text):
                sentence = sentence.strip()
                score = len(query_terms & {word.lower() for word in TOKENS.findall(sentence)})
                if sentence and score:
                    candidates.append((score, sentence, asset))
        if not candidates:
            return QAResult("The indexed files do not contain enough evidence to answer that question.", [], self.name, False, ["No matching sentence was found"])
        selected = sorted(candidates, key=lambda row: row[0], reverse=True)[:3]
        source_ids = list(dict.fromkeys(row[2].id for row in selected))
        answer = " ".join(row[1] for row in selected)
        return QAResult(answer, source_ids, self.name, True)


class Llama2Provider:
    name = "llama2"

    def __init__(
        self,
        model_id: str,
        token: str | None = None,
        *,
        quantization: str = "4bit",
        device: str = "auto",
        allow_cpu: bool = False,
        gpu_memory_gib: int | None = None,
        cpu_memory_gib: int | None = None,
        offload_folder: str | None = None,
        pipeline_loader: Callable[[LlamaRuntimeConfig], object] | None = None,
    ) -> None:
        self.config = LlamaRuntimeConfig(
            model_id=model_id,
            token=token or None,
            quantization=quantization,
            device=device,
            allow_cpu=allow_cpu,
            gpu_memory_gib=gpu_memory_gib,
            cpu_memory_gib=cpu_memory_gib,
            offload_folder=offload_folder,
        )
        self.pipeline_loader = pipeline_loader or _load_llama_pipeline
        self.pipe = None

    def _pipeline(self):
        if self.pipe is None:
            self.pipe = self.pipeline_loader(self.config)
        return self.pipe

    def _json(self, prompt: str, max_new_tokens: int) -> dict:
        output = self._pipeline()(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)[0]["generated_text"]
        start = output.find("{")
        if start < 0:
            raise ValueError("Llama 2 did not return a JSON object")
        try:
            values, _ = json.JSONDecoder().raw_decode(output[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("Llama 2 returned invalid JSON") from exc
        if not isinstance(values, dict):
            raise ValueError("Llama 2 JSON must be an object")
        return values

    def generate(self, asset: Asset, related_context: str = "") -> GeneratedMetadata:
        related_context = _safe_prompt(related_context[:2500])
        content = _safe_prompt(asset.text[:7000])
        asset_name = _safe_prompt(asset.name)
        prompt = f"""[INST] Return only valid JSON with keys title, summary, keywords, content_type, language, entities, confidence.
Create accurate metadata for {asset_name}. Use only RELATED and CONTENT; never invent facts.
RELATED:\n{related_context}\nCONTENT:\n{content} [/INST]"""
        return _validated_metadata(asset, self._json(prompt, 500), self.name)

    def answer(self, question: str, evidence: list[Asset]) -> QAResult:
        if not evidence:
            return QAResult("The index returned no evidence for that question.", [], self.name, False)
        allowed = {asset.id for asset in evidence}
        context = "\n\n".join(f"[SOURCE {asset.id} | {_safe_prompt(asset.name)}]\n{_safe_prompt(asset.text[:2500])}" for asset in evidence)
        prompt = f"""[INST] Answer the QUESTION using only the SOURCES. Return only JSON with keys answer and citations.
citations must be a list of exact SOURCE ids. If evidence is insufficient, say so and return an empty list. Do not invent facts.
QUESTION:\n{_safe_prompt(question[:2000])}\nSOURCES:\n{context} [/INST]"""
        values = self._json(prompt, 450)
        answer = values.get("answer")
        citations = values.get("citations")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Llama 2 QA answer must be a non-empty string")
        if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
            raise ValueError("Llama 2 QA citations must be a list of source ids")
        citations = list(dict.fromkeys(citations))
        invalid = set(citations) - allowed
        if invalid:
            raise ValueError(f"Llama 2 cited unknown source ids: {', '.join(sorted(invalid))}")
        return QAResult(answer.strip(), citations, self.name, bool(citations), [] if citations else ["The provider found insufficient grounded evidence"])


def _safe_prompt(value: str) -> str:
    return value.replace("[INST]", "[instruction delimiter removed]").replace("[/INST]", "[instruction delimiter removed]")


def _load_llama_pipeline(config: LlamaRuntimeConfig):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers, accelerate, torch, and bitsandbytes to use quantized Llama 2") from exc

    has_cuda = bool(torch.cuda.is_available())
    resolved_device = "cuda" if config.device == "auto" and has_cuda else config.device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not has_cuda:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if resolved_device == "cpu" and not config.allow_cpu:
        raise RuntimeError("No CUDA device is available; use heuristic mode or explicitly allow the high-memory CPU path")
    if resolved_device == "cpu" and config.quantization != "none":
        raise RuntimeError("This runtime restricts bitsandbytes 4/8-bit loading to CUDA")

    dtype = torch.float32
    if resolved_device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs: dict[str, object] = {"token": config.token, "low_cpu_mem_usage": True}
    if config.quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif config.quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = dtype
    load_kwargs["device_map"] = "auto" if config.device == "auto" else ({"": 0} if resolved_device == "cuda" else {"": "cpu"})
    max_memory: dict[object, str] = {}
    if resolved_device == "cuda" and config.gpu_memory_gib is not None:
        max_memory[0] = f"{config.gpu_memory_gib}GiB"
    if config.cpu_memory_gib is not None:
        max_memory["cpu"] = f"{config.cpu_memory_gib}GiB"
    if max_memory:
        load_kwargs["max_memory"] = max_memory
    if config.offload_folder:
        Path(config.offload_folder).mkdir(parents=True, exist_ok=True)
        load_kwargs["offload_folder"] = config.offload_folder

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=config.token)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def _validated_metadata(asset: Asset, values: object, generator: str) -> GeneratedMetadata:
    """Validate the untrusted structured output before it reaches the UI/export."""
    if not isinstance(values, dict):
        raise ValueError("Generated metadata must be a JSON object")

    def text_value(key: str) -> str:
        value = values.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Generated metadata field {key!r} must be a non-empty string")
        return value.strip()

    def string_list(key: str) -> list[str]:
        value = values.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"Generated metadata field {key!r} must be a list of non-empty strings")
        return list(dict.fromkeys(item.strip() for item in value))

    confidence = values.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Generated metadata confidence must be a finite number between 0 and 1")
    return GeneratedMetadata(
        asset_id=asset.id,
        title=text_value("title"),
        summary=text_value("summary"),
        keywords=string_list("keywords"),
        content_type=text_value("content_type"),
        language=text_value("language"),
        entities=string_list("entities"),
        confidence=float(confidence),
        generator=generator,
    )
