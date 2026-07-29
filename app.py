from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from metadata_rag.index import create_index
from metadata_rag.providers import HeuristicProvider, Llama2Provider
from metadata_rag.service import MetadataStudio

st.set_page_config(page_title="Atlas Metadata Studio", page_icon="⬡", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 90% 0,#164e63,#07131f 42%,#020617);color:#e6fbff}
.atlas{padding:1.7rem 2rem;border:1px solid #22d3ee55;border-radius:24px;background:linear-gradient(120deg,#0891b233,#7c3aed2b);animation:pulse 5s infinite alternate}
@keyframes pulse{to{box-shadow:0 0 45px #22d3ee33}}
@media (prefers-reduced-motion: reduce){.atlas{animation:none!important}}
[data-testid=stMetric]{background:#ffffff0a;border:1px solid #ffffff18;border-radius:16px;padding:1rem}
</style><div class="atlas"><h1>⬡ Atlas Metadata Studio</h1><p>Turn project files into searchable, reviewable metadata.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    provider_options = ["Heuristic · local", "Llama 2 · Hugging Face"]
    provider_default = 1 if os.getenv("METADATA_PROVIDER", "heuristic").lower() == "llama2" else 0
    provider_name = st.selectbox("Generator", provider_options, index=provider_default)
    backend_options = ["hash", "auto", "faiss"]
    configured_backend = os.getenv("METADATA_INDEX_BACKEND", "hash").lower()
    backend_default = backend_options.index(configured_backend) if configured_backend in backend_options else 0
    backend = st.selectbox("Retrieval", backend_options, index=backend_default, help="auto uses FAISS when its model can load")
    extractor_options = {
        "Native · local PDF/DOCX/code": "native",
        "Apache Tika · Java parser": "tika",
        "Amazon Textract · hosted OCR": "textract",
    }
    configured_extractor = os.getenv("METADATA_EXTRACTOR", "native")
    extractor_default = next((label for label, value in extractor_options.items() if value == configured_extractor), next(iter(extractor_options)))
    extractor_label = st.selectbox("Extraction adapter", list(extractor_options), index=list(extractor_options).index(extractor_default))
    extractor = extractor_options[extractor_label]
    quantization = st.selectbox("Llama precision", ["4bit", "8bit", "none"], index=["4bit", "8bit", "none"].index(os.getenv("LLAMA_QUANTIZATION", "4bit")) if os.getenv("LLAMA_QUANTIZATION", "4bit") in {"4bit", "8bit", "none"} else 0, disabled=not provider_name.startswith("Llama"))
    device = st.selectbox("Llama device", ["auto", "cuda", "cpu"], index=["auto", "cuda", "cpu"].index(os.getenv("LLAMA_DEVICE", "auto")) if os.getenv("LLAMA_DEVICE", "auto") in {"auto", "cuda", "cpu"} else 0, disabled=not provider_name.startswith("Llama"))
    allow_cpu = st.checkbox("Acknowledge high-memory CPU loading", value=os.getenv("LLAMA_ALLOW_CPU", "false").lower() == "true", disabled=device != "cpu" or not provider_name.startswith("Llama"))
    max_file_mb = st.number_input("Maximum file size (MiB)", min_value=1, max_value=100, value=int(os.getenv("METADATA_MAX_FILE_MB", "20")))
    st.caption("Quantized Llama uses CUDA. CPU loading is unquantized and requires an explicit high-memory acknowledgement.")

uploads = st.file_uploader("Upload project documents or code", accept_multiple_files=True, type=["txt", "md", "py", "js", "ts", "json", "csv", "sql", "yaml", "yml", "pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff"])
if uploads and st.button("Generate metadata", type="primary", use_container_width=True):
    try:
        provider = (
            Llama2Provider(
                os.getenv("LLAMA_MODEL_ID", "meta-llama/Llama-2-13b-chat-hf"),
                os.getenv("HUGGINGFACE_TOKEN"),
                quantization=quantization,
                device=device,
                allow_cpu=allow_cpu,
                gpu_memory_gib=int(os.getenv("LLAMA_GPU_MEMORY_GIB", "0")) or None,
                cpu_memory_gib=int(os.getenv("LLAMA_CPU_MEMORY_GIB", "0")) or None,
                offload_folder=os.getenv("LLAMA_OFFLOAD_DIR") or None,
            )
            if provider_name.startswith("Llama")
            else HeuristicProvider()
        )
        studio = MetadataStudio(provider, create_index(backend, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")))
        assets = []
        for upload in uploads:
            with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as handle:
                handle.write(upload.getvalue())
                path = Path(handle.name)
            try:
                asset = studio.extract(path, extractor=extractor, max_bytes=int(max_file_mb) * 1024 * 1024)
                asset.name, asset.source = upload.name, "upload"
                assets.append(asset)
            finally:
                path.unlink(missing_ok=True)
        studio.add(assets)
        records = [studio.generate(asset).to_dict() for asset in assets]
        st.session_state.studio, st.session_state.records = studio, records
    except Exception as exc:
        st.error(f"Generation failed: {exc}")

records = st.session_state.get("records", [])
if records:
    a, b, c = st.columns(3)
    a.metric("Assets", len(records)); b.metric("Keywords", sum(len(item["keywords"]) for item in records)); c.metric("Generator", records[0]["generator"])
    for record in records:
        with st.expander(f"{record['title']} · {record['confidence']:.0%}", expanded=len(records) == 1):
            st.write(record["summary"])
            st.markdown(" ".join(f"`{word}`" for word in record["keywords"]))
            st.json(record)
    serialized = json.dumps(records, indent=2, ensure_ascii=False)
    st.download_button("Download metadata.json", serialized, "metadata.json", "application/json", use_container_width=True)
    question = st.text_input("Ask across indexed assets")
    if question:
        result = st.session_state.studio.ask(question)
        st.info(result.answer)
        st.caption(f"{result.provider} · grounded: {result.grounded} · evidence assets: " + (", ".join(result.source_ids) or "none"))
        for warning in result.warnings:
            st.warning(warning)
else:
    st.info("Upload one or more supported files. The local generator needs no API key or model download.")
