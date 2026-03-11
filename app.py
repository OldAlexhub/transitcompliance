from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from chromadb.config import Settings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

SYSTEM_PROMPT = """You are a transit compliance assistant for U.S. public transportation professionals.
Answer only from the provided regulatory excerpts.
If the excerpts do not fully answer the question, say that the provided documents do not contain enough information.
Every substantive sentence must include a citation in this exact style: [49 CFR Part 37, p. 12].
Do not use outside knowledge and do not invent section numbers, page numbers, or policy details."""

USER_PROMPT = """Use only the following excerpts from the provided regulations.

Question:
{question}

Excerpts:
{context}

Write a concise answer with citations after each substantive sentence."""

EXAMPLE_PROMPTS = [
    "What accessibility requirements apply to lifts and ramps on transit vehicles?",
    "When must a transit provider offer complementary paratransit service under Part 37?",
    "What securement and wheelchair space requirements are covered in Part 38?",
]


def resolve_runtime_device() -> tuple[str, str | None]:
    if not torch.cuda.is_available():
        return "cpu", "CUDA is not available in the current PyTorch runtime."

    try:
        capability = torch.cuda.get_device_capability(0)
        capability_tag = f"sm_{capability[0]}{capability[1]}"
        supported_arches = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
        if supported_arches and capability_tag not in supported_arches:
            return (
                "cpu",
                f"Detected GPU capability {capability_tag}, but this PyTorch build only supports: "
                + ", ".join(sorted(supported_arches)),
            )

        # Force a tiny CUDA op so unsupported kernels fail here instead of during embedding.
        torch.zeros(1, device="cuda")
        return "cuda", None
    except Exception as error:
        return "cpu", f"CUDA initialization failed: {error}"


def detect_device() -> str:
    return resolve_runtime_device()[0]


def format_timestamp(value: str | None) -> str:
    if not value:
        return "Unknown"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return value


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. Run code.ipynb first to build the artifacts."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_chunk_documents() -> list[Document]:
    manifest = load_manifest()
    chunks_path = Path(manifest["paths"]["chunks_path"])
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing {chunks_path}. Run code.ipynb first to build the artifacts."
        )

    documents: list[Document] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            documents.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"],
                )
            )
    return documents


class TransitComplianceBot:
    def __init__(self) -> None:
        self.manifest = load_manifest()
        self.device, self.device_warning = resolve_runtime_device()
        self.documents = load_chunk_documents()
        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            self.manifest["models"]["generator_dir"],
            trust_remote_code=False,
        )
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token

        self.embeddings = self._build_embeddings()
        self.vector_store = self._build_vector_store()
        self.hybrid_retriever = self._build_hybrid_retriever()
        self.reranker = self._build_reranker()
        self.llm = self._build_llm()

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.manifest["models"]["embedding_dir"],
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _build_vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.manifest["retrieval"]["collection_name"],
            persist_directory=self.manifest["paths"]["vector_db_dir"],
            embedding_function=self.embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )

    def _build_hybrid_retriever(self) -> EnsembleRetriever:
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.manifest["retrieval"]["dense_top_k"],
                "fetch_k": self.manifest["retrieval"]["dense_fetch_k"],
            },
        )
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.manifest["retrieval"]["bm25_top_k"]

        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3],
            id_key="chunk_id",
        )

    def _build_reranker(self) -> CrossEncoderReranker:
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=self.manifest["models"]["reranker_dir"],
            model_kwargs={"device": self.device},
        )
        return CrossEncoderReranker(
            model=cross_encoder,
            top_n=self.manifest["retrieval"]["rerank_top_n"],
        )

    def _build_llm(self) -> HuggingFacePipeline:
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.manifest["models"]["generator_dir"],
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        text_generation = pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.generator_tokenizer,
            max_new_tokens=self.manifest["generation"]["max_new_tokens"],
            do_sample=self.manifest["generation"]["do_sample"],
            temperature=self.manifest["generation"]["temperature"],
            repetition_penalty=1.05,
            return_full_text=False,
            pad_token_id=self.generator_tokenizer.eos_token_id,
        )
        return HuggingFacePipeline(pipeline=text_generation)

    def retrieve(self, question: str) -> list[Document]:
        documents = self.hybrid_retriever.invoke(question)
        if not documents:
            return []
        reranked = self.reranker.compress_documents(documents, question)
        return list(reranked)[: self.manifest["retrieval"]["max_context_chunks"]]

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        blocks = []
        for index, document in enumerate(documents, start=1):
            citation = document.metadata.get("citation", "Unknown source")
            excerpt = " ".join(document.page_content.split())
            blocks.append(f"[{index}] {citation}\n{excerpt}")
        return "\n\n".join(blocks)

    def _build_prompt(self, question: str, documents: list[Document]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=question,
                    context=self._format_context(documents),
                ),
            },
        ]
        return self.generator_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _collect_sources(documents: list[Document]) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []
        for document in documents:
            citation = document.metadata.get("citation")
            if citation and citation not in seen:
                seen.add(citation)
                sources.append(citation)
        return sources

    @staticmethod
    def _serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for rank, document in enumerate(documents, start=1):
            payload.append(
                {
                    "rank": rank,
                    "citation": document.metadata.get("citation", "Unknown source"),
                    "source_title": document.metadata.get("source_title", "Unknown title"),
                    "source_file": document.metadata.get("source_file", "Unknown file"),
                    "page": document.metadata.get("page", "Unknown"),
                    "section_hint": document.metadata.get("section_hint", ""),
                    "chunk_id": document.metadata.get("chunk_id", ""),
                    "excerpt": document.page_content.strip(),
                }
            )
        return payload

    def generate_answer(self, question: str, documents: list[Document]) -> str:
        if not documents:
            return "I do not have enough information in the provided documents to answer that question."

        prompt = self._build_prompt(question, documents)
        response = self.llm.invoke(prompt)
        if isinstance(response, str):
            return response.strip()
        return str(response).strip()

    def ask(self, question: str) -> dict[str, Any]:
        documents = self.retrieve(question)
        answer = self.generate_answer(question, documents)
        return {
            "question": question,
            "answer": answer,
            "sources": self._collect_sources(documents),
            "contexts": self._serialize_documents(documents),
        }


@st.cache_resource(show_spinner="Loading local models and indexed regulations...")
def get_bot() -> TransitComplianceBot:
    return TransitComplianceBot()


def apply_theme() -> None:
    st.set_page_config(
        page_title="Transit Compliance Copilot",
        page_icon="🚍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

            :root {
                --bg-1: #f3eee2;
                --bg-2: #d2e2dc;
                --card: rgba(255, 252, 247, 0.95);
                --card-strong: rgba(255, 252, 247, 0.99);
                --surface: rgba(255, 255, 255, 0.92);
                --surface-strong: rgba(255, 255, 255, 0.98);
                --ink: #0d1d1a;
                --muted: #2f4843;
                --line: rgba(13, 29, 26, 0.16);
                --accent: #0f766e;
                --accent-2: #ca8a04;
                --shadow: 0 24px 60px rgba(13, 29, 26, 0.10);
                --focus: #0b5f59;
                --radius: 24px;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(202, 138, 4, 0.16), transparent 32%),
                    radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 28%),
                    linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
                color: var(--ink);
                font-family: "Space Grotesk", "Segoe UI Variable Display", sans-serif;
            }

            [data-testid="stHeader"] {
                background: rgba(0, 0, 0, 0);
            }

            [data-testid="stSidebar"] {
                background: rgba(250, 246, 238, 0.96);
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] * {
                font-family: "Space Grotesk", "Segoe UI Variable Display", sans-serif;
                color: var(--ink);
            }

            .stApp,
            .stApp p,
            .stApp li,
            .stApp label,
            .stApp .stMarkdown,
            .stApp [data-testid="stMarkdownContainer"] {
                color: var(--ink);
            }

            .stApp h1,
            .stApp h2,
            .stApp h3,
            .stApp h4 {
                color: var(--ink);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .hero-shell {
                position: relative;
                overflow: hidden;
                padding: 1.5rem 1.6rem 1.3rem 1.6rem;
                border: 1px solid rgba(255, 255, 255, 0.62);
                border-radius: 28px;
                background:
                    linear-gradient(135deg, rgba(255,255,255,0.97), rgba(255,255,255,0.84)),
                    linear-gradient(120deg, rgba(15,118,110,0.08), rgba(202,138,4,0.10));
                box-shadow: var(--shadow);
                backdrop-filter: blur(14px);
            }

            .hero-shell::after {
                content: "";
                position: absolute;
                inset: auto -20% -55% auto;
                width: 320px;
                height: 320px;
                background: radial-gradient(circle, rgba(15,118,110,0.18), transparent 62%);
                pointer-events: none;
            }

            .eyebrow {
                display: inline-flex;
                gap: 0.45rem;
                align-items: center;
                padding: 0.38rem 0.7rem;
                border-radius: 999px;
                background: rgba(15, 118, 110, 0.10);
                color: var(--accent);
                font-size: 0.78rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                font-weight: 700;
            }

            .hero-title {
                margin: 0.9rem 0 0.45rem 0;
                font-size: clamp(2rem, 4vw, 3.4rem);
                line-height: 0.96;
                color: var(--ink);
                font-weight: 700;
            }

            .hero-copy {
                max-width: 60rem;
                margin: 0;
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.6;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.9rem;
                margin-top: 1.2rem;
            }

            .metric-card {
                padding: 0.95rem 1rem;
                background: rgba(255,255,255,0.84);
                border: 1px solid rgba(13, 29, 26, 0.10);
                border-radius: 20px;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                font-weight: 700;
            }

            .metric-value {
                margin-top: 0.25rem;
                color: var(--ink);
                font-size: 1.2rem;
                font-weight: 700;
            }

            .source-chip {
                display: inline-block;
                margin: 0.3rem 0.35rem 0 0;
                padding: 0.38rem 0.6rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(13, 29, 26, 0.12);
                color: var(--ink);
                font-size: 0.82rem;
                font-weight: 600;
            }

            .panel-card {
                padding: 1rem 1rem 0.9rem 1rem;
                background: var(--card);
                border-radius: 22px;
                border: 1px solid rgba(13, 29, 26, 0.10);
                box-shadow: var(--shadow);
            }

            .source-card {
                padding: 0.95rem 1rem;
                border-radius: 18px;
                border: 1px solid rgba(13, 29, 26, 0.12);
                background: var(--card-strong);
                margin-bottom: 0.85rem;
            }

            .source-card h4 {
                margin: 0 0 0.2rem 0;
                color: var(--ink);
                font-size: 0.98rem;
            }

            .source-meta {
                color: var(--muted);
                font-size: 0.82rem;
                margin-bottom: 0.55rem;
            }

            .source-excerpt {
                color: var(--ink);
                font-size: 0.95rem;
                line-height: 1.55;
                white-space: pre-wrap;
            }

            .sidebar-card {
                padding: 0.95rem 1rem;
                background: rgba(255,255,255,0.90);
                border-radius: 18px;
                border: 1px solid rgba(13, 29, 26, 0.12);
                margin-bottom: 0.8rem;
            }

            .sidebar-kicker {
                color: var(--muted);
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 700;
            }

            .sidebar-title {
                color: var(--ink);
                font-size: 1rem;
                font-weight: 700;
                margin-top: 0.2rem;
            }

            code, pre {
                font-family: "IBM Plex Mono", Consolas, monospace !important;
            }

            code {
                color: #083d3a;
                background: rgba(15, 118, 110, 0.09);
                border-radius: 8px;
                padding: 0.12rem 0.32rem;
            }

            .stChatMessage {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(13, 29, 26, 0.10);
                border-radius: 24px;
                box-shadow: 0 12px 35px rgba(16, 35, 29, 0.06);
            }

            .stChatMessage [data-testid="stMarkdownContainer"],
            .stChatMessage p,
            .stChatMessage li,
            .stChatMessage span {
                color: var(--ink);
            }

            .stButton > button, .stDownloadButton > button {
                border-radius: 999px;
                border: 1px solid rgba(13, 29, 26, 0.14);
                background: var(--surface-strong);
                color: var(--ink);
                font-weight: 600;
                transition: transform 120ms ease, box-shadow 120ms ease;
            }

            .stButton > button:hover, .stDownloadButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 25px rgba(16, 35, 29, 0.08);
                border-color: rgba(15,118,110,0.28);
            }

            .stButton > button:focus,
            .stDownloadButton > button:focus,
            .stChatInput textarea:focus {
                outline: 2px solid var(--focus) !important;
                outline-offset: 1px !important;
                box-shadow: 0 0 0 3px rgba(11, 95, 89, 0.14) !important;
            }

            [data-baseweb="input"] textarea,
            [data-baseweb="textarea"] textarea {
                background: var(--surface-strong) !important;
                color: var(--ink) !important;
                border: 1px solid rgba(13, 29, 26, 0.16) !important;
            }

            .stChatInput textarea {
                font-size: 1rem;
            }

            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid rgba(13, 29, 26, 0.12);
                padding: 0.75rem 0.9rem;
                border-radius: 18px;
            }

            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"] {
                color: var(--ink);
            }

            [data-baseweb="tab-list"] {
                gap: 0.45rem;
            }

            [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(13, 29, 26, 0.10);
                border-radius: 999px;
                color: var(--muted);
                font-weight: 600;
            }

            [data-baseweb="tab"][aria-selected="true"] {
                background: rgba(15, 118, 110, 0.12);
                color: var(--ink);
                border-color: rgba(15, 118, 110, 0.26);
            }

            .stAlert {
                background: var(--surface-strong);
                color: var(--ink);
                border: 1px solid rgba(13, 29, 26, 0.14);
            }

            @media (max-width: 960px) {
                .metric-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("queued_prompt", None)


def queue_prompt(prompt: str) -> None:
    st.session_state["queued_prompt"] = prompt


def render_hero(manifest: dict[str, Any]) -> None:
    stats = manifest.get("stats", {})
    documents = manifest.get("documents", [])
    created_at = format_timestamp(manifest.get("created_at_utc"))
    device = detect_device()
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU mode"
    doc_names = "".join(
        f"<span class='source-chip'>{escape(doc.get('title', 'Unknown document'))}</span>"
        for doc in documents
    )

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="eyebrow">Transit regulation intelligence</div>
            <h1 class="hero-title">Cited answers across Part 37 and Part 38.</h1>
            <p class="hero-copy">
                Ask operational, compliance, ADA, vehicle, or paratransit questions against your locally indexed federal
                regulations. Retrieval, reranking, and generation all run from artifacts built by the notebook so the app
                does not rebuild indexes at question time.
            </p>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Indexed docs</div>
                    <div class="metric-value">{len(documents)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Chunk count</div>
                    <div class="metric-value">{stats.get('chunk_count', 'Unknown')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Compute</div>
                    <div class="metric-value">{escape(gpu_name)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Artifacts built</div>
                    <div class="metric-value">{escape(created_at)}</div>
                </div>
            </div>
            <div style="margin-top: 0.8rem;">{doc_names}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(manifest: dict[str, Any]) -> None:
    device, device_warning = resolve_runtime_device()
    generator_name = Path(manifest["models"]["generator_dir"]).name
    embedding_name = Path(manifest["models"]["embedding_dir"]).name
    reranker_name = Path(manifest["models"]["reranker_dir"]).name
    retrieval = manifest.get("retrieval", {})

    with st.sidebar:
        st.markdown("## Control Room")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-kicker">System status</div>
                <div class="sidebar-title">Ready for local retrieval</div>
                <div style="color:var(--muted); margin-top:0.35rem;">
                    Device: <strong>{escape(device.upper())}</strong><br>
                    Dense + BM25 hybrid retrieval with cross-encoder reranking.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if device_warning:
            st.warning(device_warning)

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-kicker">Model stack</div>
                <div class="sidebar-title">{escape(generator_name)}</div>
                <div style="color:var(--muted); margin-top:0.35rem;">
                    Embeddings: {escape(embedding_name)}<br>
                    Reranker: {escape(reranker_name)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-kicker">Retrieval profile</div>
                <div class="sidebar-title">High-recall, citation-first</div>
                <div style="color:var(--muted); margin-top:0.35rem;">
                    Dense top-k: {retrieval.get('dense_top_k', 'n/a')}<br>
                    BM25 top-k: {retrieval.get('bm25_top_k', 'n/a')}<br>
                    Reranked context: {retrieval.get('rerank_top_n', 'n/a')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Example prompts")
        for index, prompt in enumerate(EXAMPLE_PROMPTS, start=1):
            if st.button(prompt, key=f"sidebar_example_{index}", use_container_width=True):
                queue_prompt(prompt)
                st.rerun()

        st.markdown("### Documents")
        for document in manifest.get("documents", []):
            st.markdown(
                f"""
                <div class="sidebar-card">
                    <div class="sidebar-kicker">{escape(document.get('file_name', 'Unknown file'))}</div>
                    <div class="sidebar-title">{escape(document.get('title', 'Unknown title'))}</div>
                    <div style="color:var(--muted); margin-top:0.35rem;">
                        Pages indexed: {document.get('page_count', 'Unknown')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_source_cards(contexts: list[dict[str, Any]]) -> None:
    if not contexts:
        st.info("No supporting excerpts were retrieved for this response.")
        return

    for context in contexts:
        section_hint = escape(context.get("section_hint") or "Retrieved excerpt")
        citation = escape(context.get("citation", "Unknown source"))
        source_file = escape(context.get("source_file", "Unknown file"))
        excerpt = escape(context.get("excerpt", ""))
        st.markdown(
            f"""
            <div class="source-card">
                <h4>#{context.get("rank", "?")} · {citation}</h4>
                <div class="source-meta">{section_hint} · {source_file}</div>
                <div class="source-excerpt">{excerpt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_assistant_metadata(message: dict[str, Any]) -> None:
    sources = message.get("sources", [])
    contexts = message.get("contexts", [])

    if not sources and not contexts:
        return

    tabs = st.tabs(["Source trail", "Retrieved excerpts"])
    with tabs[0]:
        if sources:
            source_badges = "".join(
                f"<span class='source-chip'>{escape(source)}</span>" for source in sources
            )
            st.markdown(source_badges, unsafe_allow_html=True)
        else:
            st.caption("No citations were available for this response.")

    with tabs[1]:
        render_source_cards(contexts)


def render_message(message: dict[str, Any]) -> None:
    avatar = "assistant" if message["role"] == "assistant" else "user"
    with st.chat_message(avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_assistant_metadata(message)


def render_setup_blocker(error_text: str) -> None:
    st.error("The retrieval artifacts are missing, so the app cannot answer questions yet.")
    st.markdown(
        f"""
        <div class="panel-card">
            <strong>Setup required</strong><br><br>
            {escape(error_text)}<br><br>
            Run the notebook first so these files exist:
            <ul>
                <li><code>artifacts/manifest.json</code></li>
                <li><code>artifacts/chunks.jsonl</code></li>
                <li><code>artifacts/chroma_db/</code></li>
            </ul>
            Then launch the app with <code>streamlit run app.py</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_actions() -> None:
    st.markdown("### Jump in")
    columns = st.columns(len(EXAMPLE_PROMPTS))
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        if columns[index].button(prompt, key=f"hero_prompt_{index}", use_container_width=True):
            queue_prompt(prompt)
            st.rerun()


def collect_prompt() -> str | None:
    queued_prompt = st.session_state.pop("queued_prompt", None)
    typed_prompt = st.chat_input(
        "Ask a transit compliance question grounded in Part 37 and Part 38..."
    )
    return typed_prompt or queued_prompt


def answer_question(question: str) -> dict[str, Any]:
    with st.status("Processing question", expanded=True) as status:
        status.write("Loading the local retrieval and generation stack.")
        bot = get_bot()
        status.write("Searching the indexed regulations with dense and lexical retrieval.")
        documents = bot.retrieve(question)
        status.write(f"Selected {len(documents)} evidence chunks for the final answer.")
        status.write("Generating a citation-grounded answer from the local language model.")
        answer = bot.generate_answer(question, documents)
        response = {
            "role": "assistant",
            "content": answer,
            "sources": bot._collect_sources(documents),
            "contexts": bot._serialize_documents(documents),
        }
        status.update(label="Answer ready", state="complete", expanded=False)
        return response


def main() -> None:
    apply_theme()
    init_session_state()

    try:
        manifest = load_manifest()
    except FileNotFoundError as error:
        render_setup_blocker(str(error))
        return

    render_sidebar(manifest)

    left, right = st.columns([3.2, 1.15], gap="large")

    with left:
        render_hero(manifest)
        render_quick_actions()

        if not st.session_state["messages"]:
            st.markdown(
                """
                <div class="panel-card" style="margin-top: 1rem;">
                    <strong>What this app does</strong><br><br>
                    It answers only from the two indexed CFR documents, shows the retrieved evidence behind each answer,
                    and keeps the expensive retrieval artifacts on disk so the app does not re-embed or rebuild indexes.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for message in st.session_state["messages"]:
            render_message(message)

        prompt = collect_prompt()
        if prompt:
            user_message = {"role": "user", "content": prompt}
            st.session_state["messages"].append(user_message)
            render_message(user_message)

            assistant_message = answer_question(prompt)
            st.session_state["messages"].append(assistant_message)
            render_message(assistant_message)

    with right:
        st.markdown("### Session view")
        st.markdown(
            """
            <div class="panel-card">
                <strong>Answer design</strong><br><br>
                Responses are expected to stay concise, grounded in the indexed text, and explicitly cited by page.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel-card" style="margin-top: 1rem;">
                <strong>Recommended usage</strong><br><br>
                Ask one operational question at a time. If you need a comparison, name both topics explicitly so retrieval
                can pull the right sections from both Part 37 and Part 38.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state["messages"]:
            assistant_count = sum(
                1 for message in st.session_state["messages"] if message["role"] == "assistant"
            )
            st.metric("Answers in session", assistant_count)
            latest_sources = []
            for message in reversed(st.session_state["messages"]):
                if message["role"] == "assistant" and message.get("sources"):
                    latest_sources = message["sources"]
                    break

            if latest_sources:
                st.markdown("### Latest citations")
                st.markdown(
                    "".join(
                        f"<span class='source-chip'>{escape(source)}</span>"
                        for source in latest_sources
                    ),
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
