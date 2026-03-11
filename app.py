from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = Path(__file__).resolve().parent

EXAMPLE_PROMPTS = [
    "What accessibility requirements apply to lifts and ramps on transit vehicles?",
    "When must a transit provider offer complementary paratransit service under Part 37?",
    "What securement and wheelchair space requirements are covered in Part 38?",
]


@dataclass(frozen=True)
class SourceSpec:
    title: str
    file_name: str


@dataclass(frozen=True)
class AppConfig:
    sources: tuple[SourceSpec, ...] = (
        SourceSpec(
            title="49 CFR Part 37",
            file_name="49 CFR Part 37 (up to date as of 3-09-2026).pdf",
        ),
        SourceSpec(
            title="49 CFR Part 38",
            file_name="CFR-2024-title49-vol1-part38.pdf",
        ),
    )
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    qa_model_id: str = "distilbert-base-cased-distilled-squad"
    chunk_size: int = 950
    chunk_overlap: int = 180
    dense_top_k: int = 8
    bm25_top_k: int = 8
    final_top_k: int = 5
    min_answer_score: float = 0.12
    max_answer_candidates: int = 3


CONFIG = AppConfig()


def resolve_runtime_device() -> tuple[str, str | None]:
    if not torch.cuda.is_available():
        return "cpu", "Running on CPU. This is expected on Streamlit Community Cloud."

    try:
        capability = torch.cuda.get_device_capability(0)
        capability_tag = f"sm_{capability[0]}{capability[1]}"
        supported_arches = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
        if supported_arches and capability_tag not in supported_arches:
            return (
                "cpu",
                f"Detected GPU capability {capability_tag}, but the runtime does not support it.",
            )

        torch.zeros(1, device="cuda")
        return "cuda", None
    except Exception as error:
        return "cpu", f"CUDA initialization failed: {error}"


def normalize_text(text: str) -> str:
    text = text.replace("\u0000", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", " ".join(text.split()))
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    return cleaned or [text.strip()]


def overlap_score(question: str, sentence: str) -> float:
    question_tokens = set(tokenize(question))
    sentence_tokens = tokenize(sentence)
    if not question_tokens or not sentence_tokens:
        return 0.0
    shared = sum(1 for token in sentence_tokens if token in question_tokens)
    return shared / max(len(question_tokens), 1)


@lru_cache(maxsize=1)
def load_source_pages() -> tuple[list[Document], list[dict[str, Any]]]:
    pages: list[Document] = []
    document_stats: list[dict[str, Any]] = []

    for source in CONFIG.sources:
        pdf_path = PROJECT_ROOT / source.file_name
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing source PDF: {pdf_path}")

        loader = PyMuPDFLoader(str(pdf_path))
        raw_pages = loader.load()
        source_key = slugify(source.title)
        page_count = 0

        for raw_page in raw_pages:
            content = normalize_text(raw_page.page_content)
            if not content:
                continue

            page_number = int(raw_page.metadata.get("page", 0)) + 1
            page_count += 1
            pages.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_key": source_key,
                        "source_title": source.title,
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page": page_number,
                        "citation": f"{source.title}, p. {page_number}",
                    },
                )
            )

        document_stats.append(
            {
                "title": source.title,
                "file_name": source.file_name,
                "path": str(pdf_path),
                "page_count": page_count,
            }
        )

    return pages, document_stats


def build_chunks(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    split_docs = splitter.split_documents(pages)
    chunks: list[Document] = []

    for index, chunk in enumerate(split_docs, start=1):
        metadata = dict(chunk.metadata)
        first_line = next((line.strip() for line in chunk.page_content.splitlines() if line.strip()), "")
        metadata["chunk_id"] = f"{metadata['source_key']}::p{metadata['page']:03d}::c{index:05d}"
        metadata["section_hint"] = first_line[:160]
        metadata["char_count"] = len(chunk.page_content)
        chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

    return chunks


class TransitComplianceBot:
    def __init__(self) -> None:
        self.device, self.device_warning = resolve_runtime_device()
        self.pages, self.document_stats = load_source_pages()
        self.documents = build_chunks(self.pages)
        self.summary = self._build_summary()
        self.embeddings = self._build_embeddings()
        self.embedding_matrix = self._build_embedding_matrix()
        self.bm25 = self._build_bm25()
        self.qa = self._build_qa_pipeline()

    def _build_summary(self) -> dict[str, Any]:
        avg_chunk_chars = round(
            sum(len(document.page_content) for document in self.documents) / max(len(self.documents), 1),
            2,
        )
        return {
            "created_at_utc": None,
            "documents": self.document_stats,
            "stats": {
                "page_count": len(self.pages),
                "chunk_count": len(self.documents),
                "avg_chunk_chars": avg_chunk_chars,
            },
            "runtime": {
                "mode": "cloud-portable",
                "embedding_model_id": CONFIG.embedding_model_id,
                "qa_model_id": CONFIG.qa_model_id,
                "dense_top_k": CONFIG.dense_top_k,
                "bm25_top_k": CONFIG.bm25_top_k,
                "final_top_k": CONFIG.final_top_k,
            },
        }

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        batch_size = 32 if self.device == "cuda" else 8
        return HuggingFaceEmbeddings(
            model_name=CONFIG.embedding_model_id,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        )

    def _build_embedding_matrix(self) -> np.ndarray:
        vectors = np.asarray(
            self.embeddings.embed_documents([document.page_content for document in self.documents]),
            dtype=np.float32,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _build_bm25(self) -> BM25Retriever:
        retriever = BM25Retriever.from_documents(self.documents)
        retriever.k = CONFIG.bm25_top_k
        return retriever

    def _build_qa_pipeline(self):
        device_index = 0 if self.device == "cuda" else -1
        return pipeline(
            task="question-answering",
            model=CONFIG.qa_model_id,
            tokenizer=CONFIG.qa_model_id,
            device=device_index,
        )

    def _dense_search(self, question: str) -> list[tuple[Document, float]]:
        query_vector = np.asarray(self.embeddings.embed_query(question), dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        query_vector = query_vector / query_norm

        similarities = self.embedding_matrix @ query_vector
        top_indices = np.argsort(-similarities)[: CONFIG.dense_top_k]
        return [(self.documents[index], float(similarities[index])) for index in top_indices]

    def retrieve(self, question: str) -> list[Document]:
        fused: dict[str, dict[str, Any]] = {}

        for rank, (document, score) in enumerate(self._dense_search(question), start=1):
            chunk_id = document.metadata["chunk_id"]
            fused.setdefault(chunk_id, {"document": document, "score": 0.0})
            fused[chunk_id]["score"] += (1.15 / (rank + 2)) + (0.35 * max(score, 0.0))

        for rank, document in enumerate(self.bm25.invoke(question), start=1):
            chunk_id = document.metadata["chunk_id"]
            fused.setdefault(chunk_id, {"document": document, "score": 0.0})
            fused[chunk_id]["score"] += 1.0 / (rank + 2)

        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
        return [item["document"] for item in ranked[: CONFIG.final_top_k]]

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

    def _best_supporting_sentence(self, question: str, document: Document, answer: str = "") -> str:
        sentences = split_sentences(document.page_content)
        answer_lower = answer.lower().strip()
        ranked_sentences = []
        for sentence in sentences:
            score = overlap_score(question, sentence)
            if answer_lower and answer_lower in sentence.lower():
                score += 1.5
            ranked_sentences.append((score, sentence))

        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        return ranked_sentences[0][1].strip()

    def _extract_answer_candidates(self, question: str, documents: list[Document]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_sentences: set[str] = set()

        for document in documents:
            result = self.qa(
                question=question,
                context=document.page_content,
                handle_impossible_answer=True,
                max_answer_len=96,
            )
            answer_text = " ".join(str(result.get("answer", "")).split()).strip()
            score = float(result.get("score", 0.0))

            if not answer_text or answer_text.lower() in {"empty", "[cls]"} or score < CONFIG.min_answer_score:
                supporting_sentence = self._best_supporting_sentence(question, document)
                score = overlap_score(question, supporting_sentence)
            else:
                supporting_sentence = self._best_supporting_sentence(question, document, answer_text)
                score += overlap_score(question, supporting_sentence)

            normalized_sentence = supporting_sentence.lower()
            if not supporting_sentence or normalized_sentence in seen_sentences:
                continue

            seen_sentences.add(normalized_sentence)
            candidates.append(
                {
                    "sentence": supporting_sentence,
                    "citation": document.metadata.get("citation", "Unknown source"),
                    "score": score,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def generate_answer(self, question: str, documents: list[Document]) -> str:
        if not documents:
            return "I do not have enough information in the provided documents to answer that question."

        candidates = self._extract_answer_candidates(question, documents)
        if not candidates:
            return "I could not find a sufficiently grounded answer in the provided documents."

        selected = candidates[: CONFIG.max_answer_candidates]
        lines = [f"{selected[0]['sentence']} [{selected[0]['citation']}]"]

        if len(selected) > 1:
            lines.append("")
            lines.append("Additional relevant provisions:")
            for candidate in selected[1:]:
                lines.append(f"- {candidate['sentence']} [{candidate['citation']}]")

        return "\n".join(lines)

    def ask(self, question: str) -> dict[str, Any]:
        documents = self.retrieve(question)
        answer = self.generate_answer(question, documents)
        return {
            "question": question,
            "answer": answer,
            "sources": self._collect_sources(documents),
            "contexts": self._serialize_documents(documents),
        }


@st.cache_resource(show_spinner="Loading PDFs, building the in-memory index, and downloading lightweight models...")
def get_bot() -> TransitComplianceBot:
    return TransitComplianceBot()


def apply_theme() -> None:
    st.set_page_config(
        page_title="Transit Compliance Copilot",
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
                --shadow: 0 24px 60px rgba(13, 29, 26, 0.10);
                --focus: #0b5f59;
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

            .stButton > button {
                border-radius: 999px;
                border: 1px solid rgba(13, 29, 26, 0.14);
                background: var(--surface-strong);
                color: var(--ink);
                font-weight: 600;
                transition: transform 120ms ease, box-shadow 120ms ease;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 25px rgba(16, 35, 29, 0.08);
                border-color: rgba(15,118,110,0.28);
            }

            .stButton > button:focus,
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


def render_hero(summary: dict[str, Any], device_label: str) -> None:
    stats = summary.get("stats", {})
    documents = summary.get("documents", [])
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
                Ask operational, compliance, ADA, vehicle, or paratransit questions against the bundled CFR PDFs.
                The deployed app builds its retrieval index in memory at startup, then returns grounded answers with
                page citations and transparent evidence panels.
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
                    <div class="metric-label">Runtime</div>
                    <div class="metric-value">{escape(device_label)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Index mode</div>
                    <div class="metric-value">In memory</div>
                </div>
            </div>
            <div style="margin-top: 0.8rem;">{doc_names}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bot: TransitComplianceBot) -> None:
    runtime = bot.summary.get("runtime", {})

    with st.sidebar:
        st.markdown("## Control Room")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-kicker">System status</div>
                <div class="sidebar-title">Ready for cloud deployment</div>
                <div style="color:var(--muted); margin-top:0.35rem;">
                    Device: <strong>{escape(bot.device.upper())}</strong><br>
                    PDF-backed hybrid retrieval with a lightweight QA model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if bot.device_warning:
            st.info(bot.device_warning)

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-kicker">Model stack</div>
                <div class="sidebar-title">{escape(CONFIG.qa_model_id)}</div>
                <div style="color:var(--muted); margin-top:0.35rem;">
                    Embeddings: {escape(CONFIG.embedding_model_id)}<br>
                    Answering mode: extractive with citations
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-kicker">Retrieval profile</div>
                <div class="sidebar-title">Portable streamlit runtime</div>
                <div style="color:var(--muted); margin-top:0.35rem;">
                    Dense top-k: {runtime.get('dense_top_k', 'n/a')}<br>
                    BM25 top-k: {runtime.get('bm25_top_k', 'n/a')}<br>
                    Final evidence set: {runtime.get('final_top_k', 'n/a')}
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
        for document in bot.summary.get("documents", []):
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
                <h4>#{context.get("rank", "?")} | {citation}</h4>
                <div class="source-meta">{section_hint} | {source_file}</div>
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
    st.error("The app could not initialize its PDF-backed retrieval engine.")
    st.markdown(
        f"""
        <div class="panel-card">
            <strong>Setup required</strong><br><br>
            {escape(error_text)}<br><br>
            This deployed app expects the two source PDFs to be present in the repository root.
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
        status.write("Loading the portable transit compliance engine.")
        bot = get_bot()
        status.write("Searching the indexed regulations with dense and lexical retrieval.")
        documents = bot.retrieve(question)
        status.write(f"Selected {len(documents)} evidence chunks for answer construction.")
        status.write("Extracting the strongest grounded answer spans from the retrieved text.")
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
        bot = get_bot()
    except Exception as error:
        render_setup_blocker(str(error))
        return

    render_sidebar(bot)

    left, right = st.columns([3.2, 1.15], gap="large")

    with left:
        render_hero(bot.summary, bot.device.upper())
        render_quick_actions()

        if not st.session_state["messages"]:
            st.markdown(
                """
                <div class="panel-card" style="margin-top: 1rem;">
                    <strong>What this app does</strong><br><br>
                    It indexes the two bundled CFR PDFs at startup, retrieves the most relevant regulatory excerpts in
                    memory, and answers with explicit page citations plus a visible evidence trail.
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
                Responses are grounded in retrieved CFR text, then rendered with page citations and supporting excerpts.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel-card" style="margin-top: 1rem;">
                <strong>Deployment profile</strong><br><br>
                This version is optimized for Streamlit Community Cloud: no Chroma dependency at runtime, no prebuilt
                artifact requirement, and lightweight Hugging Face models that can initialize directly from the repo PDFs.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Chunks in memory", bot.summary["stats"]["chunk_count"])

        if st.session_state["messages"]:
            latest_sources: list[str] = []
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
