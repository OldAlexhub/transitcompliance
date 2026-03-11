# Transit Compliance Copilot

`Transit Compliance Copilot` is a local, citation-first chatbot for transit professionals who need grounded answers from two Federal Transit Administration / CFR source documents:

- `49 CFR Part 37`
- `49 CFR Part 38`

The system is designed to answer operational and compliance questions about ADA accessibility, paratransit, vehicle requirements, and related transit regulations. It does not rely on internet search at runtime and is intentionally restricted to the indexed source documents.

## What This Project Does

This project builds a local retrieval-augmented generation (RAG) pipeline over the two CFR PDFs and serves it through a Streamlit chat app.

At a high level:

1. `code.ipynb` prepares the knowledge base.
2. The notebook extracts text from the PDFs, chunks it, embeds it, and saves the retrieval artifacts.
3. `app.py` loads those saved artifacts and runs a user-friendly Streamlit interface.
4. User questions are answered with retrieved excerpts and page-level citations.

Important technical point:

- This project does **not** fine-tune or "train" a language model on the documents in the classic ML sense.
- Instead, it precomputes retrieval artifacts so the app can answer faster and more reliably at runtime.

## Current Indexed Corpus

Based on the latest built manifest in `artifacts/manifest.json`:

- Documents indexed: `2`
- Total pages indexed: `179`
- Total chunks created: `932`
- Average chunk size: `795.64` characters

Source files currently configured:

- `49 CFR Part 37 (up to date as of 3-09-2026).pdf`
- `CFR-2024-title49-vol1-part38.pdf`

## Technical Architecture

The system uses a local RAG architecture with three model roles:

- Embedding model: `BAAI/bge-base-en-v1.5`
- Reranker model: `BAAI/bge-reranker-base`
- Generator model: `Qwen/Qwen2.5-1.5B-Instruct`

All selected models are:

- Hugging Face hosted
- free to use
- ungated
- runnable locally

### Retrieval Pipeline

The notebook and app implement the following flow:

1. PDF extraction with `PyMuPDF`
2. Text normalization and cleanup
3. Recursive chunking with overlap
4. Dense embeddings using `BAAI/bge-base-en-v1.5`
5. Persistent vector storage in `Chroma`
6. Lexical retrieval with `BM25`
7. Hybrid retrieval using dense + lexical signals
8. Cross-encoder reranking with `BAAI/bge-reranker-base`
9. Final grounded answer generation with `Qwen2.5-1.5B-Instruct`

### Citation Strategy

Each chunk carries page-level metadata derived from the source PDFs. Answers are expected to cite the retrieved material in this style:

- `[49 CFR Part 37, p. 12]`

The app also exposes the retrieved excerpts used for the answer so the user can inspect the evidence trail directly.

## Why the Notebook Exists

`code.ipynb` is the offline build step for the system. It is responsible for the expensive work that should not happen every time a user asks a question.

It does the following:

- downloads and caches the local model files
- extracts and chunks the PDFs
- builds the Chroma vector database
- saves the chunk metadata file
- writes a manifest describing the model paths, retrieval settings, and corpus stats

This reduces lag in the Streamlit app because the app loads prebuilt artifacts instead of rebuilding the index during startup or on each query.

## Artifact Layout

The notebook writes the following runtime assets:

```text
artifacts/
  chroma_db/        # persisted Chroma vector store
  chunks.jsonl      # chunk text + metadata
  manifest.json     # build metadata, paths, model config, retrieval config
  models/
    embeddings/
    reranker/
    generator/
```

These files are required by `app.py`.

## Main Files

- `code.ipynb`
  Offline build notebook for model download, PDF processing, chunking, embedding, and artifact generation.

- `app.py`
  Streamlit application for interactive question answering, retrieval display, source transparency, and chat UX.

- `requirements.txt`
  Python dependencies for LangChain, Chroma, Transformers, Streamlit, PyMuPDF, and related runtime libraries.

## Streamlit App Behavior

The app is designed as a local analyst interface rather than a generic chatbot. It provides:

- a chat-based UI for regulatory questions
- preloaded local models and saved retrieval artifacts
- evidence-backed answers
- visible retrieved source excerpts
- citation display for each response
- example prompts and session history
- setup checks if artifacts are missing

The app is intentionally scoped to the indexed documents. If the answer is not supported by the retrieved excerpts, the assistant should avoid inventing information.

## Dependencies

Main libraries used in this project:

- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `langchain-chroma`
- `chromadb`
- `transformers`
- `sentence-transformers`
- `PyMuPDF`
- `rank-bm25`
- `streamlit`

See `requirements.txt` for pinned version ranges.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want GPU acceleration, install a CUDA-enabled PyTorch build that matches your GPU and CUDA stack. This project can use CUDA when available, but it also includes fallback logic to use CPU if the installed PyTorch runtime cannot execute on the detected GPU architecture.

## Build the Retrieval Artifacts

Run the notebook:

```bash
jupyter notebook code.ipynb
```

Execute the cells to:

- download the local model snapshots
- process the PDFs
- build the vector index
- save `manifest.json`, `chunks.jsonl`, and the `chroma_db`

## Run the App

After the artifacts are built:

```bash
streamlit run app.py
```

## Retrieval Configuration

The current build configuration stored in the manifest is:

- Chunk size: `900`
- Chunk overlap: `180`
- Dense top-k: `8`
- Dense fetch-k: `24`
- BM25 top-k: `8`
- Reranker top-n: `5`
- Max context chunks: `5`
- Generation max new tokens: `400`
- Temperature: `0.0`

These values favor grounded, concise answers over open-ended generation.

## Limitations

- The assistant is only grounded in the two configured PDFs.
- It is not a substitute for legal advice.
- PDF extraction quality can affect retrieval quality.
- Citation accuracy depends on the extracted page metadata and the relevance of the retrieved chunks.
- GPU acceleration depends on having a compatible PyTorch build for the actual GPU architecture in the active environment.

## Recommended Use Cases

- ADA compliance questions for public transit operations
- paratransit eligibility and service requirement questions
- vehicle accessibility requirement lookups
- quick evidence-backed regulatory checks by planners, analysts, compliance staff, and operations teams

## Future Improvements

Possible technical extensions:

- add section-level citation extraction in addition to page-level citations
- support more CFR and FTA guidance documents
- add conversation export and answer audit logs
- add structured filters by document, topic, or page range
- add evaluation prompts and benchmark queries for retrieval quality

## Summary

This repository is a local, technical RAG system for transit regulation Q&A. The notebook builds the retrieval layer once, and the Streamlit app serves fast, cited answers from the saved artifacts. The main design goal is not generic chat, but reliable document-grounded assistance for transit professionals working with Part 37 and Part 38.
