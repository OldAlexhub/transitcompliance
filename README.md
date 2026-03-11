# Transit Compliance Copilot

`Transit Compliance Copilot` is a local, citation-first chatbot for transit professionals who need grounded answers from two Federal Transit Administration / CFR source documents:

- `49 CFR Part 37`
- `49 CFR Part 38`

The system is designed to answer operational and compliance questions about ADA accessibility, paratransit, vehicle requirements, and related transit regulations. It does not rely on internet search at runtime and is intentionally restricted to the indexed source documents.

## What This Project Does

This project builds a transit-regulation question answering system over the two CFR PDFs and serves it through a Streamlit chat app.

At a high level:

1. `code.ipynb` prepares an optional local artifact-based knowledge base.
2. The notebook extracts text from the PDFs, chunks it, embeds it, and saves retrieval artifacts for local use.
3. `app.py` provides a Streamlit interface that is deployable on Streamlit Community Cloud.
4. In deployed mode, the app reads the PDFs directly, builds an in-memory index at startup, and answers with page-level citations.

Important technical point:

- This project does **not** fine-tune or "train" a language model on the documents in the classic ML sense.
- Instead, it uses retrieval and lightweight Hugging Face models to answer from the source text.
- The notebook precomputes artifacts for a heavier local workflow, while the deployed Streamlit app uses a portable in-memory workflow.

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

The repository now supports two runtime profiles:

- `Cloud / Streamlit Community Cloud mode`
- `Local artifact-backed mode`

### Cloud / Streamlit Community Cloud Mode

This is the runtime implemented by `app.py` today.

It is designed to avoid:

- `artifacts/` dependencies
- `chromadb` runtime imports
- large locally committed model snapshots

Instead, the deployed app:

1. loads the two bundled PDFs directly from the repo
2. extracts and chunks them in memory at startup
3. builds dense embeddings with `sentence-transformers/all-MiniLM-L6-v2`
4. runs lexical retrieval with BM25
5. fuses dense and lexical retrieval results
6. extracts answer spans with `distilbert-base-cased-distilled-squad`
7. returns grounded answers with page citations and visible supporting excerpts

### Local Artifact-Backed Mode

The notebook still supports a heavier local workflow using a persisted vector index and local downloaded models.

That local workflow uses:

- Embedding model: `BAAI/bge-base-en-v1.5`
- Reranker model: `BAAI/bge-reranker-base`
- Generator model: `Qwen/Qwen2.5-1.5B-Instruct`

All selected models are:

- Hugging Face hosted
- free to use
- ungated
- runnable locally

### Local Retrieval Pipeline

The notebook implements the following local build flow:

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

This notebook is now mainly for the local artifact-backed workflow. It is still useful if you want a stronger local stack with persisted retrieval artifacts, but it is not required for Streamlit Community Cloud deployment.

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

These files are used by the local artifact-backed workflow. The deployed Streamlit Cloud app does not require them.

## Main Files

- `code.ipynb`
  Offline build notebook for model download, PDF processing, chunking, embedding, and artifact generation.

- `app.py`
  Streamlit application for interactive question answering, retrieval display, source transparency, and a cloud-compatible in-memory retrieval workflow.

- `requirements.txt`
  Python dependencies for LangChain, Chroma, Transformers, Streamlit, PyMuPDF, and related runtime libraries.

## Streamlit App Behavior

The app is designed as a transit analyst interface rather than a generic chatbot. It provides:

- a chat-based UI for regulatory questions
- startup indexing directly from the bundled PDFs
- evidence-backed answers
- visible retrieved source excerpts
- citation display for each response
- example prompts and session history
- setup checks if the source PDFs are missing

The app is intentionally scoped to the indexed documents. If the answer is not supported by the retrieved excerpts, the assistant should avoid inventing information.

## Dependencies

Main runtime libraries used in the deployed app:

- `langchain-community`
- `langchain-huggingface`
- `transformers`
- `sentence-transformers`
- `PyMuPDF`
- `rank-bm25`
- `streamlit`

See `requirements.txt` for pinned version ranges.

## Setup

Install Streamlit app dependencies:

```bash
pip install -r requirements.txt
```

For the local notebook workflow, install the larger dependency set:

```bash
pip install -r requirements-local.txt
```

If you want GPU acceleration locally, install a CUDA-enabled PyTorch build that matches your GPU and CUDA stack.

## Optional Local Artifact Build

Run the notebook only if you want the local artifact-backed workflow:

```bash
jupyter notebook code.ipynb
```

Execute the cells to:

- download the local model snapshots
- process the PDFs
- build the vector index
- save `manifest.json`, `chunks.jsonl`, and the `chroma_db`

## Run the App

For the deployed or portable app:

```bash
streamlit run app.py
```

## Retrieval Configuration

### Cloud runtime

The current deployed app uses:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- QA model: `distilbert-base-cased-distilled-squad`
- Chunk size: `950`
- Chunk overlap: `180`
- Dense top-k: `8`
- BM25 top-k: `8`
- Final evidence set: `5`

### Local artifact-backed runtime

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
- Streamlit Community Cloud has limited CPU and RAM, so the deployed app uses a lighter extractive QA pipeline instead of the heavier local generator stack.

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

This repository is a technical transit-regulation QA system with two operating modes: a deployable Streamlit Cloud runtime that builds an in-memory index directly from the PDFs, and an optional heavier local artifact-backed workflow built through the notebook. In both cases, the design goal is reliable document-grounded assistance for transit professionals working with Part 37 and Part 38.
