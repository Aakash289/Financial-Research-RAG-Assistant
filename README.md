# Financial Research RAG Assistant

A hybrid, multimodal Retrieval-Augmented Generation (RAG) system that answers natural language questions about company annual reports, grounded in real source text and charts, with live quality scoring and full cost/latency observability.

---

## What This Project Is

You can ask questions like:

- *"What was Nvidia's data center revenue?"*
- *"What risks did Target mention related to supply chain?"*
- *"How did Coca-Cola's leadership discuss inflation and economic conditions?"*

...and get an answer grounded in the actual annual report text (and charts), with sources shown, not a generic LLM guess from memorized training data.

**Companies covered:** Microsoft, Nvidia, Coca-Cola, Target
**Coverage:** MSFT / NVDA / TGT — FY2023 & FY2024 · KO — FY2024 & FY2025 (an intentional exception, kept and flagged rather than force-aligned)

### What makes this more than a basic RAG demo

- **Hybrid retrieval** — combines term-based (BM25) and semantic (embedding) search, fused via Reciprocal Rank Fusion, rather than relying on one method alone
- **Multimodal** — retrieves and uses actual chart/table images as real vision input to the LLM, not just text
- **Reranking** — a Cohere cross-encoder narrows the fused candidate pool down to the most genuinely relevant chunks
- **Live quality scoring** — every answer is scored for faithfulness and relevancy in real time, using an LLM judge, no ground truth required
- **Full observability** — token usage, cost, and latency tracked automatically per query

---

## Architecture

**Layer 1, Data Ingestion**
PDF annual reports to PyMuPDF extraction to classified images to section-aware chunks to two parallel indexes (BM25, Chroma vectors)

**Layer 2, Orchestration**
LangGraph StateGraph coordinating hybrid retrieval, rerank, and generation as distinct nodes with shared state.

**Layer 3, Retrieval**
Hybrid only: BM25 + semantic, fused via Reciprocal Rank Fusion.

**Layer 4, Reranking and Generation**
Cohere Rerank scores the fused candidate pool on raw text. Top-k chunks pass to Claude Sonnet for grounded answer generation, using prompt caching and minimized context metadata.

**Layer 5, Observability**
LangSmith traces every LLM call automatically and tracks input/output tokens and cost per step. RAGAS scores faithfulness and answer relevancy live, per query, inside the pipeline itself (context precision and context recall were dropped, both require ground truth and can't be computed on a live, arbitrary user question).

**Layer 6, Presentation**
Gradio UI with a sidebar-style column for dataset guidance, query input with live pipeline stage indicators, and accordion-based result sections (answer, sources, pipeline trace, tokens, system prompt).

---

## Tools & Libraries

| Category | Tool | Why |
|---|---|---|
| PDF processing | PyMuPDF (`pymupdf`) | Text and raster image extraction |
| PDF diagnostics/rasterization | Poppler (`pdfinfo`, `pdffonts`, `pdfimages`, `pdftoppm`) | Content inventory, and converting pages to images for the vector-graphics fallback |
| Image handling | Pillow (PIL) | Resizing/re-encoding images before sending to any vision API |
| LLM (generation) | Claude Sonnet | The one task needing real reasoning |
| LLM (classification/judging) | Claude Haiku | Image classification, figure extraction, RAGAS judging — narrow tasks matched to a cheaper model |
| Embeddings | Voyage AI (`voyage-multimodal-3`) | Text and images share one vector space |
| Term-based retrieval | `rank_bm25` | Exact-match search (tickers, figures, fiscal labels) |
| Vector store | ChromaDB | Persisted semantic index with metadata filtering |
| Reranking | Cohere (`rerank-v3.5`) | Cross-encoder relevance scoring on the fused candidate pool |
| Orchestration | LangChain / LangGraph | `ChatAnthropic` wrapper (for LangSmith tracing) + graph-based pipeline (`StateGraph`) |
| Observability | LangSmith | Auto-traced tokens, cost, latency per LLM call |
| Evaluation | RAGAS | Live faithfulness + answer relevancy scoring |
| UI | Gradio | Web interface, live pipeline status, Analytics tab |
| Utilities | `python-dotenv`, `pandas` | Env var loading; log file aggregation |
| Dependency management | `uv` | Lockfile-based reproducible installs |

---

## Project Structure

```
financial-rag-assistant/
├── .env                              # API keys (not committed)
├── requirements.txt
├── data/
│   ├── raw_filings/                  # source annual report PDFs
│   ├── extracted_text/               # parsed text per document
│   ├── extracted_images/             # kept charts/tables/photos
│   ├── chroma_db/                    # persisted vector store (generated)
│   ├── chunks.json                   # chunked text + metadata (generated)
│   └── live_query_log.jsonl          # per-query log, shared by notebook + app (generated)
├── data_ingestion.ipynb              # build-time: PDFs → indexes
├── retrieval_pipeline.ipynb          # query-time pipeline, test queries, live scoring
├── app.py                            # Gradio UI (self-contained copy of the pipeline)
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python environment managed via [`uv`](https://docs.astral.sh/uv/)
- **Poppler** installed and on your system PATH (provides `pdfinfo`, `pdffonts`, `pdfimages`, `pdftoppm`) — required for ingestion's PDF diagnostics and vector-graphics fallback
- API keys for: Anthropic, Voyage AI, Cohere, and (optionally) LangSmith

### 2. Install dependencies

```cmd
uv add -r requirements.txt
```

> **Known issue:** `ragas` has a bug importing a module that was removed from newer `langchain-community` releases. If you hit `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`, run:
> ```cmd
> uv add "langchain-community<0.4.2"
> ```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=financial-rag-assistant
```

### 4. Add source documents

Place annual report PDFs in `data/raw_filings/`, named exactly:

```
EXCHANGE_TICKER_YEAR.pdf
```

e.g. `NASDAQ_MSFT_2024.pdf`, `NYSE_TGT_2023.pdf`

### 5. Run ingestion

Open `data_ingestion.ipynb` and run all cells, in order, top to bottom.

- Builds `data/chunks.json` and `data/chroma_db/`
- **Takes roughly 90 minutes**, dominated by Voyage's free-tier rate limit (3 requests/minute) during the text embedding step — a deliberate time-for-zero-cost tradeoff
- Includes built-in debug/verification cells after the steps most likely to need a sanity check

### 6. Run the app

```cmd
python app.py
```

Opens a local Gradio server (default `http://127.0.0.1:7860`). Ask a question, or click one of the sample questions in the sidebar.

Alternatively, run `retrieval_pipeline.ipynb` directly to test the pipeline in a notebook, including a small batch of test queries and live log inspection.

---

## A Note on the Engineering Process

This project's value comes as much from the bugs found and fixed by testing against real documents as from the initial design. A few examples:

- A filename collision in the PDF rasterization step was silently feeding vision the wrong page's image — caught by directly debugging a specific test case and comparing the returned file path against what was expected
- The original section-heading detector matched a fixed list of guessed phrases, which matched none of the real documents' actual headings — replaced with a shape-based detector (short line, Title Case or ALL CAPS, no trailing punctuation) that generalizes across any company's vocabulary
- RAGAS silently defaulted to OpenAI embeddings (which this project has no key for) unless explicitly told otherwise — fixed with a custom Voyage-based embeddings wrapper
- ChromaDB's undocumented-in-practice max batch size (5,461 items) broke a naive single `.add()` call for 12,685 chunks — fixed by batching the insert
- A cost investigation revealed RAGAS's faithfulness and relevancy metrics each make several LLM sub-calls internally, not one — the judge model was switched from Sonnet to Haiku after this was discovered, meaningfully cutting per-query cost
