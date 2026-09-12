"""
Financial Research RAG Assistant — Gradio UI

Self-contained copy of the pipeline logic from retrieval_pipeline.ipynb (notebooks
aren't directly importable as Python modules, so this file re-implements the same
setup/retrieval/rerank/generate/scoring functions rather than importing them).

Architecture: Hybrid retrieval (BM25 + semantic, RRF-fused) -> Cohere rerank ->
Claude Sonnet generate -> live RAGAS scoring (faithfulness + answer relevancy).
"""

# --- Standard library ---
import os
import re
import json
import time
import asyncio
import base64
import io
from pathlib import Path
from typing import TypedDict

# --- Third-party libraries ---
from dotenv import load_dotenv       # loads .env into environment variables
import gradio as gr                   # the UI framework this app is built with
import chromadb                       # ChromaDB client, reads the persisted store built during ingestion
from rank_bm25 import BM25Okapi       # BM25 term-based retrieval, rebuilt here from chunks.json
import voyageai                       # Voyage client, used to embed the live user query
import cohere                         # Cohere client, used for reranking the fused candidate pool
import pandas as pd                   # used only for reading/aggregating the live query log in the Analytics tab
from PIL import Image                 # used to resize retrieved images before sending them to Claude vision

from langchain_anthropic import ChatAnthropic     # Claude wrapper, generation call
from langgraph.graph import StateGraph, END        # orchestration: retrieve -> rerank -> generate
from langsmith import traceable                    # marks a function as its own timed step in LangSmith's trace tree

from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from langchain_core.embeddings import Embeddings


# ============================================================
# SETUP (same as retrieval_pipeline.ipynb Step 1)
# ============================================================

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
cohere_client = cohere.Client(api_key=COHERE_API_KEY)

# Claude Sonnet via LangChain so LangSmith's auto-instrumentation captures this
# call automatically (tokens, cost, latency), no manual logging needed.
llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=ANTHROPIC_API_KEY, temperature=0)

# Sonnet pricing used for the estimated-cost KPI cards. Update this if pricing changes.
SONNET_INPUT_COST_PER_MILLION = 3.0
SONNET_OUTPUT_COST_PER_MILLION = 15.0

chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
text_collection = chroma_client.get_collection(name="text_chunks")
image_collection = chroma_client.get_collection(name="image_chunks")  # queried by image_search() during retrieval

with open("data/chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)


def simple_tokenize(text):
    """Same tokenizer used during ingestion -- must match exactly, since BM25
    scores depend on consistent tokenization between indexing and querying."""
    return re.findall(r"[a-z0-9]+", text.lower())


tokenized_corpus = [simple_tokenize(chunk["text"]) for chunk in all_chunks]
bm25_index = BM25Okapi(tokenized_corpus)
chunks_by_id = {c["chunk_id"]: c for c in all_chunks}

print(f"Loaded {len(all_chunks)} chunks, {text_collection.count()} text vectors, "
      f"{image_collection.count()} image vectors. BM25 rebuilt.")


# ============================================================
# HYBRID RETRIEVAL (same as retrieval_pipeline.ipynb Step 2)
# ============================================================

RRF_K = 60
TOP_K_PER_SOURCE = 15


def bm25_search(query, top_k=TOP_K_PER_SOURCE):
    query_tokens = simple_tokenize(query)
    scores = bm25_index.get_scores(query_tokens)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [all_chunks[i]["chunk_id"] for i in top_indices]


def semantic_search(query, top_k=TOP_K_PER_SOURCE):
    query_embedding = voyage_client.multimodal_embed(
        inputs=[[query]], model="voyage-multimodal-3", input_type="query"
    ).embeddings[0]
    results = text_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["ids"][0], query_embedding  # return the embedding too, so image_search can reuse it


IMAGE_TOP_K = 3
IMAGE_DISTANCE_CUTOFF = 1.2  # Chroma cosine distance; higher = less similar


def image_search(query_embedding, top_k=IMAGE_TOP_K):
    """Finds relevant charts/tables/photos using the SAME query embedding already
    computed for text semantic search (voyage-multimodal-3 shares one vector
    space for text and images, so no second embedding call is needed)."""
    if image_collection.count() == 0:
        return []

    results = image_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    images = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        if distance > IMAGE_DISTANCE_CUTOFF:
            continue
        metadata = results["metadatas"][0][i]
        images.append({
            "image_id": results["ids"][0][i],
            "ticker": metadata["ticker"],
            "year": metadata["year"],
            "page_num": metadata["page_num"],
            "label": metadata["label"],
            "image_path": metadata["image_path"],
            "distance": distance,
        })
    return images


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    fused_scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


@traceable(name="hybrid_retrieval")
def hybrid_retrieve(query):
    bm25_ids = bm25_search(query)
    semantic_ids, query_embedding = semantic_search(query)
    images = image_search(query_embedding)

    fused = reciprocal_rank_fusion([bm25_ids, semantic_ids])

    candidates = []
    for chunk_id, rrf_score in fused:
        if chunk_id not in chunks_by_id:
            continue
        chunk = chunks_by_id[chunk_id]
        candidates.append({
            **chunk,
            "rrf_score": rrf_score,
            "from_bm25": chunk_id in bm25_ids,
            "from_semantic": chunk_id in semantic_ids,
        })
    return candidates, images


# ============================================================
# RERANK (same as retrieval_pipeline.ipynb Step 3)
# ============================================================

RERANK_TOP_N = 5


@traceable(name="rerank")
def rerank_candidates(query, candidates, top_n=RERANK_TOP_N):
    if not candidates:
        return []
    documents = [c["text"] for c in candidates]
    response = cohere_client.rerank(
        model="rerank-v3.5", query=query, documents=documents, top_n=min(top_n, len(documents)),
    )
    reranked = []
    for result in response.results:
        chunk = candidates[result.index]
        reranked.append({**chunk, "rerank_score": result.relevance_score})
    return reranked


# ============================================================
# LIVE RAGAS SCORING SETUP (same as retrieval_pipeline.ipynb Step 4)
# ============================================================

ragas_llm = LangchainLLMWrapper(llm)


class VoyageEmbeddingsForRagas(Embeddings):
    """Minimal LangChain-Embeddings-compatible adapter over Voyage, so RAGAS's
    answer_relevancy metric uses the same embedding provider as the rest of this
    project instead of defaulting to OpenAI."""

    def embed_documents(self, texts):
        result = voyage_client.multimodal_embed(
            inputs=[[t] for t in texts], model="voyage-multimodal-3", input_type="document"
        )
        return result.embeddings

    def embed_query(self, text):
        result = voyage_client.multimodal_embed(
            inputs=[[text]], model="voyage-multimodal-3", input_type="query"
        )
        return result.embeddings[0]


ragas_embeddings = LangchainEmbeddingsWrapper(VoyageEmbeddingsForRagas())
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings


def score_answer_live(query, answer, reranked_chunks):
    """Scores ONE live query's answer with faithfulness and answer relevancy,
    no ground truth needed, genuinely changes based on what was asked/retrieved."""
    sample = SingleTurnSample(
        user_input=query, response=answer, retrieved_contexts=[c["text"] for c in reranked_chunks],
    )

    async def _score():
        # BUG FIX: these two metrics are completely independent of each other,
        # but the original code awaited them one after the other, meaning
        # answer_relevancy didn't even START until faithfulness fully finished,
        # despite both being real latency contributors (faithfulness alone
        # decomposes the answer into claims and verifies each one, several
        # sub-calls). asyncio.gather runs them concurrently instead, so total
        # scoring time is roughly the SLOWER of the two, not the SUM of both.
        f_score, r_score = await asyncio.gather(
            faithfulness.single_turn_ascore(sample),
            answer_relevancy.single_turn_ascore(sample),
        )
        return f_score, r_score

    faithfulness_score, relevancy_score = asyncio.run(_score())
    return {"faithfulness": faithfulness_score, "answer_relevancy": relevancy_score}


# ============================================================
# GENERATE (same as retrieval_pipeline.ipynb Step 4)
# ============================================================

SYSTEM_PROMPT = """You are a financial research assistant. Answer the user's question \
using ONLY the provided context chunks from company annual reports. Every claim in your \
answer must be grounded in the context. If the context does not contain enough information \
to answer, say so plainly rather than guessing. When you use a fact from a chunk, mention \
which company and fiscal year it came from."""


def build_context_block(reranked_chunks):
    parts = []
    for c in reranked_chunks:
        parts.append(f"[{c['ticker']} FY{c['year']}, {c['section']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def prepare_image_for_generation(image_path, max_dimension=1568, jpeg_quality=85):
    """Resizes/re-encodes a retrieved image as base64 JPEG before sending it to
    Claude -- images can be well over the 10 MB API limit at full extracted
    resolution, and resolution beyond ~1568px adds cost without adding accuracy."""
    with open(image_path, "rb") as f:
        img = Image.open(f)
        img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > max_dimension:
        scale = max_dimension / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def build_image_content_blocks(images):
    """Turns retrieved image metadata into actual image content blocks Claude can
    see, so retrieved charts genuinely inform the answer instead of being unused
    metadata. Claude Sonnet is vision-capable."""
    blocks = []
    for img in images:
        image_b64 = prepare_image_for_generation(img["image_path"])
        blocks.append({
            "type": "text",
            "text": f"[Chart/table from {img['ticker']} FY{img['year']}, page {img['page_num']}]",
        })
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
        })
    return blocks


@traceable(name="generate")
def generate_answer(query, reranked_chunks, images=None):
    images = images or []
    context_block = build_context_block(reranked_chunks)
    system_content = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    user_content = [{"type": "text", "text": f"Context:\n\n{context_block}\n\nQuestion: {query}"}]
    user_content.extend(build_image_content_blocks(images))

    response = llm.invoke([
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ])

    usage = response.response_metadata.get("usage", {})
    answer_text = response.content
    live_scores = score_answer_live(query, answer_text, reranked_chunks)

    return {
        "answer": answer_text,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "faithfulness": live_scores["faithfulness"],
        "answer_relevancy": live_scores["answer_relevancy"],
    }


# ============================================================
# LANGGRAPH WIRING (same as retrieval_pipeline.ipynb Step 5)
# ============================================================

class PipelineState(TypedDict):
    query: str
    candidates: list
    reranked: list
    images: list                # relevant charts/tables found by image_search
    answer: str
    input_tokens: int
    output_tokens: int
    faithfulness: float
    answer_relevancy: float


def retrieve_node(state: PipelineState) -> PipelineState:
    candidates, images = hybrid_retrieve(state["query"])
    return {**state, "candidates": candidates, "images": images}


def rerank_node(state: PipelineState) -> PipelineState:
    return {**state, "reranked": rerank_candidates(state["query"], state["candidates"])}


def generate_node(state: PipelineState) -> PipelineState:
    # images pass straight through from retrieve_node -- not reranked, Cohere
    # reranks text, not images, so this is a separate channel through the pipeline
    result = generate_answer(state["query"], state["reranked"], state.get("images", []))
    return {
        **state,
        "answer": result["answer"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
    }


graph_builder = StateGraph(PipelineState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("generate", generate_node)
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "generate")
graph_builder.add_edge("generate", END)
rag_graph = graph_builder.compile()


# ============================================================
# UI HELPERS
# ============================================================

def kpi_card_html(label, value, accent=False):
    """Builds one small KPI card as raw HTML/CSS. Gradio has no built-in
    st.metric-style card widget, so this is a custom stand-in, used consistently
    everywhere a numeric result is shown (per-query and in the Analytics tab).
    Matches the app's light/blue palette: white card, blue-tinted border,
    blue value text, with a slightly stronger blue tint for "accent" cards
    (used for the quality/RAGAS metrics, to visually separate them from the
    purely operational token/cost/latency numbers)."""
    if accent:
        bg, border, value_color = "#eff6ff", "#93c5fd", "#1d4ed8"
    else:
        bg, border, value_color = "#ffffff", "#dbeafe", "#0f172a"
    return f"""
    <div style="background:{bg}; border-radius:10px; padding:12px 16px; border: 1px solid {border};
                min-width:150px; display:inline-block; margin:4px; box-shadow: 0 1px 2px rgba(15,23,42,0.04);">
        <div style="color:#64748b; font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
        <div style="color:{value_color}; font-size:22px; font-weight:700; margin-top:4px;">{value}</div>
    </div>
    """


def source_line(chunk):
    tags = ("BM25" if chunk["from_bm25"] else "") + (" + Semantic" if chunk["from_semantic"] else "")
    return (f"**[{chunk['ticker']} FY{chunk['year']} / {chunk['section']}]** "
            f"({tags.strip(' +')}, rerank score {chunk['rerank_score']:.2f})\n\n"
            f"> {chunk['text'][:280]}{'...' if len(chunk['text']) > 280 else ''}")


def log_query(query, result, elapsed):
    """Appends this query's results to the running log, same file the offline
    notebook and this app both read from for rolling Analytics averages."""
    log_entry = {
        "query": query,
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("data/live_query_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def estimate_cost(input_tokens, output_tokens):
    return (input_tokens / 1_000_000) * SONNET_INPUT_COST_PER_MILLION + \
           (output_tokens / 1_000_000) * SONNET_OUTPUT_COST_PER_MILLION


# ============================================================
# MAIN QUERY HANDLER (generator, so the UI can show live pipeline stages)
# ============================================================

def run_query_ui(query):
    """A generator function: each `yield` pushes a partial update to the UI's
    output components immediately, rather than waiting for the whole pipeline
    to finish. This is what drives the 'live pipeline stage indicator' --
    LangGraph's .stream() (rather than .invoke()) yields the state right after
    each node finishes, which we use as the hook points for status updates.
    """
    if not query or not query.strip():
        yield ("", "", "", "", "", [], "Please enter a question.")
        return

    # Stage messages go to status_display (the last output slot), the answer
    # box itself stays empty until the real answer is ready, rather than
    # temporarily showing status text where the answer belongs.
    yield ("", "", "", "", "", [], "Retrieving...")

    start = time.time()
    final_state = None

    # .stream() yields {node_name: node_output_state} after each node completes,
    # this is what lets the status indicator update mid-pipeline instead of
    # only at the very end.
    for step_output in rag_graph.stream({"query": query}):
        node_name = list(step_output.keys())[0]
        node_state = step_output[node_name]

        if node_name == "retrieve":
            yield ("", "", "", "", "", [], "Reranking...")
        elif node_name == "rerank":
            yield ("", "", "", "", "", [], "Generating and scoring...")
        elif node_name == "generate":
            final_state = node_state

    elapsed = time.time() - start

    if final_state is None:
        yield ("", "", "", "", "", [], "Something went wrong, no answer was generated.")
        return

    log_query(query, final_state, elapsed)

    # --- Answer + live score badges ---
    answer_md = (
        f"{final_state['answer']}\n\n"
        f"---\n"
        f"**Faithfulness:** {final_state['faithfulness']:.2f} / 1.00　"
        f"**Answer Relevancy:** {final_state['answer_relevancy']:.2f} / 1.00"
    )

    # --- Sources ---
    sources_md = "\n\n".join(source_line(c) for c in final_state.get("reranked", [])) or "No sources retrieved."

    # --- Retrieved images, for the gallery ---
    # gr.Gallery accepts a list of (image_path, caption) tuples.
    images = final_state.get("images", [])
    gallery_items = [
        (img["image_path"], f"{img['ticker']} FY{img['year']} p{img['page_num']} ({img['label']})")
        for img in images
    ]

    # --- Pipeline trace ---
    reranked = final_state.get("reranked", [])
    bm25_count = sum(1 for c in reranked if c["from_bm25"])
    semantic_count = sum(1 for c in reranked if c["from_semantic"])
    trace_md = (
        f"- Hybrid retrieval ran (BM25 + semantic, always both, no router)\n"
        f"- {len(reranked)} chunks survived reranking\n"
        f"- {bm25_count} from BM25, {semantic_count} from semantic search (overlap possible)\n"
        f"- {len(images)} relevant image(s) found and sent to Claude as visual context\n"
        f"- End-to-end latency: {elapsed:.2f}s"
    )

    # --- Token/cost/latency KPI cards ---
    cost = estimate_cost(final_state["input_tokens"], final_state["output_tokens"])
    kpi_html = (
        kpi_card_html("Input Tokens", final_state["input_tokens"])
        + kpi_card_html("Output Tokens", final_state["output_tokens"])
        + kpi_card_html("Est. Cost", f"${cost:.4f}")
        + kpi_card_html("Latency", f"{elapsed:.2f}s")
    )

    # --- System prompt (transparency panel) ---
    system_prompt_display = SYSTEM_PROMPT

    yield (answer_md, sources_md, trace_md, kpi_html, system_prompt_display, gallery_items, "")


# ============================================================
# ANALYTICS TAB
# ============================================================

def load_analytics():
    """Reads the running query log and computes rolling averages/session totals.
    Genuinely changes as more queries are asked, not a fixed one-time snapshot."""
    log_path = Path("data/live_query_log.jsonl")
    if not log_path.exists() or log_path.stat().st_size == 0:
        empty = kpi_card_html("Total Queries", "0")
        return empty, "No queries logged yet. Ask something in the Query tab first."

    df = pd.read_json(log_path, lines=True)
    total_queries = len(df)
    total_tokens = int(df["input_tokens"].sum() + df["output_tokens"].sum())
    avg_cost = df.apply(lambda r: estimate_cost(r["input_tokens"], r["output_tokens"]), axis=1).mean()
    avg_latency = df["latency_seconds"].mean()
    avg_faithfulness = df["faithfulness"].mean()
    avg_relevancy = df["answer_relevancy"].mean()

    session_html = (
        kpi_card_html("Total Queries", total_queries)
        + kpi_card_html("Total Tokens", f"{total_tokens:,}")
        + kpi_card_html("Avg Cost / Query", f"${avg_cost:.4f}")
        + kpi_card_html("Avg Latency", f"{avg_latency:.2f}s")
    )
    quality_html = (
        kpi_card_html("Avg Faithfulness", f"{avg_faithfulness:.2f} / 1.00", accent=True)
        + kpi_card_html("Avg Answer Relevancy", f"{avg_relevancy:.2f} / 1.00", accent=True)
    )
    return session_html + quality_html, f"Based on {total_queries} live queries so far."


# ============================================================
# GRADIO UI
# ============================================================

CUSTOM_CSS = """
/* ---- Design tokens: light page, blue accent, used consistently everywhere ---- */
.gradio-container { background: #f8fafc !important; max-width: 1180px !important; margin: 0 auto !important; }

/* Header block: kill Gradio's default block padding/border/shadow so the
   title, byline, and subtitle sit close together instead of far apart */
#header-title, #header-byline, #header-subtitle {
    border: none !important; box-shadow: none !important; background: transparent !important;
}
#header-title { color: #0f172a !important; text-align: center !important; margin: 4px 0 0 0 !important;
                 padding: 0 !important; }
#header-byline { text-align: center !important; color: #64748b !important; font-size: 13px !important;
                  margin: 2px 0 6px 0 !important; padding: 0 !important; }
#header-subtitle { text-align: center !important; margin: 0 0 18px 0 !important; padding: 0 !important; }
#header-subtitle p { color: #475569 !important; margin: 0 !important; }

/* About card: light blue tint, not dark, consistent with the rest of the page */
#about-card { background: #eff6ff; border-radius: 12px; padding: 22px 26px; margin: 0 0 22px 0;
              border: 1px solid #bfdbfe; }
#about-card h3 { margin-top: 0; margin-bottom: 8px; color: #1e40af !important; font-size: 16px; }
#about-card p, #about-card li { color: #334155 !important; line-height: 1.55; }
#about-card b, #about-card strong { color: #1e3a8a !important; }
#about-card ul { margin: 6px 0 0 0; padding-left: 20px; }

/* Topics card: same family as About, sits in the sidebar */
#topics-card { background: #ffffff; border-radius: 12px; padding: 16px 18px; margin: 14px 0 0 0;
                border: 1px solid #dbeafe; }
#topics-card h4 { margin: 0 0 10px 0; color: #1e40af !important; font-size: 13px; text-transform: uppercase;
                    letter-spacing: 0.04em; }
.topic-pill { display: inline-block; background: #dbeafe; color: #1e40af; font-size: 13px; font-weight: 500;
              padding: 5px 12px; border-radius: 999px; margin: 0 6px 8px 0; }

/* Main content cards: give the answer/sources area a defined card boundary
   instead of floating text directly on the page background */
#main-card { background: #ffffff; border-radius: 12px; padding: 22px 26px; border: 1px solid #e2e8f0;
             box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
"""

# Shown prominently at the top of the app, not buried in the sidebar, so a new
# user immediately understands what data exists and what kinds of questions
# are actually answerable before they type anything.
ABOUT_MD = """
<div id="about-card">
<h3>What's in this assistant</h3>
<p>Annual report data from four companies: <b>Microsoft, Nvidia, Coca-Cola, and Target</b>.
Coverage is FY2023–FY2024 for Microsoft, Nvidia, and Target, and FY2024–FY2025 for Coca-Cola.
Each report includes narrative text, financial tables, and charts.</p>

<h3>What you can ask</h3>
<ul>
<li>Specific financial figures — revenue, growth rates, segment performance</li>
<li>Risk factors and business risks disclosed in the filings</li>
<li>Strategic priorities and leadership commentary</li>
<li>Comparisons across companies or fiscal years</li>
</ul>
</div>
"""

# Topic pills, shown in the sidebar, so a user scanning the page (not reading
# the About card's prose) can still see at a glance what subject areas exist
# in the underlying documents before typing a question.
TOPICS = [
    "Revenue", "Profitability & Margins", "Risk Factors", "Growth & Strategy",
    "Segment Performance", "Supply Chain", "Leadership Commentary", "Financial Statements",
]
TOPICS_HTML = (
    "<div id='topics-card'><h4>Topics in this data</h4>"
    + "".join(f"<span class='topic-pill'>{t}</span>" for t in TOPICS)
    + "</div>"
)

SAMPLE_QUERIES = [
    "What was Nvidia's data center revenue?",
    "How did Microsoft describe its approach to AI infrastructure investment?",
    "What risks did Target mention related to supply chain?",
    "How did Coca-Cola's leadership discuss inflation and economic conditions?",
]

with gr.Blocks(title="Financial Research RAG Assistant") as demo:
    # theme and css are passed to launch() below, not the Blocks constructor --
    # Gradio 6.0 moved these parameters, passing them here works on older
    # Gradio versions but raises a deprecation warning on 6.0+.
    #
    # Light page background throughout (not dark): Gradio's built-in components
    # (buttons, example chips, labels) use light-theme text colors internally,
    # so a light page keeps their default, well-tested contrast intact. The
    # blue/light identity comes from the custom card elements below (About,
    # Topics, KPI cards), whose colors we fully control ourselves.

    gr.HTML("<h1 id='header-title'>Financial Research RAG Assistant</h1>")
    gr.HTML("<div id='header-byline'>Built by Aakash Bhanushali</div>")
    gr.Markdown("Hybrid retrieval (BM25 + semantic) over 4 companies' annual reports, "
                "reranked and answered by Claude, with live faithfulness/relevancy scoring.",
                elem_id="header-subtitle")
    gr.HTML(ABOUT_MD)

    with gr.Tabs():
        with gr.TabItem("Query"):
            # Created here with render=False so its logic exists before gr.Examples
            # references it below, but it visually renders later, inside the main
            # column, via the explicit query_input.render() call further down.
            # This is the correct Gradio pattern for "define once, place elsewhere"
            # rather than reassigning .inputs after the fact, which is fragile.
            query_input = gr.Textbox(
                label="Ask a question",
                placeholder="e.g. How did Nvidia's revenue change year over year?",
                lines=2,
                render=False,
            )

            with gr.Row(equal_height=False):
                # --- Sidebar column ---
                # Dataset info lives in the About card above; this column keeps
                # sample questions and the topic pills, both scannable at a glance.
                with gr.Column(scale=1, min_width=240):
                    gr.Examples(examples=SAMPLE_QUERIES, inputs=query_input, label="Try a sample question")
                    gr.HTML(TOPICS_HTML)

                # --- Main column ---
                with gr.Column(scale=3):
                    query_input.render()
                    submit_btn = gr.Button("Ask", variant="primary")
                    status_display = gr.Markdown("")

                    answer_output = gr.Markdown(label="Answer")

                    with gr.Accordion("Sources", open=True):
                        sources_output = gr.Markdown()
                        images_output = gr.Gallery(
                            label="Relevant charts/tables sent to Claude",
                            columns=3, height="auto", show_label=True,
                        )

                    with gr.Accordion("Pipeline Trace", open=False):
                        trace_output = gr.Markdown()

                    with gr.Accordion("Tokens, Cost, Latency", open=False):
                        kpi_output = gr.HTML()

                    with gr.Accordion("System Prompt", open=False):
                        system_prompt_output = gr.Markdown()

            submit_btn.click(
                fn=run_query_ui,
                inputs=[query_input],
                outputs=[answer_output, sources_output, trace_output, kpi_output,
                         system_prompt_output, images_output, status_display],
            )

        with gr.TabItem("Analytics"):
            gr.Markdown("### Session Metrics and Live RAGAS Scores")
            gr.Markdown("Rolling averages across every query actually asked, not a fixed offline snapshot.")
            refresh_btn = gr.Button("Refresh")
            analytics_html = gr.HTML()
            analytics_caption = gr.Markdown()

            refresh_btn.click(fn=load_analytics, outputs=[analytics_html, analytics_caption])
            demo.load(fn=load_analytics, outputs=[analytics_html, analytics_caption])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=CUSTOM_CSS)