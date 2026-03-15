"""
embeddings.py
-------------
Loads and wraps two embedding models for benchmarking:

  Model 1 — all-MiniLM-L6-v2  (sentence-transformers)
      General-purpose, fast, lightweight. Good baseline.

  Model 2 — BAAI/bge-small-en-v1.5
      Retrieval-optimised, stronger MTEB scores.
      Requires a special query prefix for best performance.

Both produce 384-dimensional vectors so comparisons are fair.

FIX NOTE: Pydantic v2 (used in newer LangChain) does not allow setting
arbitrary attributes on HuggingFaceEmbeddings objects. We therefore wrap
each model in a plain dict:
    {
        "embedder"     : <HuggingFaceEmbeddings>,
        "name"         : "minilm" | "bge",
        "query_prefix" : ""  |  BGE_QUERY_PREFIX,
    }
All downstream code (retrieval, vectorstore) uses this dict format.
"""

import time
from typing import List, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings


# ── Model Names ────────────────────────────────────────────────────────────────

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BGE_MODEL_NAME    = "BAAI/bge-small-en-v1.5"

# BGE works best when queries are prefixed with this string.
# Documents are NOT prefixed — only the query at retrieval time.
BGE_QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "


# ── Model Loaders ──────────────────────────────────────────────────────────────

def load_minilm(device: str = "cuda") -> Dict[str, Any]:
    """
    Loads the all-MiniLM-L6-v2 model and returns a model wrapper dict.

    Returns:
        {
            "embedder"     : HuggingFaceEmbeddings,
            "name"         : "minilm",
            "query_prefix" : "",          # no prefix needed for MiniLM
        }
    """
    print(f"[Embeddings] Loading MiniLM ({MINILM_MODEL_NAME}) on {device}...")
    start = time.time()

    embedder = HuggingFaceEmbeddings(
        model_name=MINILM_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        },
    )

    print(f"[Embeddings] MiniLM loaded in {time.time() - start:.1f}s")

    return {
        "embedder"     : embedder,
        "name"         : "minilm",
        "query_prefix" : "",
    }


def load_bge(device: str = "cuda") -> Dict[str, Any]:
    """
    Loads the BAAI/bge-small-en-v1.5 model and returns a model wrapper dict.

    BGE requires a query prefix at retrieval time — stored in the wrapper
    so the prefix stays tied to the model, not scattered in the pipeline.

    Returns:
        {
            "embedder"     : HuggingFaceEmbeddings,
            "name"         : "bge",
            "query_prefix" : BGE_QUERY_PREFIX,
        }
    """
    print(f"[Embeddings] Loading BGE ({BGE_MODEL_NAME}) on {device}...")
    start = time.time()

    embedder = HuggingFaceEmbeddings(
        model_name=BGE_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        },
    )

    print(f"[Embeddings] BGE loaded in {time.time() - start:.1f}s")

    return {
        "embedder"     : embedder,
        "name"         : "bge",
        "query_prefix" : BGE_QUERY_PREFIX,
    }


def load_both_models(device: str = "cuda") -> Dict[str, Dict[str, Any]]:
    """
    Loads both models and returns them keyed by name.

    Returns:
        {
            "minilm": { "embedder": ..., "name": "minilm", "query_prefix": "" },
            "bge"   : { "embedder": ..., "name": "bge",    "query_prefix": "..." },
        }
    """
    return {
        "minilm": load_minilm(device=device),
        "bge"   : load_bge(device=device),
    }


# ── Embedding Helpers ──────────────────────────────────────────────────────────

def embed_query(query: str, model_wrapper: Dict[str, Any]) -> List[float]:
    """
    Embeds a single query string using the given model wrapper.

    Automatically applies the BGE query prefix if present.
    MiniLM prefix is "" so it is a no-op for that model.

    Args:
        query        : The user's question string.
        model_wrapper: A model wrapper dict from load_minilm() or load_bge().

    Returns:
        A list of floats (the embedding vector).
    """
    prefix   = model_wrapper["query_prefix"]
    embedder = model_wrapper["embedder"]
    return embedder.embed_query(prefix + query)


def embed_documents(texts: List[str], model_wrapper: Dict[str, Any]) -> List[List[float]]:
    """
    Embeds a list of document/chunk texts.

    No prefix is applied to documents — only queries get prefixed (BGE spec).

    Args:
        texts        : List of chunk text strings.
        model_wrapper: A model wrapper dict.

    Returns:
        List of embedding vectors, one per text.
    """
    return model_wrapper["embedder"].embed_documents(texts)


# ── Speed Benchmark ────────────────────────────────────────────────────────────

def benchmark_embedding_speed(
    model_wrapper: Dict[str, Any],
    sample_texts: List[str],
) -> dict:
    """
    Times how long the model takes to embed a list of texts.
    Used in the latency analysis (Bonus 4).

    Returns:
        Dict with timing stats.
    """
    name = model_wrapper["name"]
    print(f"\n[Speed Test] {name} — {len(sample_texts)} texts...")

    start   = time.time()
    _       = embed_documents(sample_texts, model_wrapper)
    elapsed = time.time() - start

    stats = {
        "model"        : name,
        "num_texts"    : len(sample_texts),
        "total_seconds": round(elapsed, 3),
        "ms_per_text"  : round(1000 * elapsed / len(sample_texts), 2),
    }

    print(f"  Total time : {stats['total_seconds']}s")
    print(f"  Per text   : {stats['ms_per_text']}ms")
    return stats


# ── Entry Point (for testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from ingest import load_all_pdfs
    from chunking import recursive_chunking

    print("Loading documents and chunking for speed test...")
    docs         = load_all_pdfs()
    chunks       = recursive_chunking(docs)
    sample_texts = [c.page_content for c in chunks[:100]]

    models = load_both_models(device="cuda")

    # Speed benchmark
    stats = []
    for model_wrapper in models.values():
        s = benchmark_embedding_speed(model_wrapper, sample_texts)
        stats.append(s)

    print("\n[Embedding Speed Comparison]")
    print(f"  {'Model':<10} {'ms/text':>10}")
    print(f"  {'-'*25}")
    for s in stats:
        print(f"  {s['model']:<10} {s['ms_per_text']:>10}ms")

    # Sanity check: embed one query and show vector shape
    print("\n[Sanity Check — Query Embedding]")
    test_query = "What are the applications of AI in FMCG supply chains?"
    for model_wrapper in models.values():
        vec = embed_query(test_query, model_wrapper)
        print(f"  {model_wrapper['name']}: dim={len(vec)}, first 5={[round(v,4) for v in vec[:5]]}")