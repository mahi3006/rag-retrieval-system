"""
generation.py
-------------
Takes a user query + retrieved chunks and generates a grounded answer
using Mistral 7B via Ollama (runs locally on your GPU).

Key design decisions:
  1. Grounded prompting  — system prompt instructs the model to answer
                           ONLY from the provided context, never from
                           its own knowledge. This is what makes it RAG.

  2. Source attribution  — every answer ends with which documents were used,
                           making hallucinations traceable.

  3. Token budgeting     — we cap context to TOP_K_CHUNKS chunks so we
                           never accidentally overflow the context window.

  4. Separation of roles — this file only does generation. Retrieval,
                           chunking, and embedding live in their own files.
"""

import time
from typing import List, Dict, Any, Optional

import ollama

from retrieval import format_chunks_for_context, format_sources


# ── Configuration ──────────────────────────────────────────────────────────────

OLLAMA_MODEL   = "mistral"      # must be pulled: ollama pull mistral
TOP_K_CHUNKS   = 5              # max chunks to include in context
MAX_TOKENS     = 1024           # max tokens for the generated answer
TEMPERATURE    = 0.1            # low temperature = more factual, less creative


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise research assistant specialising in AI applications \
in FMCG supply chains.

Your job is to answer the user's question using ONLY the context passages provided below.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say:
   "The provided documents do not contain sufficient information to answer this question."
3. Be specific and cite which source (by number in brackets) supports each claim.
4. Keep your answer concise and structured — use bullet points where appropriate.
5. Never fabricate statistics, names, or facts not present in the context.
"""


# ── Build the Full Prompt ──────────────────────────────────────────────────────

def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs the full user-turn prompt by combining the query
    with the formatted retrieved context.

    Format:
        CONTEXT:
        [1] Source: file.pdf, Page 3
        <chunk text>

        [2] Source: ...

        QUESTION:
        <user query>

    Args:
        query  : The user's question string.
        chunks : List of retrieved chunk dicts from retrieval.py.

    Returns:
        A formatted prompt string for the LLM.
    """
    context_str = format_chunks_for_context(chunks)

    prompt = (
        f"CONTEXT:\n"
        f"{context_str}\n"
        f"QUESTION:\n"
        f"{query}"
    )
    return prompt


# ── Generate Answer ────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    model: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Generates a grounded answer from the query and retrieved chunks.

    Args:
        query      : The user's question string.
        chunks     : Retrieved chunk dicts from retrieval.py (top-k).
        model      : Ollama model name (default: "mistral").
        temperature: Sampling temperature. Lower = more deterministic.
        max_tokens : Maximum number of tokens in the response.

    Returns:
        {
            "query"         : original query,
            "answer"        : generated answer text,
            "sources"       : formatted source citation string,
            "num_chunks"    : number of chunks used as context,
            "latency_ms"    : generation time in milliseconds,
            "model"         : model name used,
            "prompt_preview": first 300 chars of the prompt (for debugging),
        }
    """
    # Cap chunks to TOP_K_CHUNKS
    chunks = chunks[:TOP_K_CHUNKS]

    prompt     = build_prompt(query, chunks)
    sources    = format_sources(chunks)

    start = time.time()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        answer = response["message"]["content"].strip()

    except Exception as e:
        answer = f"[Generation Error] {str(e)}"

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "query"         : query,
        "answer"        : answer,
        "sources"       : sources,
        "num_chunks"    : len(chunks),
        "latency_ms"    : latency_ms,
        "model"         : model,
        "prompt_preview": prompt[:300],
    }


# ── Full RAG Call (retrieve + generate) ───────────────────────────────────────

def rag_answer(
    query: str,
    collection,
    model_wrapper: Dict[str, Any],
    k: int = TOP_K_CHUNKS,
    llm_model: str = OLLAMA_MODEL,
) -> Dict[str, Any]:
    """
    End-to-end RAG: retrieves chunks then generates a grounded answer.

    This is the main function called by pipeline.py and the eval harness.

    Args:
        query        : The user's question.
        collection   : A loaded ChromaDB collection.
        model_wrapper: Embedding model wrapper matching the collection.
        k            : Number of chunks to retrieve.
        llm_model    : Ollama model name for generation.

    Returns:
        Combined dict with retrieval + generation results:
        {
            "query"              : ...,
            "answer"             : ...,
            "sources"            : ...,
            "chunks"             : [...],   # retrieved chunks
            "retrieval_latency_ms": ...,
            "generation_latency_ms": ...,
            "total_latency_ms"   : ...,
            "collection_name"    : ...,
            "embedding_model"    : ...,
            "llm_model"          : ...,
        }
    """
    from retrieval import retrieve   # local import to avoid circular

    # Step 1: Retrieve
    retrieval_result = retrieve(query, collection, model_wrapper, k=k)

    # Step 2: Generate
    generation_result = generate_answer(
        query=query,
        chunks=retrieval_result["chunks"],
        model=llm_model,
    )

    return {
        "query"                 : query,
        "answer"                : generation_result["answer"],
        "sources"               : generation_result["sources"],
        "chunks"                : retrieval_result["chunks"],
        "retrieval_latency_ms"  : retrieval_result["latency_ms"],
        "generation_latency_ms" : generation_result["latency_ms"],
        "total_latency_ms"      : retrieval_result["latency_ms"] + generation_result["latency_ms"],
        "collection_name"       : retrieval_result["collection_name"],
        "embedding_model"       : retrieval_result["model_name"],
        "llm_model"             : llm_model,
    }


# ── Pretty Print ───────────────────────────────────────────────────────────────

def print_rag_result(result: Dict[str, Any]) -> None:
    """Pretty-prints a full RAG result — useful for demos and debugging."""
    print(f"\n{'='*65}")
    print(f"QUERY: {result['query']}")
    print(f"{'='*65}")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\n{result['sources']}")
    print(f"\n[Collection: {result['collection_name']} | "
          f"Embed: {result['embedding_model']} | "
          f"LLM: {result['llm_model']}]")
    print(f"[Retrieval: {result['retrieval_latency_ms']}ms | "
          f"Generation: {result['generation_latency_ms']}ms | "
          f"Total: {result['total_latency_ms']}ms]")


# ── Entry Point (for testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from embedding import load_both_models
    from vectorstore import load_collection

    print("[Test] Loading models...")
    models = load_both_models(device="cuda")

    test_queries = [
        "What are the main applications of AI in FMCG supply chains?",
        "How does generative AI help with demand forecasting?",
        "What risks does AI introduce in supply chain management?",
    ]

    # Test with the best expected combination: recursive + BGE
    col           = load_collection("recursive_bge")
    model_wrapper = models["bge"]

    for query in test_queries:
        result = rag_answer(query, col, model_wrapper, k=5)
        print_rag_result(result)
        print()