"""
bonuses/query_rewriting.py
--------------------------
Bonus 3: Query rewriting using a local LLM (Mistral via Ollama).

WHY QUERY REWRITING:
  User queries are often short, ambiguous, or use different vocabulary
  than the documents. Query rewriting expands or reformulates the query
  to improve retrieval coverage.

  Example:
    Original : "model drift supply chain"
    Rewritten: "How do organisations detect and handle machine learning
                model performance degradation and concept drift in
                deployed AI supply chain systems? What retraining
                strategies are used?"

TWO STRATEGIES IMPLEMENTED:

  1. HyDE (Hypothetical Document Embeddings):
     Ask the LLM to write a hypothetical answer to the question.
     Then embed THAT answer instead of the question.
     The idea: a hypothetical answer lives closer in embedding space
     to real relevant documents than the raw question does.

  2. Query Expansion:
     Ask the LLM to rewrite the query into 3 alternative phrasings.
     Retrieve chunks for all 3 phrasings and merge/deduplicate results.
     Covers more vocabulary and phrasing variants.

USAGE:
  from bonuses.query_rewriting import QueryRewriter
  rewriter = QueryRewriter()
  result   = rewriter.hyde_retrieve(query, collection, model_wrapper, k=5)
  result   = rewriter.expanded_retrieve(query, collection, model_wrapper, k=5)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Configuration ──────────────────────────────────────────────────────────────

OLLAMA_MODEL = "mistral"
TEMPERATURE  = 0.3     # slightly higher than generation — we want creative rewrites


# ── Query Rewriter ─────────────────────────────────────────────────────────────

class QueryRewriter:
    """
    Rewrites queries using a local LLM to improve retrieval coverage.
    Implements HyDE and Query Expansion strategies.
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        print(f"[QueryRewriter] Using {model} via Ollama")

    # ── LLM Call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        """Makes a single LLM call and returns the response text."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": TEMPERATURE, "num_predict": max_tokens},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[QueryRewriter] LLM error: {e}")
            return ""

    # ── Strategy 1: HyDE ──────────────────────────────────────────────────────

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generates a hypothetical passage that would answer the query.

        The LLM writes a short, factual-sounding paragraph as if it
        were a relevant excerpt from a research paper or industry report.
        This hypothetical document is then embedded instead of the query.

        Args:
            query: The user's original question.

        Returns:
            A hypothetical answer passage as a string.
        """
        prompt = (
            f"Write a short, factual paragraph (4-6 sentences) that would appear "
            f"in a research paper or industry report and directly answers this question:\n\n"
            f"Question: {query}\n\n"
            f"Write only the paragraph. No preamble, no 'Here is', no headings. "
            f"Be specific and technical. Use supply chain and AI terminology."
        )
        return self._call_llm(prompt, max_tokens=200)

    def hyde_retrieve(
        self,
        query: str,
        collection,
        model_wrapper: Dict[str, Any],
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        HyDE retrieval: generates a hypothetical document, embeds it,
        then uses that embedding for vector search.

        Args:
            query        : Original user query.
            collection   : ChromaDB collection.
            model_wrapper: Embedding model wrapper.
            k            : Number of results.

        Returns:
            Retrieval result dict with hypothetical_doc attached.
        """
        from vectorstore import similarity_search

        start = time.time()

        # Step 1: Generate hypothetical document
        hyp_doc = self.generate_hypothetical_document(query)
        llm_time = round((time.time() - start) * 1000, 2)

        if not hyp_doc:
            hyp_doc = query   # fallback to original query

        # Step 2: Embed the hypothetical document (not the query)
        prefix   = model_wrapper["query_prefix"]
        vec_start = time.time()
        results  = similarity_search(
            query=hyp_doc,          # embed the hypothetical doc
            collection=collection,
            model_wrapper=model_wrapper,
            k=k,
        )
        vec_time = round((time.time() - vec_start) * 1000, 2)

        total_latency = round((time.time() - start) * 1000, 2)

        return {
            "query"              : query,
            "hypothetical_doc"   : hyp_doc,
            "chunks"             : results,
            "latency_ms"         : total_latency,
            "llm_latency_ms"     : llm_time,
            "vector_latency_ms"  : vec_time,
            "collection_name"    : getattr(collection, "name", "unknown") + "_hyde",
            "model_name"         : model_wrapper["name"],
            "k"                  : k,
            "rewrite_strategy"   : "hyde",
        }

    # ── Strategy 2: Query Expansion ───────────────────────────────────────────

    def generate_query_variants(self, query: str, n: int = 3) -> List[str]:
        """
        Generates n alternative phrasings of the query.

        Args:
            query: Original query string.
            n    : Number of variants to generate.

        Returns:
            List of query variant strings (may include original query).
        """
        prompt = (
            f"Rewrite the following question in {n} different ways that preserve "
            f"the meaning but use different vocabulary and phrasing. "
            f"Focus on supply chain and AI terminology.\n\n"
            f"Original question: {query}\n\n"
            f"Output exactly {n} rewritten questions, one per line, numbered 1. 2. 3.\n"
            f"Do not include any other text."
        )

        response = self._call_llm(prompt, max_tokens=200)

        # Parse numbered list from response
        variants = []
        for line in response.split("\n"):
            line = line.strip()
            # Remove numbering like "1.", "1)", "- " etc.
            import re
            line = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
            line = re.sub(r"^[-•]\s*", "", line).strip()
            if line and len(line) > 10:
                variants.append(line)

        # Always include original query as one of the variants
        if query not in variants:
            variants.insert(0, query)

        return variants[:n + 1]   # cap at n+1 (original + n variants)

    def expanded_retrieve(
        self,
        query: str,
        collection,
        model_wrapper: Dict[str, Any],
        k: int = 5,
        n_variants: int = 3,
    ) -> Dict[str, Any]:
        """
        Query expansion retrieval: generates multiple query variants,
        retrieves chunks for each, then merges and deduplicates.

        Args:
            query       : Original user query.
            collection  : ChromaDB collection.
            model_wrapper: Embedding model wrapper.
            k           : Final number of results.
            n_variants  : Number of query rewrites to generate.

        Returns:
            Retrieval result dict with query_variants attached.
        """
        from vectorstore import similarity_search

        start = time.time()

        # Step 1: Generate query variants
        variants  = self.generate_query_variants(query, n=n_variants)
        llm_time  = round((time.time() - start) * 1000, 2)

        # Step 2: Retrieve for each variant (fetch k results each)
        all_chunks  = []
        seen_texts  = set()
        fetch_per_q = max(k, 3)

        for variant in variants:
            results = similarity_search(
                query=variant,
                collection=collection,
                model_wrapper=model_wrapper,
                k=fetch_per_q,
            )
            for chunk in results:
                text_key = chunk["text"][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    chunk["source_query"] = variant   # track which variant found it
                    all_chunks.append(chunk)

        # Step 3: Re-sort merged chunks by distance and take top-k
        all_chunks.sort(key=lambda x: x["distance"])
        top_chunks = all_chunks[:k]
        for i, chunk in enumerate(top_chunks):
            chunk["rank"] = i + 1

        total_latency = round((time.time() - start) * 1000, 2)

        return {
            "query"            : query,
            "query_variants"   : variants,
            "chunks"           : top_chunks,
            "latency_ms"       : total_latency,
            "llm_latency_ms"   : llm_time,
            "total_candidates" : len(all_chunks),
            "collection_name"  : getattr(collection, "name", "unknown") + "_expanded",
            "model_name"       : model_wrapper["name"],
            "k"                : k,
            "rewrite_strategy" : "expansion",
        }


# ── Compare: Original vs Rewritten ────────────────────────────────────────────

def compare_rewrite_strategies(
    query: str,
    rewriter: "QueryRewriter",
    collection,
    model_wrapper: Dict[str, Any],
    k: int = 5,
) -> None:
    """
    Runs all three retrieval strategies and compares results.
    """
    from retrieval import retrieve

    print(f"\n{'='*65}")
    print(f"QUERY: {query}")
    print(f"{'='*65}")

    # Original
    orig = retrieve(query, collection, model_wrapper, k=k)
    print(f"\n── Original Query ──")
    for c in orig["chunks"]:
        src  = c["metadata"].get("source", "?")
        page = c["metadata"].get("page",   "?")
        print(f"  [{c['rank']}] {src} p.{page} | {c['text'][:100].strip()}...")
    print(f"  Latency: {orig['latency_ms']}ms")

    # HyDE
    hyde = rewriter.hyde_retrieve(query, collection, model_wrapper, k=k)
    print(f"\n── HyDE (hypothetical document embedding) ──")
    print(f"  Hypothetical doc: {hyde['hypothetical_doc'][:200]}...")
    for c in hyde["chunks"]:
        src  = c["metadata"].get("source", "?")
        page = c["metadata"].get("page",   "?")
        print(f"  [{c['rank']}] {src} p.{page} | {c['text'][:100].strip()}...")
    print(f"  Latency: {hyde['latency_ms']}ms (LLM: {hyde['llm_latency_ms']}ms)")

    # Expansion
    exp = rewriter.expanded_retrieve(query, collection, model_wrapper, k=k)
    print(f"\n── Query Expansion ({len(exp['query_variants'])} variants) ──")
    for i, v in enumerate(exp["query_variants"]):
        print(f"  Variant {i}: {v}")
    for c in exp["chunks"]:
        src  = c["metadata"].get("source", "?")
        page = c["metadata"].get("page",   "?")
        sq   = c.get("source_query", "")[:40]
        print(f"  [{c['rank']}] {src} p.{page} | via: '{sq}...'")
    print(f"  Latency: {exp['latency_ms']}ms | {exp['total_candidates']} total candidates")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from embedding import load_both_models
    from vectorstore import load_collection

    models     = load_both_models(device="cuda")
    collection = load_collection("recursive_bge")
    rewriter   = QueryRewriter()

    # Test on hard queries and easy queries
    test_queries = [
        "How do companies handle model drift in deployed supply chain AI systems?",
        "What is the impact of AI on warehouse operations and fulfilment?",
        "What are the main applications of AI in FMCG supply chains?",
    ]

    for query in test_queries:
        compare_rewrite_strategies(query, rewriter, collection, models["bge"], k=5)