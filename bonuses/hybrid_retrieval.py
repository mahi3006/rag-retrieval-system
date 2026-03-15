"""
bonuses/hybrid_retrieval.py
---------------------------
Bonus 1: Hybrid Retrieval — BM25 (keyword) + Vector Search combined
using Reciprocal Rank Fusion (RRF).

WHY THIS EXISTS:
  Pure vector search failed on q19 (warehouse/fulfilment) and q20 (MLOps/
  model drift) because those terms are keyword-sparse in embedding space.
  BM25 excels at exact keyword matches. Combining both covers weaknesses
  of each approach.

HOW RRF WORKS:
  Both BM25 and vector search return ranked lists independently.
  RRF score for a chunk = sum of 1/(RRF_K + rank) across all lists.
  Chunks appearing high in BOTH lists get the highest combined scores.
  RRF_K=60 is standard — dampens the impact of very high ranks.

USAGE:
  from bonuses.hybrid_retrieval import HybridRetriever
  retriever = HybridRetriever(chunks, collection, model_wrapper)
  results   = retriever.retrieve("model drift in supply chain", k=5)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vectorstore import similarity_search


# ── Configuration ──────────────────────────────────────────────────────────────

RRF_K         = 60     # RRF constant — standard value, dampens rank effects
DEFAULT_K     = 5      # default number of results to return
FETCH_K_MULT  = 3      # fetch this × k from each source before fusing
                       # e.g. k=5 → fetch 15 from BM25 and 15 from vector


# ── BM25 Index ─────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Builds a BM25 index over a list of Document chunks.

    BM25 (Best Match 25) is a classic keyword-based ranking function.
    It scores documents based on term frequency and inverse document frequency.
    Exact or near-exact keyword matches score very highly.
    """

    def __init__(self, chunks: List[Document]):
        """
        Args:
            chunks: List of Document objects (from chunking.py).
                    The index is built over their page_content.
        """
        self.chunks    = chunks
        self.tokenized = [self._tokenize(doc.page_content) for doc in chunks]
        self.bm25      = BM25Okapi(self.tokenized)
        print(f"[BM25] Index built over {len(chunks)} chunks")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple whitespace + lowercase tokenizer.
        Removes punctuation for cleaner matching.
        """
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
        return text.split()

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Returns top-k (index, score) tuples for a query.

        Args:
            query : Query string.
            k     : Number of results.

        Returns:
            List of (chunk_index, bm25_score) sorted by score descending.
        """
        tokenized_query = self._tokenize(query)
        scores          = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [(idx, float(scores[idx])) for idx in top_k_indices]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    rrf_k: int = RRF_K,
) -> Dict[str, float]:
    """
    Combines multiple ranked lists into one using RRF.

    Formula: score(chunk) = sum over all lists of 1 / (rrf_k + rank)
    where rank is 1-indexed position in each list.

    Args:
        ranked_lists: List of lists of chunk IDs, each sorted by relevance.
                      e.g. [["id_5", "id_2", "id_9"], ["id_2", "id_5", "id_1"]]
        rrf_k       : RRF constant (default 60).

    Returns:
        Dict mapping chunk_id -> RRF score (higher = more relevant).
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (rrf_k + rank)
    return scores


# ── Hybrid Retriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 keyword search and vector similarity search using RRF.

    Setup (once per collection):
        retriever = HybridRetriever(chunks, collection, model_wrapper)

    Query (many times):
        results = retriever.retrieve("your question", k=5)
    """

    def __init__(
        self,
        chunks: List[Document],
        collection,
        model_wrapper: Dict[str, Any],
    ):
        """
        Args:
            chunks       : The same chunks used to build the ChromaDB collection.
                           Needed to build the BM25 index.
            collection   : Loaded ChromaDB collection.
            model_wrapper: Embedding model wrapper matching the collection.
        """
        self.chunks        = chunks
        self.collection    = collection
        self.model_wrapper = model_wrapper

        # Build BM25 index over chunk texts
        self.bm25_index = BM25Index(chunks)

        # Build a lookup: chunk text → chunk dict for fast access
        # We use the first 200 chars as a key (text is unique enough)
        self._chunk_lookup = {
            doc.page_content[:200]: {
                "text"    : doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in chunks
        }

        col_name  = getattr(collection, "name", "unknown")
        mod_name  = model_wrapper["name"]
        print(f"[Hybrid] Ready — collection={col_name}, model={mod_name}")

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        alpha: float = 0.5,    # weight for vector vs BM25 (0=pure BM25, 1=pure vector)
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval: BM25 + vector search fused with RRF.

        Args:
            query : User's question string.
            k     : Number of final results to return.
            alpha : Unused in RRF (RRF is parameter-free for ranking).
                    Kept for interface compatibility with future weighted fusion.

        Returns:
            Same structure as retrieval.retrieve() for drop-in compatibility:
            {
                "query"           : str,
                "chunks"          : List[Dict],
                "latency_ms"      : float,
                "collection_name" : str,
                "model_name"      : str,
                "k"               : int,
                "retrieval_type"  : "hybrid",
            }
        """
        start   = time.time()
        fetch_k = k * FETCH_K_MULT   # fetch more from each source, then fuse

        # ── Step 1: BM25 retrieval ──────────────────────────────────────────
        bm25_results = self.bm25_index.search(query, k=fetch_k)
        bm25_ranked  = []
        bm25_data    = {}

        for chunk_idx, score in bm25_results:
            chunk_doc = self.chunks[chunk_idx]
            chunk_id  = f"bm25_{chunk_idx}"
            bm25_ranked.append(chunk_id)
            bm25_data[chunk_id] = {
                "text"      : chunk_doc.page_content,
                "metadata"  : chunk_doc.metadata,
                "bm25_score": score,
                "bm25_rank" : len(bm25_ranked),
            }

        # ── Step 2: Vector search ───────────────────────────────────────────
        vector_results = similarity_search(
            query=query,
            collection=self.collection,
            model_wrapper=self.model_wrapper,
            k=fetch_k,
        )
        vector_ranked = []
        vector_data   = {}

        for result in vector_results:
            # Use first 200 chars as a stable ID to match against BM25 results
            chunk_key = result["text"][:200]
            chunk_id  = f"vec_{result['rank']}"
            vector_ranked.append(chunk_id)
            vector_data[chunk_id] = {
                "text"        : result["text"],
                "metadata"    : result["metadata"],
                "vector_dist" : result["distance"],
                "vector_rank" : result["rank"],
                "chunk_key"   : chunk_key,
            }

        # ── Step 3: RRF fusion ──────────────────────────────────────────────
        rrf_scores = reciprocal_rank_fusion([bm25_ranked, vector_ranked])

        # Sort all candidate IDs by RRF score descending
        all_ids_sorted = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # ── Step 4: Build final top-k results ───────────────────────────────
        final_chunks = []
        seen_texts   = set()

        for chunk_id in all_ids_sorted:
            if len(final_chunks) >= k:
                break

            # Get chunk data from whichever source has it
            if chunk_id in bm25_data:
                data = bm25_data[chunk_id]
            elif chunk_id in vector_data:
                data = vector_data[chunk_id]
            else:
                continue

            # Deduplicate by text content
            text_key = data["text"][:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            final_chunks.append({
                "text"     : data["text"],
                "metadata" : data["metadata"],
                "distance" : rrf_scores[chunk_id],   # RRF score as "distance"
                "rank"     : len(final_chunks) + 1,
            })

        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "query"           : query,
            "chunks"          : final_chunks,
            "latency_ms"      : latency_ms,
            "collection_name" : getattr(self.collection, "name", "unknown") + "_hybrid",
            "model_name"      : self.model_wrapper["name"],
            "k"               : k,
            "retrieval_type"  : "hybrid",
        }


# ── Convenience: Compare Hybrid vs Vector ─────────────────────────────────────

def compare_hybrid_vs_vector(
    query: str,
    hybrid_retriever: "HybridRetriever",
    k: int = 5,
) -> None:
    """
    Runs the same query with both hybrid and pure vector retrieval
    and prints a side-by-side comparison.
    Useful for showing the improvement on hard queries like q19/q20.
    """
    from retrieval import retrieve, print_retrieval_result

    print(f"\n{'='*65}")
    print(f"QUERY: {query}")
    print(f"{'='*65}")

    # Vector only
    vector_result = retrieve(
        query=query,
        collection=hybrid_retriever.collection,
        model_wrapper=hybrid_retriever.model_wrapper,
        k=k,
    )
    print(f"\n── Vector Only ──")
    for chunk in vector_result["chunks"]:
        src  = chunk["metadata"].get("source", "?")
        page = chunk["metadata"].get("page",   "?")
        dist = round(chunk["distance"], 4)
        print(f"  [{chunk['rank']}] {src} p.{page} | dist={dist}")
        print(f"      {chunk['text'][:120].strip()}...")

    # Hybrid
    hybrid_result = hybrid_retriever.retrieve(query, k=k)
    print(f"\n── Hybrid (BM25 + Vector, RRF) ──")
    for chunk in hybrid_result["chunks"]:
        src  = chunk["metadata"].get("source", "?")
        page = chunk["metadata"].get("page",   "?")
        rrf  = round(chunk["distance"], 6)
        print(f"  [{chunk['rank']}] {src} p.{page} | rrf={rrf}")
        print(f"      {chunk['text'][:120].strip()}...")

    print(f"\n  Vector latency : {vector_result['latency_ms']}ms")
    print(f"  Hybrid latency : {hybrid_result['latency_ms']}ms")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from embedding import load_both_models
    from vectorstore import load_collection
    from chunking    import recursive_chunking
    from ingest      import load_all_pdfs

    print("[Hybrid] Loading corpus and building index...")
    docs   = load_all_pdfs()
    chunks = recursive_chunking(docs)

    models     = load_both_models(device="cuda")
    collection = load_collection("recursive_bge")

    retriever = HybridRetriever(
        chunks=chunks,
        collection=collection,
        model_wrapper=models["bge"],
    )

    # Test on the two queries that failed in pure vector search
    hard_queries = [
        "What is the impact of AI on warehouse operations and fulfilment?",   # q19
        "How do companies handle model drift in deployed supply chain AI systems?",  # q20
        "What are the main applications of AI in FMCG supply chains?",        # q01 (should still work)
    ]

    for query in hard_queries:
        compare_hybrid_vs_vector(query, retriever, k=5)