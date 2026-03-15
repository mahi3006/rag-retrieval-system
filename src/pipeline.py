"""
pipeline.py
-----------
Orchestrates the full RAG pipeline end-to-end.

Loads all 4 ChromaDB collections and both embedding models once,
then provides:
  - run_query()            : query all 4 combinations for one question
  - run_all_combinations() : used by the eval harness for batch evaluation
  - interactive_demo()     : type questions, see 4 answers compared live

This is the main entry point for the live demo.

Usage:
    python src/pipeline.py                        # interactive demo
    python src/pipeline.py --query "your question" # single query
"""

import sys
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding import load_both_models
from vectorstore import load_all_collections, COLLECTION_NAMES
from generation  import rag_answer, print_rag_result


# ── Pipeline State (loaded once) ──────────────────────────────────────────────

class RAGPipeline:
    """
    Holds all loaded models and collections.
    Load once, query many times — avoids re-loading on every call.
    """

    def __init__(self, device: str = "cuda", llm_model: str = "mistral"):
        print("\n[Pipeline] Initialising RAG pipeline...")
        start = time.time()

        print("[Pipeline] Loading embedding models...")
        self.models = load_both_models(device=device)

        print("[Pipeline] Loading ChromaDB collections...")
        self.collections = load_all_collections()

        self.llm_model = llm_model

        # Map each collection to its correct embedding model
        # fixed_minilm and recursive_minilm use "minilm"
        # fixed_bge    and recursive_bge    use "bge"
        self.collection_model_map = {
            "fixed_minilm"    : self.models["minilm"],
            "fixed_bge"       : self.models["bge"],
            "recursive_minilm": self.models["minilm"],
            "recursive_bge"   : self.models["bge"],
        }

        elapsed = time.time() - start
        print(f"[Pipeline] Ready in {elapsed:.1f}s\n")

    def run_query(
        self,
        query: str,
        k: int = 5,
        collections: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs a query against all 4 (or selected) combinations.

        Args:
            query      : The user's question.
            k          : Number of chunks to retrieve per combination.
            collections: List of collection names to query.
                         Defaults to all 4 if None.

        Returns:
            List of RAG result dicts, one per combination.
        """
        target_collections = collections or COLLECTION_NAMES
        results = []

        for col_name in target_collections:
            collection    = self.collections[col_name]
            model_wrapper = self.collection_model_map[col_name]

            result = rag_answer(
                query=query,
                collection=collection,
                model_wrapper=model_wrapper,
                k=k,
                llm_model=self.llm_model,
            )
            results.append(result)

        return results

    def run_all_combinations(
        self,
        queries: List[str],
        k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Runs all queries against all 4 combinations.
        Used by the evaluation harness.

        Args:
            queries: List of query strings.
            k      : Number of chunks to retrieve.

        Returns:
            {
                "fixed_minilm"    : [result_q1, result_q2, ...],
                "fixed_bge"       : [...],
                "recursive_minilm": [...],
                "recursive_bge"   : [...],
            }
        """
        print(f"\n[Pipeline] Running {len(queries)} queries × 4 combinations "
              f"= {len(queries) * 4} total RAG calls...")

        all_results = {col: [] for col in COLLECTION_NAMES}

        for i, query in enumerate(queries, 1):
            print(f"\n  Query {i}/{len(queries)}: {query[:70]}...")
            for col_name in COLLECTION_NAMES:
                collection    = self.collections[col_name]
                model_wrapper = self.collection_model_map[col_name]

                result = rag_answer(
                    query=query,
                    collection=collection,
                    model_wrapper=model_wrapper,
                    k=k,
                    llm_model=self.llm_model,
                )
                all_results[col_name].append(result)
                print(f"    ✓ {col_name} ({result['total_latency_ms']}ms)")

        return all_results


# ── Display Helpers ────────────────────────────────────────────────────────────

def print_comparison(results: List[Dict[str, Any]]) -> None:
    """
    Prints a side-by-side comparison of answers from all 4 combinations.
    Used in the interactive demo.
    """
    if not results:
        print("No results.")
        return

    print(f"\n{'='*70}")
    print(f"QUERY: {results[0]['query']}")
    print(f"{'='*70}")

    for result in results:
        label = f"{result['collection_name']} | embed={result['embedding_model']}"
        print(f"\n── {label} ──")
        print(f"   Answer    : {result['answer'][:400]}{'...' if len(result['answer']) > 400 else ''}")
        print(f"   {result['sources']}")
        print(f"   Latency   : {result['total_latency_ms']}ms total "
              f"(retrieval {result['retrieval_latency_ms']}ms + "
              f"generation {result['generation_latency_ms']}ms)")


def print_retrieval_only_comparison(results: List[Dict[str, Any]]) -> None:
    """
    Prints just the retrieved chunks across all combinations — no LLM call.
    Faster for debugging retrieval quality.
    """
    if not results:
        return

    print(f"\n{'='*70}")
    print(f"QUERY: {results[0]['query']}")
    print(f"{'='*70}")

    for result in results:
        print(f"\n── {result['collection_name']} ──")
        for chunk in result["chunks"]:
            source = chunk["metadata"].get("source", "?")
            page   = chunk["metadata"].get("page",   "?")
            dist   = round(chunk["distance"], 4)
            print(f"  [{chunk['rank']}] {source} p.{page} | dist={dist}")
            print(f"      {chunk['text'][:150].strip()}...")


# ── Interactive Demo ───────────────────────────────────────────────────────────

def interactive_demo(pipeline: RAGPipeline) -> None:
    """
    Starts an interactive loop where the user types questions
    and sees answers from all 4 combinations compared side by side.

    Type 'quit' or 'exit' to stop.
    Type 'retrieval' to toggle retrieval-only mode (no LLM generation).
    """
    print("\n" + "="*70)
    print("  RAG Benchmarking System — Interactive Demo")
    print("  Collections: fixed_minilm | fixed_bge | recursive_minilm | recursive_bge")
    print("  Commands: 'quit' to exit | 'retrieval' for retrieval-only mode")
    print("="*70)

    retrieval_only = False

    while True:
        try:
            mode_label = "[retrieval-only]" if retrieval_only else "[full RAG]"
            query = input(f"\n{mode_label} Enter question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Demo] Exiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("[Demo] Goodbye.")
            break

        if query.lower() == "retrieval":
            retrieval_only = not retrieval_only
            status = "ON" if retrieval_only else "OFF"
            print(f"[Demo] Retrieval-only mode: {status}")
            continue

        if retrieval_only:
            # Fast path: just retrieval, no generation
            from retrieval import retrieve
            results = []
            for col_name in COLLECTION_NAMES:
                col           = pipeline.collections[col_name]
                model_wrapper = pipeline.collection_model_map[col_name]
                result        = retrieve(query, col, model_wrapper, k=5)
                result["collection_name"] = col_name
                results.append(result)
            print_retrieval_only_comparison(results)
        else:
            results = pipeline.run_query(query, k=5)
            print_comparison(results)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Benchmarking Pipeline")
    parser.add_argument("--query",  type=str, help="Run a single query against all combinations")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--model",  type=str, default="mistral", help="Ollama model name")
    args = parser.parse_args()

    pipeline = RAGPipeline(device=args.device, llm_model=args.model)

    if args.query:
        results = pipeline.run_query(args.query, k=5)
        print_comparison(results)
    else:
        interactive_demo(pipeline)