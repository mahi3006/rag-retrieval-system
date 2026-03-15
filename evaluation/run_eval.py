"""
run_eval.py
-----------
Runs the full evaluation harness:
  1. Loads all 20 test queries from test_queries.json
  2. Runs retrieval across all 4 combinations for each query
  3. Computes precision@k, recall@k, MRR, hit rate per query per combination
  4. Aggregates results into a comparison table
  5. Saves results to evaluation/results/eval_results.csv
  6. Prints the final comparison table

NOTE: This runs retrieval only — no LLM generation needed for metrics.
     This means the full eval runs in ~2 minutes, not 40 minutes.
     LLM generation is only needed for the demo and qualitative scoring.

Usage:
    python evaluation/run_eval.py           # full eval, all 4 combinations
    python evaluation/run_eval.py --k 3     # use top-3 instead of top-5
    python evaluation/run_eval.py --combo recursive_bge  # single combo
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding  import load_both_models
from vectorstore import load_all_collections, COLLECTION_NAMES
from retrieval   import retrieve
from metrics     import compute_query_metrics, compute_aggregate_metrics


# ── Paths ──────────────────────────────────────────────────────────────────────

QUERIES_PATH = PROJECT_ROOT / "evaluation" / "test_queries.json"
RESULTS_DIR  = PROJECT_ROOT / "evaluation" / "results"


# ── Load Test Queries ──────────────────────────────────────────────────────────

def load_test_queries(path: Path = QUERIES_PATH) -> List[Dict[str, Any]]:
    """Loads test queries from JSON file."""
    with open(path, "r") as f:
        queries = json.load(f)
    print(f"[Eval] Loaded {len(queries)} test queries from {path.name}")
    return queries


# ── Run Eval for One Combination ───────────────────────────────────────────────

def evaluate_combination(
    collection_name: str,
    collection,
    model_wrapper: Dict[str, Any],
    test_queries: List[Dict[str, Any]],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Runs retrieval + metric computation for all queries on one combination.

    Args:
        collection_name: e.g. "recursive_bge"
        collection     : ChromaDB collection object
        model_wrapper  : Embedding model wrapper
        test_queries   : List of test query dicts
        k              : Top-k cutoff for metrics

    Returns:
        List of per-query result dicts with metrics attached.
    """
    print(f"\n[Eval] Evaluating: {collection_name}")
    results = []

    for query_dict in test_queries:
        query_id         = query_dict["id"]
        query_text       = query_dict["query"]
        keywords         = query_dict["relevant_keywords"]
        difficulty       = query_dict["difficulty"]
        category         = query_dict["category"]

        # Retrieve
        retrieval_result = retrieve(
            query=query_text,
            collection=collection,
            model_wrapper=model_wrapper,
            k=k,
        )

        # Compute metrics
        query_metrics = compute_query_metrics(
            chunks=retrieval_result["chunks"],
            relevant_keywords=keywords,
            k=k,
        )

        # Top sources retrieved (for qualitative inspection)
        top_sources = [
            f"{c['metadata'].get('source','?')} p.{c['metadata'].get('page','?')}"
            for c in retrieval_result["chunks"][:3]
        ]

        results.append({
            # Query info
            "query_id"          : query_id,
            "query"             : query_text,
            "difficulty"        : difficulty,
            "category"          : category,
            # Combination info
            "collection"        : collection_name,
            "embedding_model"   : model_wrapper["name"],
            "chunking_strategy" : collection_name.split("_")[0],  # "fixed" or "recursive"
            # Metrics
            "precision_at_k"    : query_metrics["precision_at_k"],
            "recall_at_k"       : query_metrics["recall_at_k"],
            "reciprocal_rank"   : query_metrics["reciprocal_rank"],
            "hit_rate_at_k"     : query_metrics["hit_rate_at_k"],
            "num_relevant_in_k" : query_metrics["num_relevant_in_k"],
            # Latency
            "retrieval_latency_ms": retrieval_result["latency_ms"],
            # Top sources (for inspection)
            "top_sources"       : " | ".join(top_sources),
            "k"                 : k,
        })

        # Live progress
        p  = query_metrics["precision_at_k"]
        rr = query_metrics["reciprocal_rank"]
        print(f"  {query_id} | P@{k}={p:.2f} RR={rr:.2f} | {query_text[:55]}...")

    return results


# ── Run Full Eval ──────────────────────────────────────────────────────────────

def run_full_eval(
    k: int = 5,
    target_combos: List[str] = None,
) -> pd.DataFrame:
    """
    Runs the full evaluation across all (or selected) combinations.

    Args:
        k             : Top-k cutoff.
        target_combos : Subset of COLLECTION_NAMES to evaluate.
                        Defaults to all 4.

    Returns:
        DataFrame with one row per (query, combination) pair.
    """
    combos = target_combos or COLLECTION_NAMES

    print("\n[Eval] Loading models and collections...")
    models      = load_both_models(device="cuda")
    collections = load_all_collections()

    collection_model_map = {
        "fixed_minilm"    : models["minilm"],
        "fixed_bge"       : models["bge"],
        "recursive_minilm": models["minilm"],
        "recursive_bge"   : models["bge"],
    }

    test_queries = load_test_queries()

    all_rows  = []
    eval_start = time.time()

    for combo in combos:
        rows = evaluate_combination(
            collection_name=combo,
            collection=collections[combo],
            model_wrapper=collection_model_map[combo],
            test_queries=test_queries,
            k=k,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    total_time = round(time.time() - eval_start, 1)
    print(f"\n[Eval] Completed in {total_time}s — {len(df)} rows total")

    return df


# ── Aggregate & Print Results Table ───────────────────────────────────────────

def print_results_table(df: pd.DataFrame, k: int = 5) -> None:
    """
    Prints the main comparison table aggregated by combination.
    This is the table that goes into your documentation.
    """
    agg = df.groupby("collection").agg(
        mean_precision   =("precision_at_k",      "mean"),
        mean_recall      =("recall_at_k",          "mean"),
        mrr              =("reciprocal_rank",       "mean"),
        mean_hit_rate    =("hit_rate_at_k",         "mean"),
        mean_latency_ms  =("retrieval_latency_ms",  "mean"),
        num_queries      =("query_id",              "count"),
    ).round(4)

    print(f"\n{'='*75}")
    print(f"  RETRIEVAL EVALUATION RESULTS  (k={k}, n={agg['num_queries'].iloc[0]} queries)")
    print(f"{'='*75}")
    print(f"  {'Collection':<25} {'P@k':>6} {'R@k':>6} {'MRR':>6} {'Hit@k':>6} {'Lat(ms)':>9}")
    print(f"  {'-'*65}")
    for col_name, row in agg.iterrows():
        print(f"  {col_name:<25} "
              f"{row['mean_precision']:>6.4f} "
              f"{row['mean_recall']:>6.4f} "
              f"{row['mean_mrr'] if 'mean_mrr' in row else row['mrr']:>6.4f} "
              f"{row['mean_hit_rate']:>6.4f} "
              f"{row['mean_latency_ms']:>9.1f}")
    print(f"{'='*75}")

    # Best combo per metric
    print(f"\n  Best P@k   : {agg['mean_precision'].idxmax()}")
    print(f"  Best R@k   : {agg['mean_recall'].idxmax()}")
    print(f"  Best MRR   : {agg['mrr'].idxmax()}")
    print(f"  Best Hit@k : {agg['mean_hit_rate'].idxmax()}")
    print(f"  Fastest    : {agg['mean_latency_ms'].idxmin()}")


def print_difficulty_breakdown(df: pd.DataFrame) -> None:
    """Breaks down performance by query difficulty per combination."""
    print(f"\n[Results by Difficulty]")
    breakdown = df.groupby(["collection", "difficulty"]).agg(
        mean_precision=("precision_at_k", "mean"),
        mean_mrr      =("reciprocal_rank", "mean"),
        count         =("query_id",        "count"),
    ).round(3)
    print(breakdown.to_string())


def print_category_breakdown(df: pd.DataFrame) -> None:
    """Breaks down performance by query category — shows where each combo excels."""
    print(f"\n[Results by Category — MRR]")
    pivot = df.pivot_table(
        values="reciprocal_rank",
        index="category",
        columns="collection",
        aggfunc="mean",
    ).round(3)
    print(pivot.to_string())


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame) -> Path:
    """Saves the full results DataFrame to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Eval] Results saved to {out_path}")
    return out_path


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation Harness")
    parser.add_argument("--k",     type=int, default=5,    help="Top-k cutoff (default: 5)")
    parser.add_argument("--combo", type=str, default=None,
                        help=f"Single combo to evaluate. Options: {COLLECTION_NAMES}")
    args = parser.parse_args()

    target = [args.combo] if args.combo else None

    # Run evaluation
    df = run_full_eval(k=args.k, target_combos=target)

    # Print tables
    print_results_table(df, k=args.k)
    print_difficulty_breakdown(df)
    print_category_breakdown(df)

    # Save to CSV
    save_results(df)

    print("\n[Eval] Done. Next step: open notebooks/analysis.ipynb to visualise results.")