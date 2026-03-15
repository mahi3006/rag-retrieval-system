"""
bonuses/latency_analysis.py
---------------------------
Bonus 4: Latency and cost analysis across all pipeline variants.

WHAT THIS MEASURES:
  End-to-end latency broken down by stage for each combination:
    - Embedding load time (one-time)
    - Query embedding time
    - Vector search time
    - Reranking time (if applicable)
    - LLM generation time
    - Total end-to-end time

  Token cost estimation:
    Estimates approximate cost if the same workload were run on
    commercial APIs (OpenAI, Anthropic) vs our free local setup.
    This makes a strong case for the open-source stack.

USAGE:
    python bonuses/latency_analysis.py
    Results saved to evaluation/results/latency_analysis.csv
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# ── Token Counting ─────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Rough token count: ~4 characters per token (standard approximation).
    Good enough for cost estimation.
    """
    return len(text) // 4


# ── Commercial API Cost Rates (as of early 2025, USD per 1M tokens) ───────────

COST_RATES = {
    "openai_gpt4o": {
        "input_per_1m" : 5.00,
        "output_per_1m": 15.00,
    },
    "openai_gpt4o_mini": {
        "input_per_1m" : 0.15,
        "output_per_1m": 0.60,
    },
    "anthropic_claude_sonnet": {
        "input_per_1m" : 3.00,
        "output_per_1m": 15.00,
    },
    "local_mistral_7b": {
        "input_per_1m" : 0.00,    # free — running locally
        "output_per_1m": 0.00,
    },
}

# Average tokens per RAG call (estimated from our pipeline)
AVG_INPUT_TOKENS  = 1200   # system prompt + 5 chunks + query
AVG_OUTPUT_TOKENS = 250    # typical answer length


# ── Stage Latency Measurement ─────────────────────────────────────────────────

def measure_query_embedding_latency(
    queries: List[str],
    model_wrapper: Dict[str, Any],
    n_runs: int = 5,
) -> Dict[str, float]:
    """
    Measures average query embedding latency over n_runs.

    Args:
        queries      : List of query strings to embed.
        model_wrapper: Embedding model wrapper.
        n_runs       : Number of times to repeat for averaging.

    Returns:
        Dict with min, max, mean latency in ms.
    """
    embedder = model_wrapper["embedder"]
    prefix   = model_wrapper["query_prefix"]
    latencies = []

    for _ in range(n_runs):
        for query in queries:
            start = time.time()
            embedder.embed_query(prefix + query)
            latencies.append((time.time() - start) * 1000)

    return {
        "min_ms" : round(min(latencies), 2),
        "max_ms" : round(max(latencies), 2),
        "mean_ms": round(sum(latencies) / len(latencies), 2),
    }


def measure_vector_search_latency(
    queries: List[str],
    collection,
    model_wrapper: Dict[str, Any],
    k: int = 5,
    n_runs: int = 5,
) -> Dict[str, float]:
    """
    Measures average vector search latency (embed + search) over n_runs.
    """
    from vectorstore import similarity_search
    latencies = []

    for _ in range(n_runs):
        for query in queries:
            start = time.time()
            similarity_search(query, collection, model_wrapper, k=k)
            latencies.append((time.time() - start) * 1000)

    return {
        "min_ms" : round(min(latencies), 2),
        "max_ms" : round(max(latencies), 2),
        "mean_ms": round(sum(latencies) / len(latencies), 2),
    }


def measure_generation_latency(
    queries: List[str],
    collection,
    model_wrapper: Dict[str, Any],
    n_samples: int = 3,
) -> Dict[str, float]:
    """
    Measures LLM generation latency on a small sample of queries.
    Uses n_samples queries to keep total time reasonable.
    """
    from generation import rag_answer
    latencies = []

    print(f"  [Latency] Measuring generation on {n_samples} queries...")
    for query in queries[:n_samples]:
        result = rag_answer(query, collection, model_wrapper, k=5)
        latencies.append(result["generation_latency_ms"])
        print(f"    {query[:50]}... → {result['generation_latency_ms']}ms")

    return {
        "min_ms" : round(min(latencies), 2),
        "max_ms" : round(max(latencies), 2),
        "mean_ms": round(sum(latencies) / len(latencies), 2),
    }


# ── Cost Estimation ────────────────────────────────────────────────────────────

def estimate_cost(
    n_queries: int,
    avg_input_tokens: int  = AVG_INPUT_TOKENS,
    avg_output_tokens: int = AVG_OUTPUT_TOKENS,
) -> pd.DataFrame:
    """
    Estimates cost of running n_queries through each LLM option.

    Args:
        n_queries         : Number of queries to cost.
        avg_input_tokens  : Average input tokens per query.
        avg_output_tokens : Average output tokens per query.

    Returns:
        DataFrame with cost estimates per LLM option.
    """
    rows = []
    total_input_tokens  = n_queries * avg_input_tokens
    total_output_tokens = n_queries * avg_output_tokens

    for model_name, rates in COST_RATES.items():
        input_cost  = (total_input_tokens  / 1_000_000) * rates["input_per_1m"]
        output_cost = (total_output_tokens / 1_000_000) * rates["output_per_1m"]
        total_cost  = input_cost + output_cost

        rows.append({
            "model"               : model_name,
            "n_queries"           : n_queries,
            "total_input_tokens"  : total_input_tokens,
            "total_output_tokens" : total_output_tokens,
            "input_cost_usd"      : round(input_cost,  4),
            "output_cost_usd"     : round(output_cost, 4),
            "total_cost_usd"      : round(total_cost,  4),
        })

    return pd.DataFrame(rows)


# ── Full Latency Report ────────────────────────────────────────────────────────

def run_latency_analysis(
    test_queries: List[str],
    models: Dict[str, Any],
    collections: Dict[str, Any],
    n_gen_samples: int = 3,
) -> None:
    """
    Runs the full latency analysis and prints a report.

    Args:
        test_queries  : List of query strings.
        models        : Dict from load_both_models().
        collections   : Dict from load_all_collections().
        n_gen_samples : Queries to use for generation latency measurement.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    collection_model_map = {
        "fixed_minilm"    : models["minilm"],
        "fixed_bge"       : models["bge"],
        "recursive_minilm": models["minilm"],
        "recursive_bge"   : models["bge"],
    }

    print("\n" + "="*65)
    print("  LATENCY ANALYSIS")
    print("="*65)

    all_rows = []

    for combo_name, collection in collections.items():
        model_wrapper = collection_model_map[combo_name]
        print(f"\n[Latency] Measuring: {combo_name}")

        # Query embedding latency
        embed_stats = measure_query_embedding_latency(
            test_queries[:5], model_wrapper, n_runs=3
        )

        # Vector search latency
        search_stats = measure_vector_search_latency(
            test_queries[:5], collection, model_wrapper, k=5, n_runs=3
        )

        all_rows.append({
            "combo"                  : combo_name,
            "embedding_model"        : model_wrapper["name"],
            "chunking"               : combo_name.split("_")[0],
            "embed_query_mean_ms"    : embed_stats["mean_ms"],
            "embed_query_min_ms"     : embed_stats["min_ms"],
            "embed_query_max_ms"     : embed_stats["max_ms"],
            "vector_search_mean_ms"  : search_stats["mean_ms"],
            "vector_search_min_ms"   : search_stats["min_ms"],
            "vector_search_max_ms"   : search_stats["max_ms"],
        })

    latency_df = pd.DataFrame(all_rows)

    # Generation latency (once, using best combo)
    print(f"\n[Latency] Measuring generation latency (recursive_bge, {n_gen_samples} queries)...")
    gen_stats = measure_generation_latency(
        queries=test_queries,
        collection=collections["recursive_bge"],
        model_wrapper=models["bge"],
        n_samples=n_gen_samples,
    )

    # Print retrieval latency table
    print(f"\n{'='*65}")
    print("  RETRIEVAL LATENCY (ms) — averaged over 3 runs × 5 queries")
    print(f"{'='*65}")
    print(f"  {'Combo':<25} {'Embed(ms)':>10} {'Search(ms)':>12} {'Total(ms)':>10}")
    print(f"  {'-'*60}")
    for _, row in latency_df.iterrows():
        total = row["embed_query_mean_ms"] + row["vector_search_mean_ms"]
        print(f"  {row['combo']:<25} "
              f"{row['embed_query_mean_ms']:>10.1f} "
              f"{row['vector_search_mean_ms']:>12.1f} "
              f"{total:>10.1f}")

    # Print generation latency
    print(f"\n  LLM Generation (Mistral 7B local):")
    print(f"    Mean : {gen_stats['mean_ms']}ms")
    print(f"    Min  : {gen_stats['min_ms']}ms")
    print(f"    Max  : {gen_stats['max_ms']}ms")

    # Cost comparison
    print(f"\n{'='*65}")
    print("  COST ESTIMATION (20 queries, local vs commercial APIs)")
    print(f"{'='*65}")
    cost_df = estimate_cost(n_queries=20)
    for _, row in cost_df.iterrows():
        print(f"  {row['model']:<30} ${row['total_cost_usd']:>8.4f}")
    print(f"\n  → Local Mistral saves "
          f"${cost_df[cost_df['model']=='openai_gpt4o']['total_cost_usd'].iloc[0]:.4f} "
          f"vs GPT-4o for this eval run.")

    # Save results
    out_path = RESULTS_DIR / "latency_analysis.csv"
    latency_df.to_csv(out_path, index=False)

    cost_path = RESULTS_DIR / "cost_analysis.csv"
    cost_df.to_csv(cost_path, index=False)

    # Save generation stats
    gen_path = RESULTS_DIR / "generation_latency.json"
    with open(gen_path, "w") as f:
        json.dump(gen_stats, f, indent=2)

    print(f"\n[Latency] Results saved:")
    print(f"  {out_path}")
    print(f"  {cost_path}")
    print(f"  {gen_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json

    from embedding import load_both_models
    from vectorstore import load_all_collections

    # Load test queries
    queries_path = PROJECT_ROOT / "evaluation" / "test_queries.json"
    with open(queries_path) as f:
        test_queries_data = _json.load(f)
    test_queries = [q["query"] for q in test_queries_data]

    print("[Latency] Loading models and collections...")
    models      = load_both_models(device="cuda")
    collections = load_all_collections()

    run_latency_analysis(
        test_queries=test_queries,
        models=models,
        collections=collections,
        n_gen_samples=3,    # only 3 generation calls to keep it fast
    )