"""
vectorstore.py
--------------
Builds and persists 4 ChromaDB collections — one per combination of
chunking strategy x embedding model:

    fixed_minilm       fixed chunks   + MiniLM embeddings
    fixed_bge          fixed chunks   + BGE embeddings
    recursive_minilm   recursive chunks + MiniLM embeddings
    recursive_bge      recursive chunks + BGE embeddings

Collections are persisted to data/chroma_db/ so we embed once and
query as many times as needed without re-embedding.

Usage:
    # Build all 4 collections (run once)
    python src/vectorstore.py --build

    # Then query a collection
    from vectorstore import load_collection, similarity_search
    col = load_collection("recursive_bge")
    results = similarity_search("AI in demand forecasting", col, k=5)
"""

import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# All valid collection names
COLLECTION_NAMES = [
    "fixed_minilm",
    "fixed_bge",
    "recursive_minilm",
    "recursive_bge",
]


# ── ChromaDB Client ────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client that saves data to CHROMA_DB_DIR.
    Creates the directory if it does not exist.
    """
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client


# ── Build a Single Collection ──────────────────────────────────────────────────

def build_collection(
    collection_name: str,
    chunks: List[Document],
    model_wrapper: Dict[str, Any],
    batch_size: int = 100,
    overwrite: bool = False,
) -> chromadb.Collection:
    """
    Embeds chunks and stores them in a named ChromaDB collection.

    Args:
        collection_name : One of the 4 COLLECTION_NAMES.
        chunks          : List of Document objects (from chunking.py).
        model_wrapper   : Embedding model wrapper dict (from embeddings.py).
        batch_size      : Number of chunks to embed and insert at once.
                          Keeps GPU memory usage stable.
        overwrite       : If True, delete and recreate the collection.
                          If False, skip if collection already exists.

    Returns:
        The populated ChromaDB Collection object.
    """
    client = get_chroma_client()

    # Handle existing collections
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        if overwrite:
            print(f"[VectorStore] Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            print(f"[VectorStore] Collection '{collection_name}' already exists — skipping.")
            print(f"             Use overwrite=True to rebuild.")
            return client.get_collection(collection_name)

    print(f"\n[VectorStore] Building collection: {collection_name}")
    print(f"  Chunks to embed : {len(chunks)}")
    print(f"  Embedding model : {model_wrapper['name']}")
    print(f"  Batch size      : {batch_size}")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for normalised vectors
    )

    embedder   = model_wrapper["embedder"]
    start_time = time.time()

    # Process in batches to keep memory stable
    for batch_start in tqdm(
        range(0, len(chunks), batch_size),
        desc=f"Embedding {collection_name}",
    ):
        batch = chunks[batch_start : batch_start + batch_size]

        texts     = [doc.page_content for doc in batch]
        metadatas = [doc.metadata     for doc in batch]
        ids       = [f"{collection_name}_{batch_start + i}" for i in range(len(batch))]

        # Embed the batch (no query prefix for documents)
        embeddings = embedder.embed_documents(texts)

        # Insert into ChromaDB
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    elapsed = time.time() - start_time
    print(f"  ✓ Done in {elapsed:.1f}s — {collection.count()} chunks stored")

    return collection


# ── Build All 4 Collections ────────────────────────────────────────────────────

def build_all_collections(
    fixed_chunks: List[Document],
    recursive_chunks: List[Document],
    models: Dict[str, Dict[str, Any]],
    overwrite: bool = False,
) -> Dict[str, chromadb.Collection]:
    """
    Builds all 4 collections from the two chunk sets and two models.

    Args:
        fixed_chunks     : Output of fixed_size_chunking()
        recursive_chunks : Output of recursive_chunking()
        models           : Output of load_both_models() — dict with "minilm" and "bge"
        overwrite        : Whether to rebuild existing collections

    Returns:
        Dict mapping collection_name -> ChromaDB Collection
    """
    combinations = {
        "fixed_minilm"    : (fixed_chunks,     models["minilm"]),
        "fixed_bge"       : (fixed_chunks,     models["bge"]),
        "recursive_minilm": (recursive_chunks, models["minilm"]),
        "recursive_bge"   : (recursive_chunks, models["bge"]),
    }

    collections = {}
    total_start = time.time()

    for name, (chunks, model_wrapper) in combinations.items():
        collections[name] = build_collection(
            collection_name=name,
            chunks=chunks,
            model_wrapper=model_wrapper,
            overwrite=overwrite,
        )

    print(f"\n[VectorStore] All collections built in {time.time() - total_start:.1f}s")
    print_collection_stats(collections)

    return collections


# ── Load an Existing Collection ────────────────────────────────────────────────

def load_collection(collection_name: str) -> chromadb.Collection:
    """
    Loads an already-built collection from disk.

    Args:
        collection_name: One of the 4 COLLECTION_NAMES.

    Returns:
        ChromaDB Collection object ready to query.
    """
    assert collection_name in COLLECTION_NAMES, (
        f"Unknown collection: {collection_name}\n"
        f"Valid names: {COLLECTION_NAMES}"
    )

    client = get_chroma_client()
    existing = [c.name for c in client.list_collections()]

    if collection_name not in existing:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in {CHROMA_DB_DIR}.\n"
            f"Run: python src/vectorstore.py --build"
        )

    return client.get_collection(collection_name)


def load_all_collections() -> Dict[str, chromadb.Collection]:
    """Loads all 4 collections from disk."""
    return {name: load_collection(name) for name in COLLECTION_NAMES}


# ── Similarity Search ──────────────────────────────────────────────────────────

def similarity_search(
    query: str,
    collection: chromadb.Collection,
    model_wrapper: Dict[str, Any],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Embeds the query and retrieves the top-k most similar chunks.

    Args:
        query        : User's question string.
        collection   : A loaded ChromaDB collection.
        model_wrapper: The embedding model wrapper (must match the collection).
        k            : Number of results to return.

    Returns:
        List of dicts, each containing:
            {
                "text"     : chunk text,
                "metadata" : chunk metadata (source, page, etc.),
                "distance" : cosine distance (lower = more similar),
                "rank"     : 1-indexed rank in results,
            }
    """
    # Apply query prefix for BGE, empty string for MiniLM
    prefix        = model_wrapper["query_prefix"]
    query_vector  = model_wrapper["embedder"].embed_query(prefix + query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested result format
    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text"    : results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "rank"    : i + 1,
        })

    return output


# ── Stats ──────────────────────────────────────────────────────────────────────

def print_collection_stats(collections: Dict[str, chromadb.Collection]) -> None:
    """Prints a summary table of all collections."""
    print("\n[Collection Summary]")
    print(f"  {'Name':<25} {'Chunks':>8}")
    print(f"  {'-'*35}")
    for name, col in collections.items():
        print(f"  {name:<25} {col.count():>8}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build",     action="store_true", help="Build all 4 collections")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing collections")
    parser.add_argument("--stats",     action="store_true", help="Print collection stats only")
    args = parser.parse_args()

    if args.stats:
        cols = load_all_collections()
        print_collection_stats(cols)

    elif args.build:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from ingest    import load_all_pdfs
        from chunking  import get_both_chunk_sets
        from embedding import load_both_models

        print("[Build] Loading documents...")
        docs = load_all_pdfs()

        print("[Build] Chunking...")
        fixed_chunks, recursive_chunks = get_both_chunk_sets(docs)

        print("[Build] Loading embedding models...")
        models = load_both_models(device="cuda")

        print("[Build] Building all 4 collections...")
        build_all_collections(
            fixed_chunks=fixed_chunks,
            recursive_chunks=recursive_chunks,
            models=models,
            overwrite=args.overwrite,
        )

    else:
        parser.print_help()