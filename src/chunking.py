"""
chunking.py
-----------
Implements two chunking strategies for benchmarking:

  Strategy 1 — Fixed Size Chunking
      Splits text every CHUNK_SIZE characters with CHUNK_OVERLAP overlap.
      Fast and simple but can cut mid-sentence.

  Strategy 2 — Recursive Semantic Chunking
      Splits on natural boundaries (\n\n → \n → . → space) in order.
      Respects paragraph and sentence structure.
      Falls back to next separator only when chunk is still too large.

Both use the same chunk_size and overlap so the only variable is
HOW the split happens — keeping the experiment clean and fair.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from tqdm import tqdm




# ── Configuration ──────────────────────────────────────────────────────────────

CHUNK_SIZE    = 512   # characters per chunk (same for both strategies)
CHUNK_OVERLAP = 50    # characters of overlap between consecutive chunks


# ── Strategy 1: Fixed Size Chunking ───────────────────────────────────────────

def fixed_size_chunking(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits each Document into fixed-size character chunks.

    How it works:
      - Uses a single separator: "" (no separator preference)
      - Cuts every chunk_size characters regardless of word/sentence boundaries
      - Adds chunk_overlap characters of overlap between consecutive chunks
        so context is not lost at boundaries

    Args:
        documents   : List of Document objects from ingest.py
        chunk_size  : Number of characters per chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunked Document objects with updated metadata.
    """
    splitter = CharacterTextSplitter(
        separator="\n",          # split at newlines first, then hard-cut
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    for doc in tqdm(documents, desc="Fixed-size chunking"):
        split_docs = splitter.split_documents([doc])

        # Add chunk-level metadata so we can trace each chunk back to its source
        for i, chunk in enumerate(split_docs):
            chunk.metadata.update({
                "chunk_index"   : i,
                "chunk_strategy": "fixed",
                "chunk_size"    : chunk_size,
                "chunk_overlap" : chunk_overlap,
                "chunk_length"  : len(chunk.page_content),
            })
            chunks.append(chunk)

    return chunks


# ── Strategy 2: Recursive Semantic Chunking ────────────────────────────────────

def recursive_chunking(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits each Document using recursive character splitting on semantic boundaries.

    How it works:
      - Tries separators in this order: ["\n\n", "\n", ". ", " ", ""]
      - Tries to split on \n\n (paragraph breaks) first
      - If a chunk is still too large, falls back to \n (line breaks)
      - Then falls back to ". " (sentence boundaries)
      - Then to " " (word boundaries)
      - Finally hard-cuts at the character level as a last resort
      - This means chunks are as semantically coherent as possible

    Args:
        documents   : List of Document objects from ingest.py
        chunk_size  : Number of characters per chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunked Document objects with updated metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",   # paragraph boundary  — preferred
            "\n",     # line boundary
            ". ",     # sentence boundary
            " ",      # word boundary
            "",       # hard cut            — last resort
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    for doc in tqdm(documents, desc="Recursive chunking"):
        split_docs = splitter.split_documents([doc])

        for i, chunk in enumerate(split_docs):
            chunk.metadata.update({
                "chunk_index"   : i,
                "chunk_strategy": "recursive",
                "chunk_size"    : chunk_size,
                "chunk_overlap" : chunk_overlap,
                "chunk_length"  : len(chunk.page_content),
            })
            chunks.append(chunk)

    return chunks


# ── Chunking Stats ─────────────────────────────────────────────────────────────

def print_chunk_stats(chunks: List[Document], strategy_name: str) -> None:
    """
    Prints statistics about a set of chunks — useful for comparing strategies.

    Shows:
      - Total number of chunks produced
      - Min, max, average chunk length in characters
      - Distribution of chunk lengths (short / medium / long)
    """
    lengths = [len(c.page_content) for c in chunks]

    short  = sum(1 for l in lengths if l < 200)
    medium = sum(1 for l in lengths if 200 <= l < 400)
    long_  = sum(1 for l in lengths if l >= 400)

    print(f"\n[Chunk Stats — {strategy_name}]")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Min length   : {min(lengths)} chars")
    print(f"  Max length   : {max(lengths)} chars")
    print(f"  Avg length   : {sum(lengths) // len(lengths)} chars")
    print(f"  Distribution :")
    print(f"    Short  (<200 chars) : {short}  ({100*short//len(chunks)}%)")
    print(f"    Medium (200–400)    : {medium} ({100*medium//len(chunks)}%)")
    print(f"    Long   (>400 chars) : {long_}  ({100*long_//len(chunks)}%)")


def get_both_chunk_sets(
    documents: List[Document],
) -> Tuple[List[Document], List[Document]]:
    """
    Convenience function: runs both chunking strategies on the same documents
    and returns both chunk lists.

    Returns:
        (fixed_chunks, recursive_chunks)
    """
    print("\n[Chunking] Running Strategy 1 — Fixed Size...")
    fixed_chunks = fixed_size_chunking(documents)
    print_chunk_stats(fixed_chunks, "Fixed Size")

    print("\n[Chunking] Running Strategy 2 — Recursive Semantic...")
    recursive_chunks = recursive_chunking(documents)
    print_chunk_stats(recursive_chunks, "Recursive Semantic")

    return fixed_chunks, recursive_chunks


# ── Entry Point (for testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Import ingest here only for standalone testing
    from ingest import load_all_pdfs

    print("Loading documents...")
    docs = load_all_pdfs()

    fixed_chunks, recursive_chunks = get_both_chunk_sets(docs)

    # Show a side-by-side example of how the same page gets chunked differently
    print("\n[Example — Same source page, different chunking]")
    source_page = docs[0].page_content

    print(f"\nOriginal page ({len(source_page)} chars):")
    print(source_page[:300] + "...\n")

    print("Fixed chunk 0:")
    print(fixed_chunks[0].page_content[:300])

    print("\nRecursive chunk 0:")
    print(recursive_chunks[0].page_content[:300])