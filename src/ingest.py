"""
ingest.py
---------
Loads all PDF files from data/raw/, extracts text page-by-page,
cleans it, and returns a list of LangChain Document objects.

Each Document contains:
  - page_content : cleaned text of one page
  - metadata     : { "source": "filename.pdf", "page": <int> }
"""

import os
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from a PDF page.

    Steps:
      1. Replace non-breaking spaces and other unicode whitespace with a normal space
      2. Collapse multiple spaces into one
      3. Collapse more than 2 consecutive newlines into exactly 2 (preserve paragraphs)
      4. Strip leading/trailing whitespace
    """
    # Step 1: Normalize unicode whitespace characters
    text = text.replace("\xa0", " ")       # non-breaking space
    text = text.replace("\t", " ")         # tab → space

    # Step 2: Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Step 3: Collapse excessive newlines (keep paragraph breaks as \n\n)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 4: Strip edges
    text = text.strip()

    return text


def is_useful_page(text: str, min_chars: int = 100) -> bool:
    """
    Returns True if the page has enough meaningful content to be worth keeping.

    We skip:
      - Blank pages
      - Pages with only headers/footers (very short text)
      - Pages that are just page numbers or copyright notices
    """
    return len(text) >= min_chars


# ── Core Ingestion ─────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> List[Document]:
    """
    Loads a single PDF and returns one Document per useful page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects, one per page (skipping blank/short pages).
    """
    documents = []

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  [WARNING] Could not read {pdf_path.name}: {e}")
        return documents

    for page_num, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            print(f"  [WARNING] Could not extract page {page_num} from {pdf_path.name}: {e}")
            continue

        cleaned = clean_text(raw_text)

        # Skip pages that don't have enough content
        if not is_useful_page(cleaned):
            continue

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": pdf_path.name,       # e.g. "mckinsey_fmcg_ai_2023.pdf"
                "page": page_num + 1,           # 1-indexed page number
                "total_pages": len(reader.pages),
                "file_path": str(pdf_path),
            }
        )
        documents.append(doc)

    return documents


def load_all_pdfs(data_dir: Path = RAW_DATA_DIR) -> List[Document]:
    """
    Scans data_dir for all .pdf files and loads them all.

    Args:
        data_dir: Directory containing PDF files. Defaults to data/raw/.

    Returns:
        A flat list of all Document objects across all PDFs.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please create it and add your PDF files:\n"
            f"  mkdir -p {data_dir}"
        )

    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {data_dir}\n"
            f"Please add PDFs to that folder before running ingestion."
        )

    print(f"\n[Ingest] Found {len(pdf_files)} PDF(s) in {data_dir}")
    print("-" * 50)

    all_documents = []

    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        docs = load_pdf(pdf_path)
        print(f"  ✓ {pdf_path.name}: {len(docs)} pages loaded")
        all_documents.extend(docs)

    print("-" * 50)
    print(f"[Ingest] Total pages loaded: {len(all_documents)}")
    print(f"[Ingest] Sources: {len(pdf_files)} PDF(s)\n")

    return all_documents


# ── Quick Stats ────────────────────────────────────────────────────────────────

def print_corpus_stats(documents: List[Document]) -> None:
    """
    Prints a quick summary of the loaded corpus — useful for sanity checking.
    """
    total_chars = sum(len(d.page_content) for d in documents)
    avg_chars   = total_chars // len(documents) if documents else 0
    sources     = sorted(set(d.metadata["source"] for d in documents))

    print("\n[Corpus Stats]")
    print(f"  Total documents (pages) : {len(documents)}")
    print(f"  Total characters        : {total_chars:,}")
    print(f"  Avg chars per page      : {avg_chars:,}")
    print(f"  Unique source files     : {len(sources)}")
    for src in sources:
        count = sum(1 for d in documents if d.metadata["source"] == src)
        print(f"    - {src}: {count} pages")


# ── Entry Point (for testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_all_pdfs()
    print_corpus_stats(docs)

    # Preview first document
    print("\n[Preview — First Document]")
    print(f"  Source  : {docs[0].metadata['source']}")
    print(f"  Page    : {docs[0].metadata['page']}")
    print(f"  Content : {docs[0].page_content[:300]}...")