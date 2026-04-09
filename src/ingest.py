from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import fitz  # pymupdf
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config_loader import load_config


@dataclass
class Chunk:
    id: str
    doc_id: int
    doc_name: str
    text: str


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract raw text from a PDF using pymupdf.
    """
    doc = fitz.open(pdf_path)
    texts: List[str] = []
    for page in doc:
        page_text = page.get_text("text")
        # Basic cleanup: strip trailing spaces
        texts.append(page_text.strip())
    doc.close()
    # Join pages with a page delimiter
    return "\n\n".join(texts)


def build_corpus(cfg_path: str = "config/base.yaml") -> List[Chunk]:
    cfg = load_config(cfg_path)
    pdf_dir = Path(cfg["data"]["pdf_dir"])
    corpus_cache = Path(cfg["data"]["corpus_cache"])
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir.resolve()}")

    print(f"Found {len(pdf_paths)} PDFs in {pdf_dir}")

    # 1) Extract text
    raw_docs: List[Dict[str, Any]] = []
    for doc_id, pdf_path in enumerate(pdf_paths):
        print(f"Extracting text from: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        raw_docs.append(
            {
                "doc_id": doc_id,
                "doc_name": pdf_path.name,
                "text": text,
            }
        )

    # 2) Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Chunk] = []
    chunk_idx = 0
    for doc in raw_docs:
        doc_id = doc["doc_id"]
        doc_name = doc["doc_name"]
        text = doc["text"]

        doc_chunks = splitter.split_text(text)
        print(f"Doc {doc_name}: {len(doc_chunks)} chunks")

        for local_idx, chunk_text in enumerate(doc_chunks):
            chunk_id = f"doc{doc_id}_chunk{local_idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    text=chunk_text.strip(),
                )
            )
            chunk_idx += 1

    print(f"Total chunks: {len(chunks)}")

    # 3) Save to cache
    corpus_cache.parent.mkdir(parents=True, exist_ok=True)
    with corpus_cache.open("wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved corpus to {corpus_cache.resolve()}")
    return chunks


if __name__ == "__main__":
    build_corpus()
