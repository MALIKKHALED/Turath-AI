"""Index: chunks -> embeddings -> FAISS, saved to data/processed/faiss_index.

Run: python -m app.rag.index  (first run downloads ~500MB model, one time;
16.7k chunks embed in minutes on CPU, no GPU needed)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

INDEX_DIRNAME = "faiss_index"
SMOKE_QUERY = "غزوة بدر الكبرى"


def load_chunks(proc_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for jf in sorted(proc_dir.glob("*.chunks.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                docs.append(Document(
                    page_content=c["text"],
                    metadata={"id": c["id"], "source": c["source"],
                              "heading": c.get("heading", ""), "index": c["index"]},
                ))
    return docs


def build_index(docs: list[Document], embeddings) -> FAISS:
    return FAISS.from_documents(docs, embeddings)


def main(argv: list[str] | None = None) -> int:
    proc_dir = Path(argv[1] if argv and len(argv) > 1 else "data/processed")
    docs = load_chunks(proc_dir)
    print(f"Loaded {len(docs)} chunks")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    index = build_index(docs, embeddings)
    index.save_local(str(proc_dir / INDEX_DIRNAME))
    print(f"Saved index ({index.index.ntotal} vectors)")
    print(f"--- smoke query: {SMOKE_QUERY} ---")
    for d in index.similarity_search(SMOKE_QUERY, k=3):
        print(f"[{d.metadata['source']} | {d.metadata['heading'][:60]}]")
        print(f"  {d.page_content[:150]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))