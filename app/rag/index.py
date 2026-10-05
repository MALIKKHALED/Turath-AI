"""Index: enriched chunks -> versioned FAISS dir + bm25.pkl.

Embedding text = [source | heading] + raw (deterministic contextualization).
Display/LLM text stays raw via metadata["raw_text"] — enrichment never
leaks into answers. Run: python -m app.rag.index (ONE heavy run, GPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.rag.bm25 import build_bm25, save_bm25

SMOKE_QUERY = "غزوة بدر الكبرى"


def enriched_text(source: str, heading: str, text: str) -> str:
    head = f"[{source}" + (f" | {heading}]" if heading else "]")
    return f"{head}\n{text}"


def get_embeddings():
    """Single construction site for the embedding model (device switch)."""
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.device},
        encode_kwargs={"normalize_embeddings": True},
    )
    if settings.use_fp16:
        emb._client.half()  # halve VRAM post-load; if this errors, set use_fp16=false
    return emb


def load_chunks(proc_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for jf in sorted(proc_dir.glob("*.chunks.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                heading = c.get("heading", "")
                docs.append(Document(
                    page_content=enriched_text(c["source"], heading, c["text"]),
                    metadata={"id": c["id"], "source": c["source"],
                              "heading": heading, "index": c["index"],
                              "raw_text": c["text"]},
                ))
    return docs


def build_index(docs: list[Document], embeddings) -> FAISS:
    return FAISS.from_documents(docs, embeddings)


def embed_in_batches(texts: list[str], embeddings, batch_size: int) -> list:
    out: list = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        out.extend(embeddings.embed_documents(texts[i:i + batch_size]))
        print(f"embedded batch {i // batch_size + 1}/{total} ({len(out)}/{len(texts)})")
    return out


def main(argv: list[str] | None = None) -> int:
    proc_dir = Path(argv[1] if argv and len(argv) > 1 else "data/processed")
    out_dir = Path(settings.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_chunks(proc_dir)
    print(f"Loaded {len(docs)} chunks")
    embeddings = get_embeddings()
    vectors = embed_in_batches([d.page_content for d in docs],
                               embeddings, settings.embed_batch_size)
    index = FAISS.from_embeddings(
        [(d.page_content, v) for d, v in zip(docs, vectors)],
        embeddings, metadatas=[d.metadata for d in docs])
    index.save_local(str(out_dir))
    save_bm25(build_bm25([d.page_content for d in docs]), out_dir / "bm25.pkl")
    print(f"Saved index ({index.index.ntotal} vectors) + bm25 -> {out_dir}")
    for d in index.similarity_search(SMOKE_QUERY, k=3):
        print(f"[{d.metadata['source']}] {d.metadata['raw_text'][:120]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))