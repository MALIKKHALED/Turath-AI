"""POST /query: hybrid retrieve (FAISS + BM25/RRF) -> ask() -> {answer, citations}."""

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.bm25 import bm25_top_n, load_bm25
from app.rag.index import get_embeddings, load_chunks
from app.rag.qa import ask

router = APIRouter()


class Citation(BaseModel):
    source: str
    heading: str = ""
    index: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=settings.retriever_k, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


RRF_DEPTH = 20


@lru_cache(maxsize=1)
def get_index() -> FAISS:
    """Lazy singleton over the versioned index dir."""
    return FAISS.load_local(settings.index_dir, get_embeddings(),
                            allow_dangerous_deserialization=True)


@lru_cache(maxsize=1)
def get_bm25():
    return load_bm25(Path(settings.index_dir) / "bm25.pkl")


@lru_cache(maxsize=1)
def get_corpus() -> list:
    return load_chunks(Path(settings.processed_dir))


def _rrf_fuse(dense_ids: list[str], sparse_ids: list[str],
              k: int, rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def retrieve(question: str, k: int):
    if not settings.use_bm25:
        return get_index().similarity_search(question, k=k)  # proven 0.80 path
    dense = get_index().similarity_search(question, k=RRF_DEPTH)
    corpus = get_corpus()
    sparse_docs = [corpus[i] for i in bm25_top_n(get_bm25(), question, RRF_DEPTH)]
    fused = _rrf_fuse([d.metadata["id"] for d in dense],
                      [d.metadata["id"] for d in sparse_docs], k, settings.rrf_k)
    by_id = {d.metadata["id"]: d for d in dense + sparse_docs}
    return [by_id[cid] for cid in fused]



@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return ask(req.question, retrieve(req.question, req.k))