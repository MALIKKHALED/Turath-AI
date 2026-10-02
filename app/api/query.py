"""POST /query: FAISS retrieve -> ask() -> {answer, citations}."""

from functools import lru_cache

from fastapi import APIRouter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from app.config import settings
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


@lru_cache(maxsize=1)
def get_index() -> FAISS:
    """Lazy singleton: loads bge-m3 + index once, on first request."""
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local("data/processed/faiss_index", embeddings,
                            allow_dangerous_deserialization=True)


def retrieve(question: str, k: int):
    return get_index().similarity_search(question, k=k)


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return ask(req.question, retrieve(req.question, req.k))