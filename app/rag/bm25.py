"""Sparse side of hybrid retrieval: BM25 over enriched chunk texts.

Whitespace tokens on already-normalized text (no stemmer v1):
BM25 is the complement (exact names/entities), not the lead.
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.split()


def build_bm25(texts: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in texts])


def save_bm25(bm25: BM25Okapi, path: Path) -> None:
    path.write_bytes(pickle.dumps(bm25))


def load_bm25(path: Path) -> BM25Okapi:
    return pickle.loads(path.read_bytes())


def bm25_top_n(bm25: BM25Okapi, query: str, n: int) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:n]