from types import SimpleNamespace

import app.api.query as qmod
from app.api.query import _rrf_fuse


def _doc(cid):
    return SimpleNamespace(page_content=cid, metadata={
        "id": cid, "source": "s", "heading": "", "index": 0, "raw_text": cid})


def test_rrf_rewards_both_lists():
    assert _rrf_fuse(["a", "b"], ["b", "c"], k=2, rrf_k=60) == ["b", "a"]


def test_retrieve_hybrid_fuses_dense_and_sparse(monkeypatch):
    docs = {c: _doc(c) for c in "abcd"}
    monkeypatch.setattr(qmod, "get_index", lambda: SimpleNamespace(
        similarity_search=lambda q, k: [docs["a"], docs["b"]]))
    monkeypatch.setattr(qmod, "get_corpus",
                        lambda: [docs["c"], docs["d"], docs["a"], docs["b"]])
    monkeypatch.setattr(qmod, "get_bm25", lambda: SimpleNamespace())
    monkeypatch.setattr(qmod, "bm25_top_n", lambda b, q, n: [0, 1])
    monkeypatch.setattr(qmod.settings, "use_bm25", True)
    assert [d.metadata["id"] for d in qmod.retrieve("q", 2)] == ["a", "c"]


def test_retrieve_dense_only_when_bm25_off(monkeypatch):
    monkeypatch.setattr(qmod.settings, "use_bm25", False)
    monkeypatch.setattr(qmod, "get_index", lambda: SimpleNamespace(
        similarity_search=lambda q, k: ["D"]))
    assert qmod.retrieve("q", 3) == ["D"]