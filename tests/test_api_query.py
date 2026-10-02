from fastapi.testclient import TestClient

import app.api.query as qmod
from app.main import app

client = TestClient(app)


def _docs():
    return [{"text": "وقعت بدر في الثانية للهجرة.", "source": "Sira",
             "heading": "الغزوات", "index": 0}]


def test_query_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr(qmod, "retrieve", lambda question, k: _docs())
    monkeypatch.setattr(qmod, "ask",
                        lambda question, chunks, llm=None: {
                            "answer": "في الثانية.",
                            "citations": [{"source": "Sira", "heading": "الغزوات",
                                           "index": 0}]})
    resp = client.post("/query", json={"question": "متى بدر؟"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "في الثانية."
    assert body["citations"][0]["source"] == "Sira"


def test_query_rejects_empty_question():
    assert client.post("/query", json={"question": ""}).status_code == 422