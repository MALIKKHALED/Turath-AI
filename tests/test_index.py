from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.index import build_index, load_chunks, enriched_text


class _StubEmbeddings(Embeddings):
    """Length-based vectors: no model download, no network in tests."""

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def _doc(text):
    return Document(page_content=text,
                    metadata={"id": "t:0", "source": "t", "heading": "", "index": 0})


def test_build_and_retrieve_nearest():
    docs = [_doc("aaaa"), _doc("bbbbbbbbbbbb")]
    index = build_index(docs, _StubEmbeddings())
    hits = index.similarity_search("ccc", k=1)
    assert hits[0].page_content == "aaaa"  # |4-3| < |12-3| in vector space


def test_load_chunks_reads_jsonl(tmp_path):
    f = tmp_path / "v.chunks.jsonl"
    f.write_text('{"id": "v:0000", "text": "نص تجريبي", "source": "v", '
                 '"heading": "باب", "index": 0}\n', encoding="utf-8")
    docs = load_chunks(tmp_path)
    assert len(docs) == 1 and docs[0].metadata["heading"] == "باب"


def test_enriched_prefix_format():
    assert enriched_text("S", "باب", "نص") == "[S | باب]\nنص"
    assert enriched_text("S", "", "نص") == "[S]\nنص"