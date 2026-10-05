from types import SimpleNamespace

from app.rag.eval import CASES, hit_rate, run, sources_of


def _doc(source):
    return SimpleNamespace(metadata={"source": source})


def test_hit_rate_math():
    assert hit_rate(["a", "b", "a", "c", "a"], {"a"}, 5) == 0.6
    assert hit_rate([], {"a"}) == 0.0

def test_run_scores_each_case_through_retrieve_fn():
    scores = run(retrieve_fn=lambda q, k: [_doc("Sira-Ibn-Hisham")] * k)
    assert len(scores) == len(CASES)
    assert scores["متى وقعت غزوة بدر الكبرى؟"][0] == 1.0
    assert scores["من هم الخلفاء الراشدون؟"][0] == 0.0


def test_sources_of_extracts_source_list():
    docs = [_doc("a"), _doc("b")]
    assert sources_of(docs) == ["a", "b"]