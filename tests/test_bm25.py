from app.rag.bm25 import bm25_top_n, build_bm25, load_bm25, save_bm25


def test_exact_term_recall():
    texts = ["القط يجلس على السجادة", "بويع معاوية بالخلافة", "السماء زرقاء"]
    assert bm25_top_n(build_bm25(texts), "معاوية الخلافة", 1) == [1]

def test_save_load_roundtrip(tmp_path):
    texts = ["نص أول", "نص ثان", "كلام ثالث", "خبر رابع"]
    b = build_bm25(texts)
    p = tmp_path / "bm25.pkl"
    save_bm25(b, p)
    assert bm25_top_n(load_bm25(p), "ثان", 1) == [1]