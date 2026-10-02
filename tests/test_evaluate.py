import zipfile

from app.rag.evaluate import evaluate_file, load_epub


def test_clean_arabic_txt_accepts(tmp_path):
    p = tmp_path / "clean.txt"
    p.write_text("الحمد لله رب العالمين، والصلاة والسلام على سيد المرسلين. " * 20,
                 encoding="utf-8")
    assert evaluate_file(p).verdict == "ACCEPT"


def test_latin_file_reviews(tmp_path):
    p = tmp_path / "latin.txt"
    p.write_text("Hello world, this is English text. " * 20, encoding="utf-8")
    assert evaluate_file(p).verdict == "REVIEW"


def test_epub_extraction(tmp_path):
    epub = tmp_path / "mini.epub"
    xhtml = "<html><body><p>بسم الله الرحمن الرحيم</p></body></html>"
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr("OEBPS/ch1.xhtml", xhtml)
    assert "بسم الله الرحمن الرحيم" in load_epub(epub)