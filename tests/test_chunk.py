from app.rag.chunk import chunk_file, pack_sentences


def test_heading_split_and_metadata(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("### | النسب الشريف\nمحمد بن عبد الله بن عبد المطلب.\n### | المولد\nولد عام الفيل.",
                 encoding="utf-8")
    chunks = chunk_file(p)
    assert len(chunks) == 2
    assert chunks[0].heading == "النسب الشريف"
    assert chunks[0].source == "v" and chunks[0].id == "v:0000"


def test_packing_respects_target_with_overlap():
    sents = ["جملة أولى قصيرة.", "جملة ثانية قصيرة.", "جملة ثالثة " + "طويلة " * 78]
    chunks = pack_sentences(sents, "s", "")
    assert all(len(c.text) <= 600 for c in chunks)
    assert len(chunks) == 2
    assert chunks[1].text.startswith("جملة ثانية قصيرة.")  # overlap carried