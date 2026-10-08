from app.rag.qa import SYSTEM_EN, build_prompt, ask


def _chunks():
    return [
        {"text": "وقعت غزوة بدر في السنة الثانية للهجرة.", "source": "Sira",
         "heading": "الغزوات", "index": 0},
        {"text": "قاد النبي المسلمين في بدر.", "source": "Raheeq",
         "heading": "", "index": 3},
    ]


def test_prompt_contains_question_and_sourced_context():
    p = build_prompt("متى وقعت بدر؟", _chunks())
    assert "متى وقعت بدر؟" in p
    assert "[المصدر: Sira]" in p and "وقعت غزوة بدر" in p


def test_ask_returns_answer_and_citations():
    out = ask("متى وقعت بدر؟", _chunks(), llm=lambda prompt: "إجابة ثابتة")
    assert out["answer"].startswith("إجابة ثابتة\n\n## المصادر")
    assert out["answer"].endswith("[1] Sira — الباب: الغزوات\n[2] Raheeq")
    assert out["citations"] == [
        {"source": "Sira", "heading": "الغزوات", "index": 0},
        {"source": "Raheeq", "heading": "", "index": 3},
    ]



def test_en_prompt_carries_arabic_only_directive():
    p = build_prompt("س؟", _chunks(), system=SYSTEM_EN)
    assert "Arabic ONLY" in p and "متى وقعت بدر؟" not in p and "س؟" in p   


def test_v3_numbered_blocks_and_server_side_refs():
    out = ask("س؟", _chunks(), llm=lambda prompt: "متن الإجابة", system=SYSTEM_EN)
    assert out["answer"].endswith("## المصادر\n[1] Sira — الباب: الغزوات\n[2] Raheeq")
    assert "[1] [المصدر: Sira]" in build_prompt("س؟", _chunks(), system=SYSTEM_EN)