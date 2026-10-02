from app.rag.qa import ask, build_prompt


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
    assert out["answer"] == "إجابة ثابتة"
    assert out["citations"] == [
        {"source": "Sira", "heading": "الغزوات", "index": 0},
        {"source": "Raheeq", "heading": "", "index": 3},
    ]