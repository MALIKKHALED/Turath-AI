from app.rag.clean import clean_text, is_furniture, normalize


def test_normalize_unifies_variants():
    assert normalize("یی کك أإآٱ ى ـًٌ") == "يي كك اااا ي "


def test_furniture_dropped():
    assert is_furniture("#META# 020.BookTITLE :: foo")
    assert is_furniture("12345")
    assert is_furniture("")
    assert not is_furniture("الحمد لله رب العالمين")


def test_quarantine_mangled_verse():
    vocalized = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"  # clean but vocalized: survives
    mangled = "بسم الله ©€§¶#©€§"  # symbol salad: quarantined
    text = f"سيرة النبي محمد\n{vocalized}\n{mangled}\nالهجرة الي المدينة"
    cleaned, dropped, quarantined = clean_text(text)
    assert quarantined == 1
    assert "بسم الله الرحمن الرحيم" in cleaned
    assert "©" not in cleaned


def test_clean_end_to_end_counts():
    cleaned, dropped, quarantined = clean_text("بسم الله\n\n42\nالحمد لله")
    assert dropped == 2 and quarantined == 0 and "الحمد لله" in cleaned