"""Source quality gate: score raw book files BEFORE ingestion.

Usage:  python -m app.rag.evaluate [data/raw]
Prints ACCEPT/REVIEW per file + sample lines for human judgment.
Stdlib only — no new dependencies.
"""

from __future__ import annotations

import html
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ARABIC_RE = re.compile(r"[ء-غف-ي]")
ALLOWED_PUNCT = set(".,;:!?()\"'«»—–-ـ٪؟،؛٫٬×÷ \t\n")


def _is_junk(char: str) -> bool:
    return not (char.isalnum() or char in ALLOWED_PUNCT or ARABIC_RE.match(char))


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_epub(path: Path) -> str:
    """Extract text from EPUB with stdlib zipfile (EPUB = zipped XHTML)."""
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.lower().endswith((".xhtml", ".xhtm", ".html")))
        for name in names:
            raw = zf.read(name).decode("utf-8", errors="ignore")
            raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S)
            raw = re.sub(r"<[^>]+>", " ", raw)
            parts.append(html.unescape(raw))
    return "\n".join(parts)


def load_source(path: Path) -> str:
    if path.suffix == ".epub":
        return load_epub(path)
    return load_txt(path)


@dataclass
class Report:
    name: str
    chars: int
    arabic_ratio: float
    junk_ratio: float
    farsi_yeh: int
    farsi_kaf: int
    verdict: str
    sample: list[str]


def evaluate_file(path: Path) -> Report:
    text = load_source(path)
    chars = len(text)
    arabic = len(ARABIC_RE.findall(text))
    junk = sum(1 for c in text if _is_junk(c))
    sample = [ln.strip()[:120] for ln in text.splitlines()
              if ln.strip()][:8]
    if chars == 0:
        verdict = "REVIEW"
    elif arabic / chars >= 0.50 and junk / chars <= 0.05:
        verdict = "ACCEPT"
    else:
        verdict = "REVIEW"
    return Report(
        name=path.name,
        chars=chars,
        arabic_ratio=round(arabic / max(chars, 1), 3),
        junk_ratio=round(junk / max(chars, 1), 3),
        farsi_yeh=text.count("ی"),
        farsi_kaf=text.count("ک"),
        verdict=verdict,
        sample=sample,
    )


def main(argv: list[str] | None = None) -> int:
    raw_dir = Path(argv[1] if argv and len(argv) > 1 else "data/raw")
    files = sorted(p for p in raw_dir.iterdir()
                   if p.suffix in (".txt", ".epub") and p.name != ".gitkeep")
    if not files:
        print(f"No source files in {raw_dir}")
        return 1
    for path in files:
        try:
            r = evaluate_file(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{path.name}: ERROR {exc}")
            continue
        print(f"{r.verdict:7} ar={r.arabic_ratio:.3f} junk={r.junk_ratio:.3f} "
              f"yeh={r.farsi_yeh} kaf={r.farsi_kaf} chars={r.chars}  {r.name}")
        for line in r.sample[:3]:
            print(f"         | {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))