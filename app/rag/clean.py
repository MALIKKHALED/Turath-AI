"""Cleaner: raw book text -> retrieval-ready text in data/processed.

Rules: normalize orthography (keep text human-quotable for citations),
strip furniture, quarantine mangled-OCR lines. Diacritics stripped
(users query without them; embeddings match undiacritized better).
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import re

from app.rag.evaluate import ARABIC_RE, load_source


ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
                          "ی": "ي", "ک": "ك", "ى": "ي"})
META_PREFIX = "#META#"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ALEF_MAP)
    return "".join(c for c in text if c not in DIACRITICS_RE_set())


def DIACRITICS_RE_set() -> set[str]:
    return set("ًٌٍَُِّْٰـ")


def junk_ratio(line: str) -> float:
    if not line:
        return 0.0
    allowed = set(".,;:!?()\"'«»—–-ـ٪؟،؛٫٬×÷0123456789 ")
    bad = sum(1 for c in line
              if not (c.isalnum() or c in allowed or ARABIC_RE.match(c)))
    return bad / len(line)


def is_furniture(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith(META_PREFIX) or s.startswith("######"):
        return True
    arabic = len(ARABIC_RE.findall(s))
    return arabic / len(s) < 0.30  # page numbers, OCR header junk, Latin scraps

def _strip_openiti(line: str) -> str:
    """Strip OpenITI mARkdown line prefixes; keep ### headings as section markers."""
    s = line.strip()
    if s.startswith("###"):
        return s  # heading marker — chunker splits on these
    if s.startswith("~~"):
        s = s[2:].strip()
    elif s.startswith("# "):
        s = s[2:].strip()
    return re.sub(r"PageV\d+P\d+", " ", s).strip()


def clean_text(text: str) -> tuple[str, int, int]:
    """Return (cleaned, dropped_furniture, quarantined_verses)."""
    kept: list[str] = []
    dropped = quarantined = 0
    for raw in text.splitlines():
        line = normalize(_strip_openiti(raw)).strip()
        if is_furniture(line):
            dropped += 1
        elif junk_ratio(line) > 0.40:
            quarantined += 1
        else:
            kept.append(line)
    return "\n".join(kept), dropped, quarantined


@dataclass
class CleanReport:
    name: str
    before: int
    after: int
    dropped: int
    quarantined: int


def pick_sources(raw_dir: Path) -> list[Path]:
    """EPUB wins when both EPUB + txt exist for the same volume (txt = backup)."""
    files = {p.stem: p for p in sorted(raw_dir.iterdir()) if p.suffix == ".txt"}
    for p in sorted(raw_dir.iterdir()):
        if p.suffix == ".epub" and p.stem in files:
            files[p.stem] = p  # EPUB overrides its txt twin
        elif p.suffix == ".epub":
            files[p.stem] = p
    return [files[k] for k in sorted(files) if not k.startswith(".gitkeep")]


def main(argv: list[str] | None = None) -> int:
    raw_dir = Path(argv[1] if argv and len(argv) > 1 else "data/raw")
    out_dir = Path(argv[2] if argv and len(argv) > 2 else "data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in pick_sources(raw_dir):
        if src.name == ".gitkeep":
            continue
        cleaned, dropped, quarantined = clean_text(load_source(src))
        out = out_dir / f"{src.stem}.txt"
        out.write_text(cleaned, encoding="utf-8")
        print(f"{src.name} -> {out.name}: {len(load_source(src))} -> "
              f"{len(cleaned)} chars (furniture={dropped}, quarantined={quarantined})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))