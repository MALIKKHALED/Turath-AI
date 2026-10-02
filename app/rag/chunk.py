"""Chunker: cleaned volume -> sentence-packed passages (JSONL).

Sections split on ### headings (Ibn Hisham); marker-less files (EPUBs)
are one section. Sentences greedily packed to ~500 chars, last sentence
carried as overlap. Output: data/processed/<stem>.chunks.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

TARGET = 500
HARD_CAP = 600
SENT_SPLIT = re.compile(r"(?<=[.!؟])\s+")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    heading: str
    index: int


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def pack_sentences(sentences: list[str], source: str, heading: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            i = len(chunks)
            chunks.append(Chunk(id=f"{source}:{i:04d}", text=" ".join(buf),
                                source=source, heading=heading, index=i))
            buf.clear()

    for sent in sentences:
        if len(sent) > HARD_CAP:  # pathological single sentence: hard-cut
            flush()
            for k in range(0, len(sent), TARGET):
                buf.append(sent[k:k + TARGET])
                flush()
            continue
        if buf and len(" ".join(buf)) + 1 + len(sent) > TARGET:
            overlap = [buf[-1]] if len(buf) > 1 else []
            flush()
            buf.extend(overlap)
        buf.append(sent)
    flush()
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    source = path.stem
    sections: list[tuple[str, list[str]]] = []
    heading, paras = "", []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("###"):
            if paras:
                sections.append((heading, paras))
            heading = re.sub(r"^#+\s*\|*\s*", "", line).strip()
            paras = []
        elif line.strip():
            paras.append(line.strip())
    if paras:
        sections.append((heading, paras))
    chunks: list[Chunk] = []
    for h, ps in sections or [("", [])]:
        chunks.extend(pack_sentences(split_sentences(" ".join(ps)), source, h))
    # re-index globally
    for i, c in enumerate(chunks):
        c.index, c.id = i, f"{source}:{i:04d}"
    return chunks


def main(argv: list[str] | None = None) -> int:
    proc_dir = Path(argv[1] if argv and len(argv) > 1 else "data/processed")
    total = 0
    for path in sorted(proc_dir.glob("*.txt")):
        chunks = chunk_file(path)
        out = proc_dir / f"{path.stem}.chunks.jsonl"
        out.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False)
                                 for c in chunks), encoding="utf-8")
        avg = sum(len(c.text) for c in chunks) // max(len(chunks), 1)
        print(f"{path.name}: {len(chunks)} chunks, avg {avg} chars -> {out.name}")
        total += len(chunks)
    print(f"TOTAL: {total} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))