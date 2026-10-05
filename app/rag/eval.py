"""Retrieval eval harness: fixed probes -> hit-rate@k + per-query breakdown.

Measures the PRODUCTION path (app.api.query.retrieve).
Offline: FAISS only, zero API quota. Run: python -m app.rag.eval
"""

from __future__ import annotations

import sys

from app.api.query import retrieve

K = 5

CASES: list[tuple[str, set[str]]] = [
    ("متى وقعت غزوة بدر الكبرى؟",
     {"Sira-Ibn-Hisham", "الرحيق المختوم", "02-السيرة", "03-الخلفاء الراشدون"}),
    ("ما هو نسب النبي محمد صلى الله عليه وسلم؟", {"Sira-Ibn-Hisham"}),
    ("من هم الخلفاء الراشدون؟", {"03-الخلفاء الراشدون"}),
    ("متى بدأ العهد الأموي؟", {"04-العهد الأموي"}),
    ("كيف قام العهد العثماني؟", {"08-العهد العثماني"}),
    ("متى سقطت الدولة العباسية؟", {"05-الدولة العباسية1", "06-الدولة العباسية2"}),
    ("من هم المماليك؟", {"07-العصر المملوكي"}),
    ("كيف كانت الهجرة إلى المدينة؟",
     {"Sira-Ibn-Hisham", "الرحيق المختوم", "02-السيرة"}),
    ("متى كان فتح مكة؟", {"Sira-Ibn-Hisham", "الرحيق المختوم", "02-السيرة"}),
]


def hit_rate(sources: list[str], expected: set[str], k: int = K) -> float:
    top = sources[:k]
    return sum(1 for s in top if s in expected) / k


def sources_of(docs: list) -> list[str]:
    return [d.metadata["source"] for d in docs]


def run(retrieve_fn=retrieve, k: int = K) -> dict[str, tuple[float, list[str]]]:
    report: dict[str, tuple[float, list[str]]] = {}
    for question, expected in CASES:
        srcs = sources_of(retrieve_fn(question, k))
        report[question] = (round(hit_rate(srcs, expected, k), 2), srcs)
    return report


def main(argv: list[str] | None = None) -> int:
    report = run()
    for question, (score, srcs) in report.items():
        print(f"{score:.2f}  {question}")
        print(f"      got: {srcs}")
    mean = sum(s for s, _ in report.values()) / len(report)
    print(f"MEAN: {mean:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())