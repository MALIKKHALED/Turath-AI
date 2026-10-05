"""QA chain: retrieve -> Arabic prompt -> OpenRouter LLM -> cited answer.

LLM access is raw httpx (no langchain-openai dep): one POST, full control.
`ask` takes llm + retriever as arguments so tests inject stubs (no network).
"""

from __future__ import annotations

import httpx

import time

from app.config import settings

RETRY_WAITS = (5.0, 20.0)

SYSTEM = (
    "أنت مساعد متخصص في التاريخ الإسلامي والسيرة النبوية. "
    "أجب باللغة العربية الفصحى معتمدا على السياق المرفق فقط. "
    "كل مقتطف يبدأ باسم مصدره بصيغة [المصدر: اسم الكتاب]، "
    "وعند ذكر معلومة اذكر مصدرها بالصيغة نفسها تماما، "
    "مثال: [المصدر: الرحيق المختوم]. "
    "إذا كان السياق غير كاف للإجابة فقل ذلك صراحة ولا تخترع معلومات."
)



def _chunk_text(c) -> str:
    if isinstance(c, dict):
        return c.get("raw_text", c["text"])
    return c.metadata.get("raw_text", c.page_content)

def _chunk_meta(c) -> tuple:
    if isinstance(c, dict):
        return c["source"], c.get("heading", ""), c["index"]
    return c.metadata["source"], c.metadata.get("heading", ""), c.metadata["index"]

def build_prompt(question: str, chunks: list) -> str:
    blocks = []
    for c in chunks:
        src, heading, _ = _chunk_meta(c)
        head = f"[المصدر: {src}]" + (f" [الباب: {heading}]" if heading else "")
        blocks.append(f"{head}\n{_chunk_text(c)}")
    return f"{SYSTEM}\n\nالسياق:\n" + "\n\n".join(blocks) + f"\n\nالسؤال: {question}\nالإجابة:"


def _post(prompt: str, model: str) -> str:
    resp = httpx.post(
        f"{settings.openrouter_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt: str) -> str:
    models = [settings.openrouter_model] + settings.openrouter_fallbacks.split(",")
    last_err: Exception | None = None
    for model in models:
        for wait in (0.0, *RETRY_WAITS):
            if wait:
                time.sleep(wait)
            try:
                ans = _post(prompt, model.strip())
                print(f"[model: {model.strip()}]")
                return ans
            except httpx.HTTPStatusError as exc:
                last_err = exc
                if exc.response.status_code != 429:
                    raise
    raise last_err  # type: ignore[misc]


def ask(question: str, chunks: list, llm=call_openrouter) -> dict:
    answer = llm(build_prompt(question, chunks))
    citations = [{"source": m[0], "heading": m[1], "index": m[2]}
                 for m in (_chunk_meta(c) for c in chunks)]
    return {"answer": answer, "citations": citations}

