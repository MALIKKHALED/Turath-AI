"""QA chain: retrieve -> Arabic prompt -> OpenRouter LLM -> cited answer.

LLM access is raw httpx (no langchain-openai dep): one POST, full control.
`ask` takes llm + retriever as arguments so tests inject stubs (no network).
"""

from __future__ import annotations

import httpx

import time

from app.config import settings

RETRY_WAITS = (5.0, 20.0)

SYSTEM_AR = (
    "أنت مساعد متخصص في التاريخ الإسلامي والسيرة النبوية. "
    "أجب باللغة العربية الفصحى معتمدا على السياق المرفق فقط. "
    "كل مقتطف يبدأ باسم مصدره بصيغة [المصدر: اسم الكتاب]، "
    "وعند ذكر معلومة اذكر مصدرها بالصيغة نفسها تماما، "
    "مثال: [المصدر: الرحيق المختوم]. "
    "إذا كان السياق غير كاف للإجابة فقل ذلك صراحة ولا تخترع معلومات."
)

SYSTEM_EN = (
    "You are a specialized assistant in Islamic history and the Prophetic biography (Seerah). "
    "Write the ENTIRE answer in Arabic ONLY (Modern Standard Arabic). "
    "Never use English in the answer. "
    "Use ONLY the attached context. Passages are numbered [1] to [N] with their source. "
    "RULES: reformulate in your own clear, natural style as if explaining to a reader; "
    "quote directly ONLY lineages, dates, and sensitive facts; never paste a whole passage. "
    "Cite every fact with its source exactly as [المصدر: book name], "
    "e.g. [المصدر: الرحيق المختوم]. "
    "Structure the answer exactly as: a short headline starting with ##, then ## الإجابة "
    "(direct answer, 2-4 sentences), then ## التفاصيل (organized points). "
    "Do NOT write a sources section; it is appended automatically. "
    "If the context is insufficient, say so explicitly in Arabic "
    "and do not invent information."
)



def _chunk_text(c) -> str:
    if isinstance(c, dict):
        return c.get("raw_text", c["text"])
    return c.metadata.get("raw_text", c.page_content)

def _chunk_meta(c) -> tuple:
    if isinstance(c, dict):
        return c["source"], c.get("heading", ""), c["index"]
    return c.metadata["source"], c.metadata.get("heading", ""), c.metadata["index"]

def build_prompt(question: str, chunks: list, system: str = SYSTEM_AR) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        src, heading, _ = _chunk_meta(c)
        head = f"[{i}] [المصدر: {src}]" + (f" [الباب: {heading}]" if heading else "")
        blocks.append(f"{head}\n{_chunk_text(c)}")
    return f"{system}\n\nالسياق:\n" + "\n\n".join(blocks) + f"\n\nالسؤال: {question}\nالإجابة:"


def _post(prompt: str, model: str) -> str:
    resp = httpx.post(
        f"{settings.openrouter_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"malformed provider payload: {str(data)[:200]}") from exc


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
                code = exc.response.status_code
                if code == 404:
                    break  # retired model ID: next model, no waiting
                if code not in (429, 500, 502, 503):
                    raise  # 401/402/400: real problems, fail loud
            except (ValueError, httpx.DecodingError) as exc:
                last_err = exc  # malformed payload: retry, then rotate


def ask(question: str, chunks: list, llm=call_openrouter, system: str = SYSTEM_AR) -> dict:
    answer = llm(build_prompt(question, chunks, system))
    citations = [{"source": m[0], "heading": m[1], "index": m[2]}
                 for m in (_chunk_meta(c) for c in chunks)]
    names = list(dict.fromkeys(c["source"] for c in citations))
    return {"answer": answer.rstrip() + "\n\n**المصادر:** " + "، ".join(names),
            "citations": citations}
