# Turath AI (تراث) — The Intelligent Islamic History Assistant

Arabic-first question-answering over Islamic history and Seerah books.
Ask in Arabic, get precise answers with source citations — or an honest
refusal when the books don't cover the question.

Built with FastAPI + LangChain + FAISS + bge-m3. Generation via OpenRouter.

## Features

- Natural-language Arabic QA over 10 book sources (Seerah + history to the Ottoman era)
- Deterministic source citations on every answer; refusal (no invention) out of corpus
- Measured retrieval: 0.96 hit-rate@5 on a fixed 9-probe eval harness
- Resilient free-tier LLM client (retry with backoff + automatic model rotation)
- Arabic RTL chat UI with Islamic theme over the same `/query` API
- Full test suite (pytest); Docker support for the API

## Library (v1)

| Source | Volumes | Origin |
|---|---|---|
| السيرة النبوية — ابن هشام (d. 213H) | complete | OpenITI `.completed` text |
| الرحيق المختوم — المباركفوري | complete | archive.org EPUB |
| التاريخ الإسلامي — محمود شاكر السوري | vols 1–8 (pre-Islam → Ottoman) | archive.org EPUBs |

Later volumes (9–22) can be added as an upgrade without code changes.

## Architecture

```
data/raw  ->  evaluate (quality gate)  ->  clean (normalize, strip, quarantine)
          ->  chunk (500 chars, headings, metadata)  ->  *.chunks.jsonl
          ->  index (bge-m3 enriched embeddings -> FAISS + BM25 parked)
                        |
/query -> retrieve (top-5) -> prompt v3 EN -> OpenRouter LLM -> {answer, citations}
   |                                                              |
 health                                                     Streamlit UI (HTTP)
```

## Quickstart (local)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e .
copy .env.example .env   # then fill HF_TOKEN + OPENROUTER_API_KEY
```

Always use `.\.venv\Scripts\python.exe -m pip` (never bare `pip` — it may
resolve to a different Python on some machines).

## Build the knowledge base (order matters, each output feeds the next)

```powershell
python -m app.rag.evaluate [data/raw]   # quality gate: ACCEPT/REVIEW per file
python -m app.rag.clean                 # -> data/processed/*.txt
python -m app.rag.chunk                 # -> data/processed/*.chunks.jsonl
python -m app.rag.index                 # -> data/processed/faiss_index_v2/ + bm25.pkl
python -m app.rag.eval                  # retrieval score (expect MEAN ~0.96)
```

Notes:

- `data/` is gitignored (books + indexes stay local). Fresh clones fetch the
  10 sources first: Shakir vols 1–8 EPUBs (`archive.org/details/22_20221205_20221205_2231`,
  vol 8 from `archive.org/details/22-110808_20221019`), Ibn Hisham
  `0213IbnHisham.SiraNabawiyya.Shamela0023833-ara1.completed` (OpenITI `0225AH`
  repo), Raheeq EPUB (`archive.org/details/2014-alraheq_almakhtom-almubarakfuri`).
- First `index` run downloads ~2.2 GB (bge-m3, one time). CPU build takes
  hours; set `DEVICE=cuda` + `USE_FP16=true` with a CUDA torch for ~15 min
  GPU builds (GTX 1650 Ti 4GB verified).
- `python -m app.rag.eval` measures the production retrieve path, offline.

## Run

```powershell
# terminal 1: API (first /query pays ~1-2 min model load, then fast)
uvicorn app.main:app --reload
# terminal 2: chat UI
streamlit run streamlit_app.py
```

- API: `http://127.0.0.1:8000/docs` (Swagger), health at `/health`
- UI: `http://localhost:8501`
- Docker (API only): `docker compose up --build`, same API URLs

## API reference

`GET /health` -> `{"status": "ok", "app": "Turath AI"}`

`POST /query`

```json
{ "question": "متى وقعت غزوة بدر الكبرى؟", "k": 5 }
```

```json
{
  "answer": "## الإجابة\n... \n\n**المصادر:** الرحيق المختوم، Sira-Ibn-Hisham",
  "citations": [
    {"source": "الرحيق المختوم", "heading": "", "index": 12}
  ]
}
```

## Retrieval scoreboard (fixed 9-probe harness, hit-rate@5)

| Probe | Score |
|---|---|
| Badr / lineage / Rashidun / Ottoman / Hijra / Fath | 0.80 – 1.00 |
| Umayyad era | 1.00 (was 0.40 before heading enrichment) |
| Abbasid fall | 1.00 (was 0.40 before heading enrichment) |
| Mamluks | 0.80 |
| **MEAN** | **0.96** |

Every retrieval change must move this number or get reverted — that rule
killed query expansion (no-op), cross-encoder re-ranking (0.68), and hybrid
BM25 (0.76) on this corpus.

## Configuration (`.env`, see `.env.example`)

| Key | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace downloads (higher limits) |
| `OPENROUTER_API_KEY` | — | generation API key |
| `OPENROUTER_MODEL` | nemotron-3-ultra:free | primary LLM (free list rotates; this ID is env, not code) |
| `DEVICE` / `USE_FP16` | cpu / false | `cuda`+`true` for GPU index builds |
| `INDEX_DIR` | …/faiss_index_v2 | versioned index (v1 kept for rollback) |
| `USE_BM25` | false | parked hybrid branch (proven worse: 0.76) |
| `RETRIEVER_K` | 5 | chunks per answer |

## Project structure

```
app/
  main.py            FastAPI entry (/health, mounts /query)
  config.py          all settings (env-driven, winning defaults in code)
  api/query.py       POST /query; hybrid-ready retrieve() + RRF (parked-off)
  rag/
    evaluate.py      source quality gate (ACCEPT/REVIEW)
    clean.py         normalize/footer-strip/verse-quarantine
    chunk.py         heading sections -> ~500-char sentence packs (JSONL)
    index.py         bge-m3 enriched embeddings -> versioned FAISS + bm25.pkl
    bm25.py          sparse index (parked via USE_BM25)
    qa.py            prompt v3 + resilient OpenRouter client + server-side refs
    eval.py          9-probe retrieval harness (the ruler)
tests/               pytest suite mirroring app/ (offline stubs, no network)
streamlit_app.py     Arabic RTL chat UI (thin HTTP client of /query)
assets/bg.png        Islamic-pattern UI background
data/                raw books + processed artifacts (gitignored, local only)
```

## Key decisions (why things are the way they are)

- **bge-m3 embeddings:** MiniLM failed a live Badr test; research + our own
  measurement picked bge-m3 (Arabic RAG #1 in 2025 studies).
- **Heading-enriched vectors:** stamping `[source | heading]` on each chunk
  before embedding fixed retrospective-intrusion failures (0.80 -> 0.96).
- **English system prompt:** A/B-tested vs Arabic; EN won on format
  compliance with zero language leakage. Content/query/output stay Arabic.
- **Parked, not deleted, BM25:** tested worse on this corpus; kill-switch
  preserves the option for future volumes.
- **Model rotation client:** OpenRouter free models retire without notice
  (Qwen died mid-project); 404-rotation + env-held IDs make that a config
  change, not a redesign. Free tier: 50 req/day.
- **No GPU/GraphRAG/HyDE v1:** measured need drives complexity. Recorded
  here so future work starts from evidence, not hype.

## Requirements

- Python 3.12, ~8 GB free disk (models + books + indexes, all local/cached)
- Docker optional (API image; UI runs locally)
- OpenRouter account + free API key; HuggingFace read token

## License

MIT — see LICENSE.
