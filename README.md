# Turath AI (تراث) — The Intelligent Islamic History Assistant

Arabic-first AI question-answering over Islamic history and Seerah books.
Built with FastAPI + LangChain + FAISS. HuggingFace models.

## Quickstart (local)

    python -m venv .venv; .\.venv\Scripts\Activate.ps1
    pip install -e .
    copy .env.example .env
    uvicorn app.main:app --reload

Open http://127.0.0.1:8000/health — expect {"status":"ok","app":"Turath AI"}.
API docs: http://127.0.0.1:8000/docs

## Run with Docker

    docker compose up --build

Same URLs as above. Stop with Ctrl+C, then run: docker compose down

## Run tests

    pytest -v

## Project structure

- app/main.py — FastAPI entry, routes
- app/config.py — Settings (env vars, model names, paths)
- app/api/ — Routers (added Phase 3: /query)
- app/rag/ — RAG pipeline (added Phase 2)
- app/graph/ — Graph RAG placeholder (Phase 4)
- data/raw/ — Source book files (gitignored)
- data/processed/ — Cleaned text + chunks (gitignored)
- tests/ — pytest suite
- docs/ — Design notes, decisions

## Requirements

- Python 3.12
- Docker (optional, for container run)
- HuggingFace account + read token (HF_TOKEN)