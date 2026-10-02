from fastapi import FastAPI

from app.api.query import router as query_router
from app.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(query_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}