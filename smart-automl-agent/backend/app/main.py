"""
FastAPI app entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, chat, rag as rag_router, datasets, drive, gdrive_storage, kaggle_import, kaggle_training, payments, projects, training
from app.core.config import settings
from app.db.session import init_db


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Initializing database (creating tables if missing)…")
    init_db()
    logger.info("Model Gen X is ready.")
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI-powered AutoML platform — graduation project edition.",
    lifespan=lifespan,
)


# ---------- CORS ----------
# Open during local development; lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://[::1]:5500",
        "http://[::]:5500",
        "http://185.205.246.109:5001",
        "http://185.205.246.109",
        "http://localhost:3000",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Routers ----------
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(drive.router)
app.include_router(gdrive_storage.router)
app.include_router(kaggle_import.router)
app.include_router(kaggle_training.router)
app.include_router(training.router)
app.include_router(chat.router)
app.include_router(payments.router)
app.include_router(rag_router.router)


# ---------- Health + root ----------
@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "env": settings.app_env,
        "docs": "/docs",
        "gemini_enabled": bool(settings.gemini_api_key),
        "google_oauth_enabled": bool(settings.google_client_id),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------- Global error fallback ----------
@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/api/debug-env")
def debug_env():
    """
    Visit http://localhost:8000/api/debug-env to diagnose config issues.
    Shows which .env file was found and whether keys are set.
    Remove this route before going to production.
    """
    return settings.debug_env()

@app.get("/api/debug-gemini")
def debug_gemini():
    """Check Gemini status — visit http://localhost:8000/api/debug-gemini"""
    from app.services.gemini_service import gemini
    return gemini.status()