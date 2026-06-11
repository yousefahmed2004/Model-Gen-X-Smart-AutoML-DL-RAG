"""
Application settings loaded from environment / .env.

.env search order (first found wins):
  1. backend/.env  next to the app/ package  <- most reliable
  2. backend/.env  relative to repo root
  3. .env          in current working directory
  4. backend/.env  relative to current working directory
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# config.py lives at  <repo>/backend/app/core/config.py
# parent.parent.parent  =>  <repo>
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _find_env() -> str:
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",   # backend/.env (next to app/)
        BASE_DIR / "backend" / ".env",                      # <repo>/backend/.env
        Path.cwd() / ".env",                                # wherever uvicorn was launched
        Path.cwd() / "backend" / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return str(BASE_DIR / "backend" / ".env")  # fallback, may not exist


_ENV_PATH = _find_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Model Gen X"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    frontend_origin: str = "http://localhost:5500"

    # Security
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 1440
    algorithm: str = "HS256"

    # Database
    database_url: str = "sqlite:///./smart_automl.db"

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/auth/google/callback"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Kaggle
    kaggle_username: str = ""
    kaggle_key: str = ""

    # Tokens
    free_tokens_on_signup:    int = 1000
    token_cost_per_train:     int = 100   # ML Auto
    token_cost_per_predict:   int = 1
    token_cost_dl_tabular:    int = 200
    token_cost_dl_image:      int = 500
    token_cost_dl_text:       int = 500
    token_cost_upload_small:  int = 0     # < 10MB
    token_cost_upload_medium: int = 50    # 10-50MB
    token_cost_upload_large:  int = 150   # 50-200MB
    token_cost_upload_xlarge: int = 300   # 200MB+

    # Storage
    upload_dir: str = "./uploads"
    model_dir: str = "./trained_models"
    max_upload_mb: int = 200

    @property
    def upload_path(self) -> Path:
        p = (BASE_DIR / self.upload_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def model_path(self) -> Path:
        p = (BASE_DIR / self.model_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def debug_env(self) -> dict:
        return {
            "env_file_used": _ENV_PATH,
            "env_file_exists": Path(_ENV_PATH).is_file(),
            "google_client_id_set": bool(self.google_client_id),
            "gemini_key_set": bool(self.gemini_api_key),
            "database_url": self.database_url,
            "frontend_origin": self.frontend_origin,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()