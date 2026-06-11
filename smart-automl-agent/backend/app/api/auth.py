"""
Authentication endpoints.

Two flows:
  1. Google OAuth 2.0  — /api/auth/google/login → callback → redirect to frontend
  2. Email + password  — /api/auth/register, /api/auth/login
"""
from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models.user import User
from app.schemas.schemas import EmailLoginIn, EmailRegisterIn, TokenResponse, UserOut
from app.services import token_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── Email / password ───────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse)
def register(data: EmailRegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")
    user = User(email=data.email, name=data.name, hashed_password=hash_password(data.password))
    db.add(user); db.commit(); db.refresh(user)
    token_service.signup_grant(db, user)
    return TokenResponse(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(data: EmailLoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not user.hashed_password or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    return TokenResponse(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


# ── Google OAuth ────────────────────────────────────────────────────────
GOOGLE_AUTH     = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN    = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO = "https://www.googleapis.com/oauth2/v3/userinfo"

_pending_states: dict[str, bool] = {}


@router.get("/google/login")
def google_login():
    if not settings.google_client_id:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
            "Google OAuth is not configured. Use email/password login.")
    state = secrets.token_urlsafe(16)
    _pending_states[state] = True
    params = {
        "client_id":     settings.google_client_id,
        "redirect_uri":  settings.google_redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile https://www.googleapis.com/auth/drive.file",
        "state":         state,
        "access_type":   "offline",
        "prompt":        "consent",
    }
    return RedirectResponse(f"{GOOGLE_AUTH}?{urlencode(params)}")


@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    code  = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state or not _pending_states.pop(state, False):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid OAuth state")

    async with httpx.AsyncClient(timeout=15) as client:
        tok = await client.post(GOOGLE_TOKEN, data={
            "code": code,
            "client_id":     settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri":  settings.google_redirect_uri,
            "grant_type":    "authorization_code",
        })
        if tok.status_code != 200:
            logger.error("Google token exchange failed: %s", tok.text)
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Token exchange failed")
        tok_data      = tok.json()
        google_access = tok_data.get("access_token")
        drive_token = __import__("json").dumps({
            "access_token":  tok_data.get("access_token"),
            "refresh_token": tok_data.get("refresh_token"),
            "token_type":    tok_data.get("token_type", "Bearer"),
        })

        info = await client.get(GOOGLE_USERINFO,
                                headers={"Authorization": f"Bearer {google_access}"})
        info.raise_for_status()
        profile = info.json()

    sub   = profile.get("sub")
    email = profile.get("email")
    if not sub or not email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing Google profile fields")

    user = db.query(User).filter((User.google_sub == sub) | (User.email == email)).first()
    if not user:
        user = User(email=email, name=profile.get("name"),
                    picture=profile.get("picture"), google_sub=sub,
                    drive_token=drive_token)
        db.add(user); db.commit(); db.refresh(user)
        token_service.signup_grant(db, user)
    else:
        if not user.google_sub:
            user.google_sub = sub
            user.picture = profile.get("picture") or user.picture
        user.drive_token = drive_token  # always update token
        db.commit()

    # Create Drive folders for user in background
    if drive_token:
        try:
            from app.services.gdrive_service import GDriveService
            drive = GDriveService(drive_token)
            drive.get_app_folders()
            logger.info("Drive folders created for user %s", email)
        except Exception as e:
            logger.warning("Drive folder creation failed (non-critical): %s", e)

    jwt_token = create_access_token(user.id)

    # ── Redirect directly to the login page with token in URL fragment ──
    # The login page JS reads the fragment, stores it, and goes to dashboard.
    # This avoids any inline-script / eval / CSP issues completely.
    frontend = settings.frontend_origin.rstrip("/")
    return RedirectResponse(
        f"{frontend}/pages/login.html#token={jwt_token}",
        status_code=302,
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user