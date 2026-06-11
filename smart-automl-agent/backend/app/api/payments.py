"""Payment API — Vodafone Cash payments + admin approval."""
from __future__ import annotations
import json, logging, os, shutil, uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.payment import Payment
from app.models.user import User
from app.services import token_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/payments", tags=["payments"])

ADMIN_EMAILS = {"yousefahmed87a6@gmail.com"}  # ← add your email here

PLANS = [
    {"id": "starter",  "tokens": 5000,  "price": 500,  "label": "Starter",  "desc": "~50 ML trainings"},
    {"id": "pro",      "tokens": 12000, "price": 1000, "label": "Pro",      "desc": "~120 ML trainings"},
    {"id": "premium",  "tokens": 30000, "price": 2000, "label": "Premium",  "desc": "~300 ML trainings"},
]

VODAFONE_NUMBER = "01067461769"


def _is_admin(user: User) -> bool:
    return user.email in ADMIN_EMAILS


@router.get("/plans")
def get_plans():
    return {"plans": PLANS, "vodafone_number": VODAFONE_NUMBER}


@router.post("/submit")
async def submit_payment(
    plan_id:     str        = Form(...),
    sender_phone: str       = Form(...),
    screenshot:  UploadFile = File(...),
    db:          Session    = Depends(get_db),
    user:        User       = Depends(get_current_user),
):
    plan = next((p for p in PLANS if p["id"] == plan_id), None)
    if not plan:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid plan")

    # Save screenshot
    upload_dir = Path(settings.upload_path) / "payment_screenshots"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext      = Path(screenshot.filename).suffix or ".jpg"
    fname    = f"pay_{uuid.uuid4().hex[:12]}{ext}"
    fpath    = upload_dir / fname
    with open(fpath, "wb") as f:
        shutil.copyfileobj(screenshot.file, f)

    pay = Payment(
        user_id    = user.id,
        amount     = plan["price"],
        tokens     = plan["tokens"],
        phone      = sender_phone.strip(),
        screenshot = str(fpath),
        status     = "pending",
    )
    db.add(pay); db.commit(); db.refresh(pay)
    return {"message": "تم إرسال طلب الدفع بنجاح ✓", "payment_id": pay.id}


@router.get("/my")
def my_payments(
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    pays = db.query(Payment).filter(Payment.user_id == user.id).order_by(Payment.created_at.desc()).all()
    return [_fmt(p) for p in pays]


# ── Admin endpoints ──────────────────────────────────────────────────────

@router.get("/admin/all")
def admin_all(
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    if not _is_admin(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    pays = db.query(Payment).order_by(Payment.created_at.desc()).all()
    return [_fmt(p, admin=True) for p in pays]


@router.post("/admin/approve/{payment_id}")
def approve(
    payment_id: int,
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    if not _is_admin(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    pay = db.query(Payment).filter(Payment.id == payment_id).first()
    if not pay:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Payment not found")
    if pay.status == "approved":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Already approved")

    pay.status      = "approved"
    pay.approved_at = datetime.utcnow()
    # Grant tokens
    target_user = db.query(User).filter(User.id == pay.user_id).first()
    if target_user:
        target_user.tokens += pay.tokens
    db.commit()
    return {"message": f"✓ Approved — {pay.tokens} tokens added to {target_user.email if target_user else '?'}"}


@router.post("/admin/reject/{payment_id}")
def reject(
    payment_id: int,
    note: str  = Form(""),
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    if not _is_admin(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    pay = db.query(Payment).filter(Payment.id == payment_id).first()
    if not pay:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Payment not found")
    pay.status = "rejected"
    pay.note   = note
    db.commit()
    return {"message": "Payment rejected"}


@router.get("/admin/screenshot/{payment_id}")
def get_screenshot(
    payment_id: int,
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    if not _is_admin(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    pay = db.query(Payment).filter(Payment.id == payment_id).first()
    if not pay or not pay.screenshot or not Path(pay.screenshot).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Screenshot not found")
    return FileResponse(pay.screenshot)


def _fmt(p: Payment, admin=False) -> dict:
    d = {
        "id":         p.id,
        "amount":     p.amount,
        "tokens":     p.tokens,
        "status":     p.status,
        "phone":      p.phone,
        "note":       p.note,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "approved_at":p.approved_at.isoformat() if p.approved_at else None,
    }
    if admin:
        d["user_id"]    = p.user_id
        d["user_email"] = p.user.email if p.user else None
        d["user_name"]  = p.user.name  if p.user else None
        d["screenshot"] = bool(p.screenshot)
    return d