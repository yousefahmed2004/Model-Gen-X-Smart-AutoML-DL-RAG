"""Dataset upload, listing, and preview."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.ml.dataset_loader import inspect_dataset, suggest_target
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.user import User
from app.schemas.schemas import DatasetOut, DatasetPreview


from app.services import token_service

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

ALLOWED_EXT = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json"}

MB = 1024 * 1024

def _upload_token_cost(size_bytes: int) -> int:
    mb = size_bytes / MB
    if mb < 10:   return settings.token_cost_upload_small
    if mb < 50:   return settings.token_cost_upload_medium
    if mb < 200:  return settings.token_cost_upload_large
    return settings.token_cost_upload_xlarge


@router.post("/upload", response_model=DatasetPreview)
async def upload_dataset(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    orig_name = file.filename or "upload.csv"
    suffix = Path(orig_name).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type {suffix}. Allowed: {sorted(ALLOWED_EXT)}",
        )

    # Save under a uuid-prefixed name to avoid collisions
    unique = f"{uuid.uuid4().hex[:12]}_{orig_name}"
    dest: Path = settings.upload_path / unique
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    size_bytes = dest.stat().st_size
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {settings.max_upload_mb} MB",
        )

    # Charge tokens based on file size
    token_cost = _upload_token_cost(size_bytes)
    if token_cost > 0:
        try:
            token_service.spend(db, user, token_cost, f"upload_{dest.name}")
        except token_service.InsufficientTokens as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, str(exc))

    try:
        info = inspect_dataset(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not parse: {exc}")

    ds = Dataset(
        project_id=project.id,
        filename=orig_name,
        path=str(dest),
        file_type=suffix.lstrip("."),
        size_bytes=size_bytes,
        n_rows=info["n_rows"],
        n_cols=info["n_cols"],
        columns_json=json.dumps(info["columns"]),
        target_column=suggest_target(info["columns"]),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    # Auto-sync to user's Google Drive
    drive_link = None
    if user.drive_token:
        try:
            from app.services.gdrive_service import GDriveService
            drive_svc = GDriveService(user.drive_token)
            folders   = drive_svc.get_app_folders()
            result    = drive_svc.upload_file(dest, folders["datasets"], orig_name)
            drive_link = result.get("webViewLink")
        except Exception as e:
            pass  # non-critical

    preview = DatasetPreview(
        dataset_id=ds.id,
        columns=info["columns"],
        head=info["head"],
        n_rows=info["n_rows"],
        n_cols=info["n_cols"],
    )
    if drive_link:
        return {**preview.model_dump(), "drive_link": drive_link}
    return preview


@router.get("/project/{project_id}", response_model=list[DatasetOut])
def list_datasets(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return db.query(Dataset).filter(Dataset.project_id == project_id).all()


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ds = (
        db.query(Dataset)
        .join(Project)
        .filter(Dataset.id == dataset_id, Project.user_id == user.id)
        .first()
    )
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return ds

@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ds = (
        db.query(Dataset)
        .join(Project)
        .filter(Dataset.id == dataset_id, Project.user_id == user.id)
        .first()
    )
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    info = inspect_dataset(ds.path)
    return DatasetPreview(dataset_id=ds.id, **info)

@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    import os
    ds = (
        db.query(Dataset)
        .join(Project)
        .filter(Dataset.id == dataset_id, Project.user_id == user.id)
        .first()
    )
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if ds.path and os.path.exists(ds.path):
        try:
            os.remove(ds.path)
        except Exception as e:
            pass
    try:
        from app.services.gdrive_service import gdrive
        if hasattr(ds, 'drive_file_id') and ds.drive_file_id:
            gdrive.delete_file(user.id, ds.drive_file_id)
    except Exception:
        pass
    db.delete(ds)
    db.commit()
    return {"message": "Dataset deleted"}
