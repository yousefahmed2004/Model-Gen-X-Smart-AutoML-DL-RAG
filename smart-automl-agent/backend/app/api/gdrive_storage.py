"""
Google Drive Storage endpoints.

GET  /api/gdrive/storage  — user's Drive storage info
GET  /api/gdrive/files    — list user's Model Gen X files
POST /api/gdrive/sync     — sync local dataset to user's Drive
"""
from __future__ import annotations
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.project import Project
from app.models.user import User
from app.services.gdrive_service import GDriveService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/gdrive", tags=["gdrive"])


def _get_drive(user: User) -> GDriveService:
    if not user.drive_token:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Google Drive not connected. Please sign out and sign in again with Google."
        )
    return GDriveService(user.drive_token)


@router.get("/storage")
def storage_info(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get user's Google Drive storage usage."""
    drive = _get_drive(user)
    try:
        info    = drive.get_storage_info()
        folders = drive.get_app_folders()
        return {**info, "folders": folders, "connected": True}
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Drive error: {e}")


@router.get("/files")
def list_files(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List all Model Gen X files in user's Drive."""
    drive = _get_drive(user)
    try:
        folders  = drive.get_app_folders()
        datasets = drive.list_folder(folders["datasets"])
        models   = drive.list_folder(folders["models"])
        return {"datasets": datasets, "models": models}
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Drive error: {e}")


@router.post("/sync/dataset/{dataset_id}")
def sync_dataset_to_drive(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a dataset to user's Google Drive."""
    ds = (
        db.query(Dataset).join(Project)
        .filter(Dataset.id == dataset_id, Project.user_id == user.id)
        .first()
    )
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")

    drive = _get_drive(user)
    try:
        folders = drive.get_app_folders()
        result  = drive.upload_file(ds.path, folders["datasets"], ds.filename)
        return {
            "message":      "Dataset synced to your Google Drive ✓",
            "drive_file_id": result["id"],
            "drive_link":   result.get("webViewLink"),
            "filename":     ds.filename,
        }
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Drive sync failed: {e}")


@router.post("/sync/model/{model_id}")
def sync_model_to_drive(
    model_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a trained model to user's Google Drive."""
    m = (
        db.query(MLModel).join(Project)
        .filter(MLModel.id == model_id, Project.user_id == user.id)
        .first()
    )
    if not m:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    if not Path(m.artifact_path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model file not found")

    drive = _get_drive(user)
    try:
        folders  = drive.get_app_folders()
        filename = f"model_{m.id}_{m.name.replace(' ','_')}{Path(m.artifact_path).suffix}"
        result   = drive.upload_file(m.artifact_path, folders["models"], filename)
        return {
            "message":      "Model synced to your Google Drive ✓",
            "drive_file_id": result["id"],
            "drive_link":   result.get("webViewLink"),
            "filename":     filename,
        }
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Drive sync failed: {e}")