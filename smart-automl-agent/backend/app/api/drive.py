"""
Google Drive dataset import.

Accepts any public sharing link and downloads the file directly.

Supported link formats:
  https://drive.google.com/file/d/<FILE_ID>/view
  https://drive.google.com/open?id=<FILE_ID>
  https://docs.google.com/spreadsheets/d/<FILE_ID>/...
  https://drive.google.com/uc?id=<FILE_ID>
  Bare file ID

For PRIVATE files the user must share the file as "Anyone with the link".
We use the direct-download URL so no OAuth scope is needed on the server.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.ml.dataset_loader import inspect_dataset, suggest_target
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.user import User
from app.schemas.schemas import DatasetPreview

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/drive", tags=["drive"])


def _extract_file_id(url: str) -> str:
    """Pull the Google Drive file ID from any common share URL."""
    url = url.strip()

    # Already a bare ID (no slashes, no dots)
    if re.fullmatch(r"[A-Za-z0-9_-]{25,}", url):
        return url

    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/spreadsheets/d/([A-Za-z0-9_-]+)",
        r"/document/d/([A-Za-z0-9_-]+)",
        r"/presentation/d/([A-Za-z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)

    raise ValueError(
        "Could not extract a Google Drive file ID from that URL. "
        "Make sure it's a valid Google Drive sharing link."
    )


def _direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"


def _confirm_url(file_id: str, confirm_code: str) -> str:
    return (
        f"https://drive.google.com/uc?export=download"
        f"&id={file_id}&confirm={confirm_code}"
    )


ALLOWED_CONTENT_TYPES = {
    "text/csv", "text/plain",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",
    "application/zip",
}

EXT_FROM_MIME = {
    "text/csv":                ".csv",
    "text/plain":              ".csv",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}


async def _download_drive_file(file_id: str, dest: Path) -> str:
    """
    Download a Drive file to `dest`.  Handles the virus-scan warning
    redirect that Google adds for large files.

    Returns the local filename with proper extension.
    """
    url = _direct_download_url(file_id)
    filename = f"drive_{file_id}"

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        resp = await client.get(url)

        # Google may redirect to a confirm page for large files
        if resp.status_code == 200 and b"virus scan warning" in resp.content[:2000].lower():
            # Extract confirm token from page
            m = re.search(r'confirm=([A-Za-z0-9_-]+)', resp.text)
            if m:
                resp = await client.get(_confirm_url(file_id, m.group(1)))

        if resp.status_code != 200:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Google Drive returned HTTP {resp.status_code}. "
                "Make sure the file is shared as 'Anyone with the link'."
            )

        # Detect extension from Content-Disposition or Content-Type
        cd = resp.headers.get("content-disposition", "")
        ct = resp.headers.get("content-type", "").split(";")[0].strip()

        ext = ".csv"  # default
        m_cd = re.search(r'filename[^;=\n]*=["\']?([^"\';\n]+)', cd)
        if m_cd:
            ext = Path(m_cd.group(1).strip()).suffix.lower() or ext
        elif ct in EXT_FROM_MIME:
            ext = EXT_FROM_MIME[ct]

        # Reject obviously non-tabular files
        if ext.lower() in {".jpg", ".png", ".mp4", ".pdf", ".zip"}:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"File type '{ext}' is not supported. Upload a CSV or Excel file."
            )

        filename = f"drive_{uuid.uuid4().hex[:10]}{ext}"
        dest_file = dest.parent / filename
        dest_file.write_bytes(resp.content)
        return filename


class DriveImportRequest:
    pass


from pydantic import BaseModel

class DriveImport(BaseModel):
    url: str
    project_id: int


@router.post("/import", response_model=DatasetPreview)
async def import_from_drive(
    body: DriveImport,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = (
        db.query(Project)
        .filter(Project.id == body.project_id, Project.user_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    # Extract Drive file ID
    try:
        file_id = _extract_file_id(body.url)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    # Download to uploads directory
    dest_placeholder = settings.upload_path / f"drive_{file_id}_tmp"
    try:
        filename = await _download_drive_file(file_id, dest_placeholder)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Drive download failed")
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Download failed: {exc}. Make sure the file is shared publicly."
        )

    dest = settings.upload_path / filename
    size_bytes = dest.stat().st_size

    # Inspect
    try:
        info = inspect_dataset(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not parse file: {exc}")

    ds = Dataset(
        project_id=project.id,
        filename=filename,
        path=str(dest),
        file_type=Path(filename).suffix.lstrip("."),
        size_bytes=size_bytes,
        n_rows=info["n_rows"],
        n_cols=info["n_cols"],
        columns_json=json.dumps(info["columns"]),
        target_column=suggest_target(info["columns"]),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    # Sync to user's Drive folder
    drive_link = None
    if user.drive_token:
        try:
            from app.services.gdrive_service import GDriveService
            drive_svc = GDriveService(user.drive_token)
            folders   = drive_svc.get_app_folders()
            result    = drive_svc.upload_file(dest, folders["datasets"], filename)
            drive_link = result.get("webViewLink")
        except Exception:
            pass

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