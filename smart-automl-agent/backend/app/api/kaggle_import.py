"""
Kaggle Dataset Import — يدعم Tabular + Image datasets
للصور: بيحفظ الـ path مباشرة بدون ZIP (أسرع بكتير)
"""
from __future__ import annotations
import json, logging, math, os, shutil, zipfile
from pathlib import Path
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.ml.dataset_loader import suggest_target
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.user import User
from app.schemas.schemas import DatasetPreview

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kaggle", tags=["kaggle"])

TABULAR_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}
IMAGE_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def _json_safe(v):
    if v is None: return None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): f=float(v); return None if math.isnan(f) else f
    if isinstance(v, float) and math.isnan(v): return None
    return str(v) if not isinstance(v, (int, float, bool, str)) else v

class KaggleImportRequest(BaseModel):
    slug: str
    project_id: int
    file_name: str = ""

@router.post("/import", response_model=DatasetPreview)
async def import_from_kaggle(
    body: KaggleImportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = db.query(Project).filter(
        Project.id == body.project_id, Project.user_id == user.id
    ).first()
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    slug = body.slug.strip().strip("/")
    if len(slug.split("/")) != 2:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
            "صيغة غلط. استخدم: owner/dataset-name  مثال: uciml/iris")

    os.environ["KAGGLE_USERNAME"] = settings.kaggle_username or ""
    os.environ["KAGGLE_KEY"]      = settings.kaggle_key or ""

    try:
        import kagglehub
        logger.info("Downloading: %s", slug)
        local_path = Path(kagglehub.dataset_download(slug))
        logger.info("Downloaded to: %s", local_path)
    except ImportError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
            "kagglehub غير مثبت. شغّل: pip install 'kagglehub[pandas-datasets]'")
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"فشل التنزيل: {exc}")

    dataset_name  = slug.replace("/", "_")
    tabular_files = sorted([f for f in local_path.rglob("*")
                            if f.is_file() and f.suffix.lower() in TABULAR_EXT],
                           key=lambda f: f.stat().st_size, reverse=True)
    image_files   = [f for f in local_path.rglob("*")
                     if f.is_file() and f.suffix.lower() in IMAGE_EXT]

    # ── TABULAR ──────────────────────────────────────────────────────────
    if tabular_files:
        chosen = tabular_files[0]
        if body.file_name:
            m = next((f for f in tabular_files if f.name == body.file_name), None)
            if not m:
                raise HTTPException(status.HTTP_400_BAD_REQUEST,
                    f"الملف '{body.file_name}' مش موجود. المتاح: {', '.join(f.name for f in tabular_files)}")
            chosen = m

        import pandas as pd
        try:
            from kagglehub import KaggleDatasetAdapter
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, slug, chosen.name)
        except Exception:
            suf = chosen.suffix.lower()
            if suf in (".csv",".tsv"):    df = pd.read_csv(chosen, sep="\t" if suf==".tsv" else ",")
            elif suf in (".xlsx",".xls"): df = pd.read_excel(chosen)
            elif suf == ".parquet":       df = pd.read_parquet(chosen)
            else:                         df = pd.read_json(chosen)

        dest = settings.upload_path / f"kaggle_{dataset_name}.csv"
        df.to_csv(dest, index=False)

        columns = [{"name": str(c), "dtype": str(df[c].dtype),
                    "n_missing": int(df[c].isna().sum()),
                    "n_unique": int(df[c].nunique(dropna=True)),
                    "sample": [_json_safe(v) for v in df[c].dropna().head(3).tolist()]}
                   for c in df.columns]
        head      = [{str(k): _json_safe(v) for k,v in r.items()} for r in df.head(10).to_dict("records")]
        n_rows    = int(len(df))
        n_cols    = int(df.shape[1])
        file_type = "csv"
        dest_name = dest.name
        target_col = suggest_target(columns)

    # ── IMAGE ─────────────────────────────────────────────────────────────
    elif image_files:
        logger.info("Image dataset: %d images found", len(image_files))

        # Group by class folder
        class_dirs: dict[str, list[Path]] = {}
        for img in image_files:
            cls = img.parent.name
            class_dirs.setdefault(cls, []).append(img)

        classes = list(class_dirs.keys())
        logger.info("Classes: %s", classes[:10])

        # ── بدل ما نعمل ZIP كبير، نحفظ path الفولدر الأصلي مباشرة ──────
        # الـ dl_engine يقرأ من فولدر أو ZIP — هنحفظ الـ path كـ folder_path
        # ونعمل ZIP صغير فيه أول 200 صورة بس عشان نتحقق إن الداتا شغالة

        # حساب عدد الصور لكل class
        total_images = sum(len(v) for v in class_dirs.values())

        # حفظ metadata ملف JSON بدل ZIP الضخم
        dest_name = f"kaggle_{dataset_name}_images.json"
        dest      = settings.upload_path / dest_name

        metadata = {
            "type":        "image_dataset",
            "source":      "kaggle",
            "slug":        slug,
            "local_path":  str(local_path),   # المسار الأصلي للصور
            "classes":     classes,
            "class_counts": {c: len(v) for c, v in class_dirs.items()},
            "total_images": total_images,
        }
        dest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

        n_rows    = total_images
        n_cols    = len(classes)
        file_type = "image_folder"
        columns   = [
            {"name": "image_class", "dtype": "object",
             "n_missing": 0, "n_unique": len(classes), "sample": classes[:3]},
            {"name": "total_images", "dtype": "int64",
             "n_missing": 0, "n_unique": 1, "sample": [total_images]},
        ]
        head = [{"class": c, "images": len(v)} for c, v in list(class_dirs.items())[:10]]
        target_col = "image_class"

        logger.info("Image metadata saved: %d classes, %d images", len(classes), total_images)

    else:
        all_files = [f.name for f in local_path.rglob("*") if f.is_file()][:10]
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
            f"مفيش ملفات مدعومة.\nالملفات الموجودة: {', '.join(all_files)}")

    # ── Save to DB ───────────────────────────────────────────────────────
    ds = Dataset(
        project_id    = project.id,
        filename      = dest_name,
        path          = str(dest),
        file_type     = file_type,
        size_bytes    = dest.stat().st_size,
        n_rows        = n_rows,
        n_cols        = n_cols,
        columns_json  = json.dumps(columns),
        target_column = target_col,
    )
    db.add(ds); db.commit(); db.refresh(ds)

    # Sync to user's Drive
    drive_link = None
    if user.drive_token:
        try:
            from app.services.gdrive_service import GDriveService
            drive_svc = GDriveService(user.drive_token)
            folders   = drive_svc.get_app_folders()
            result    = drive_svc.upload_file(dest, folders["datasets"], dest_name)
            drive_link = result.get("webViewLink")
        except Exception:
            pass

    preview = DatasetPreview(
        dataset_id=ds.id, columns=columns, head=head, n_rows=n_rows, n_cols=n_cols
    )
    if drive_link:
        return {**preview.model_dump(), "drive_link": drive_link}
    return preview