"""Kaggle Training API — submit training to Kaggle GPU."""
from __future__ import annotations
import json, logging, threading, time, uuid
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kaggle-train", tags=["kaggle-training"])

# In-memory job store (use Redis in production)
_jobs: dict = {}

# Simple encryption key — add FERNET_KEY to .env
def _get_fernet():
    key = getattr(settings, "fernet_key", None)
    if not key:
        # Generate and cache a key (not persistent across restarts)
        key = Fernet.generate_key().decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


class KaggleKeyIn(BaseModel):
    username: str
    key:      str


class TrainRequest(BaseModel):
    dataset_id:    int
    target_column: str
    task_type:     str   # classification | regression
    model_name:    str   # Auto (Best) | Random Forest | etc.


@router.post("/connect")
def connect_kaggle(req: KaggleKeyIn,
                   db: Session = Depends(get_db),
                   user: User  = Depends(get_current_user)):
    """Store encrypted Kaggle API credentials."""
    # Test connection first
    from app.services.kaggle_service import KaggleService
    svc = KaggleService(req.username, req.key)
    if not svc.test_connection():
        raise HTTPException(400, "Invalid Kaggle credentials — please check your username and API key")

    # Encrypt and store
    f   = _get_fernet()
    enc = f.encrypt(json.dumps({"username": req.username, "key": req.key}).encode()).decode()
    user.kaggle_token = enc
    db.commit()
    return {"message": "Kaggle connected ✓", "username": req.username}


@router.get("/status")
def kaggle_status(db: Session = Depends(get_db),
                  user: User  = Depends(get_current_user)):
    """Check if Kaggle is connected."""
    connected = bool(getattr(user, "kaggle_token", None))
    username  = None
    if connected:
        try:
            f    = _get_fernet()
            data = json.loads(f.decrypt(user.kaggle_token.encode()))
            username = data.get("username")
        except Exception:
            connected = False
    return {"connected": connected, "username": username}


@router.delete("/disconnect")
def disconnect_kaggle(db: Session = Depends(get_db),
                      user: User  = Depends(get_current_user)):
    user.kaggle_token = None
    db.commit()
    return {"message": "Kaggle disconnected"}


@router.post("/train")
def start_kaggle_training(req: TrainRequest,
                          db: Session  = Depends(get_db),
                          user: User   = Depends(get_current_user)):
    """Start training job on Kaggle GPU."""
    if not getattr(user, "kaggle_token", None):
        raise HTTPException(400, "Kaggle not connected. Please add your Kaggle API key in Settings.")

    # Decrypt credentials
    f    = _get_fernet()
    creds = json.loads(f.decrypt(user.kaggle_token.encode()))
    kg_user = creds["username"]
    kg_key  = creds["key"]

    # Get dataset file path
    from app.models.dataset import Dataset
    from app.models.project import Project
    ds = db.query(Dataset).join(Project).filter(
        Dataset.id == req.dataset_id, Project.user_id == user.id
    ).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status":   "uploading",
        "progress": 5,
        "message":  "Uploading dataset to Kaggle...",
        "user_id":  user.id,
        "model_path": None,
        "metrics":  None,
        "error":    None,
    }

    # Run in background
    def _run():
        from app.services.kaggle_service import KaggleService
        svc     = KaggleService(kg_user, kg_key)
        ds_slug = f"mgx-{job_id[:8]}"
        nb_slug = f"mgx-train-{job_id[:8]}"

        try:
            # Step 1: Upload dataset
            _jobs[job_id].update({"status": "uploading", "progress": 10,
                                   "message": "Uploading dataset to Kaggle..."})
            svc.upload_dataset(ds.path, ds_slug, f"MGX Dataset {job_id[:6]}")

            # Step 2: Create and run notebook
            _jobs[job_id].update({"status": "queued", "progress": 25,
                                   "message": "Starting Kaggle GPU notebook..."})
            kernel_slug = svc.create_and_run_notebook(
                dataset_slug  = ds_slug,
                target_column = req.target_column,
                task_type     = req.task_type,
                model_name    = req.model_name,
                notebook_slug = nb_slug,
            )

            # Step 3: Poll until done
            _jobs[job_id].update({"status": "running", "progress": 40,
                                   "message": "Training on Kaggle GPU..."})
            for attempt in range(120):  # max 60 minutes
                time.sleep(30)
                st = svc.get_kernel_status(kernel_slug)
                kaggle_status = st.get("status", "").lower()
                logger.info("Kaggle job %s: %s", job_id, kaggle_status)

                if kaggle_status == "complete":
                    _jobs[job_id].update({"progress": 85, "message": "Downloading model..."})
                    break
                elif kaggle_status in ("error", "failed", "cancelack"):
                    raise Exception(f"Kaggle training failed: {st.get('error', 'Unknown error')}")
                else:
                    pct = min(40 + attempt, 84)
                    _jobs[job_id].update({"progress": pct,
                                          "message": f"Training... ({attempt*30//60}m elapsed)"})
            else:
                raise Exception("Kaggle training timed out after 60 minutes")

            # Step 4: Download output
            _jobs[job_id].update({"status": "downloading", "progress": 88,
                                   "message": "Downloading model from Kaggle..."})
            dest_dir  = Path(settings.model_dir) / str(user.id) / job_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            files = svc.download_output(kernel_slug, str(dest_dir))

            model_path = next((f for f in files if f.endswith(".pkl")), None)
            meta_path  = next((f for f in files if f.endswith(".json")), None)

            metrics = {}
            if meta_path:
                with open(meta_path) as mf:
                    meta_data = json.load(mf)
                    metrics = meta_data.get("metrics", {})

            _jobs[job_id].update({
                "status":     "complete",
                "progress":   100,
                "message":    "Training complete! 🎉",
                "model_path": model_path,
                "metrics":    metrics,
            })
            logger.info("Kaggle job %s complete: %s", job_id, metrics)

        except Exception as e:
            logger.error("Kaggle job %s failed: %s", job_id, e)
            _jobs[job_id].update({
                "status":  "failed",
                "message": str(e),
                "error":   str(e),
            })

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "message": "Training started on Kaggle GPU 🚀"}


@router.get("/job/{job_id}")
def get_job_status(job_id: str,
                   user: User = Depends(get_current_user)):
    """Poll training job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("user_id") != user.id:
        raise HTTPException(403, "Not your job")
    return {k: v for k, v in job.items() if k != "user_id"}