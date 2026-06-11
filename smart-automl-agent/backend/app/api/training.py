"""Train models + list models + predict.

Training runs in a background thread so the HTTP request returns immediately.
Frontend polls /api/training/status/{job_id} to check progress.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import SessionLocal, get_db
from app.ml.automl_engine import predict, train_from_file
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.project import Project
from app.models.user import User
from app.schemas.schemas import (
    DLTrainRequest,
    ModelOut,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainingResult,
)
from app.services import token_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])

# ── In-memory job store ───────────────────────────────────────────────────
_jobs: dict[str, dict[str, Any]] = {}


def _save_model_to_db(project, result: dict, db: Session) -> MLModel:
    metrics = result["metrics"]
    model = MLModel(
        project_id=project.id,
        name=result["best_model_name"],
        framework=result["framework"],
        artifact_path=result["artifact_path"],
        task_type=result["task_type"],
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1=metrics.get("f1"),
        metrics_json=json.dumps(metrics),
        feature_columns_json=json.dumps(result["feature_columns"]),
        target_column=result["target_column"],
        confusion_matrix_json=json.dumps(result["confusion_matrix"]) if result["confusion_matrix"] else None,
        training_log=result["log"],
    )
    db.add(model)
    project.status = "completed"
    db.commit()
    db.refresh(model)
    return model


def _run_training_thread(job_id: str, ds_path: str, ds_id: int,
                          target_column: str, task_type: str,
                          project_id: int, user_id: int,
                          dl_type: str = "", model_arch: str = "resnet18",
                          epochs: int = 20, img_size: int = 128,
                          text_column: str = "", selected_model: str = "Auto (Best)",
                          output_format: str = "joblib",
                          hyperparams: dict | None = None,
                          preprocessing: dict | None = None):
    log_lines = []

    def _log(msg):
        log_lines.append(msg)
        _jobs[job_id]["log"] = "\n".join(log_lines)
        logger.info("[job:%s] %s", job_id[:8], msg)

    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        project.status = "training"
        db.commit()

        _log("Starting training...")

        if dl_type:
            from app.ml.dl_engine import (
                train_image_classification,
                train_tabular_dl,
                train_text_classification,
            )
            _log(f"Deep Learning: {dl_type} | arch: {model_arch} | epochs: {epochs}")
            if dl_type == "image":
                result = train_image_classification(
                    zip_path=ds_path, model_arch=model_arch,
                    epochs=epochs, img_size=img_size,
                )
            elif dl_type == "text":
                result = train_text_classification(
                    dataset_path=ds_path, text_column=text_column,
                    label_column=target_column, epochs=epochs,
                )
            else:
                result = train_tabular_dl(
                    dataset_path=ds_path, target_column=target_column,
                    epochs=epochs,
                )
        else:
            _log(f"AutoML: target={target_column} task={task_type} model={selected_model} format={output_format}")
            result = train_from_file(
                dataset_path   = ds_path,
                target_column  = target_column,
                task_type      = task_type,
                selected_model = selected_model,
                hyperparams    = hyperparams,
                preprocessing  = preprocessing,
            )
            if output_format != "joblib":
                try:
                    from app.ml.automl_engine import convert_model
                    new_path = convert_model(result["artifact_path"], output_format, result["feature_columns"])
                    result["artifact_path"] = new_path
                    _log(f"Converted to {output_format.upper()}: {new_path}")
                except Exception as e:
                    _log(f"Format conversion failed ({output_format}): {e} - keeping .joblib")

        if result.get("log"):
            for line in result["log"].split("\n"):
                if line.strip():
                    log_lines.append(line)
        result["log"] = "\n".join(log_lines)

        model = _save_model_to_db(project, result, db)
        _jobs[job_id].update({
            "status":   "completed",
            "model_id": model.id,
            "log":      result["log"],
            "metrics":  result["metrics"],
            "confusion_matrix": result["confusion_matrix"],
            "feature_columns":  result["feature_columns"],
        })
        _log(f"Done! Model id={model.id}")

        try:
            user_obj = db.query(User).filter(User.id == user_id).first()
            if user_obj and user_obj.drive_token:
                from app.services.gdrive_service import GDriveService
                from pathlib import Path as _Path
                drive_svc = GDriveService(user_obj.drive_token)
                folders   = drive_svc.get_app_folders()
                fname     = f"model_{model.id}_{model.name.replace(' ','_')}{_Path(model.artifact_path).suffix}"
                res       = drive_svc.upload_file(model.artifact_path, folders["models"], fname)
                _log(f"Model synced to Google Drive: {res.get('webViewLink','')}")
        except Exception as e:
            _log(f"Drive sync skipped: {e}")

    except Exception as exc:
        logger.exception("Training thread failed: %s", exc)
        _jobs[job_id].update({"status": "failed", "error": str(exc),
                               "log": "\n".join(log_lines) + f"\n {exc}"})
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "failed"
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# ── Submit training ───────────────────────────────────────────────────────

@router.post("/train")
def train(
    req: TrainRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ds = (
        db.query(Dataset).join(Project)
        .filter(Dataset.id == req.dataset_id, Project.user_id == user.id)
        .first()
    )
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")

    use_kaggle = getattr(req, "use_kaggle", False)
    if use_kaggle:
        if not getattr(user, "kaggle_token", None):
            raise HTTPException(400, "Kaggle not connected. Go to Settings and connect your Kaggle API key.")
        from app.api.kaggle_training import start_kaggle_training, TrainRequest as KaggleReq
        kaggle_req = KaggleReq(
            dataset_id    = req.dataset_id,
            target_column = req.target_column,
            task_type     = req.task_type,
            model_name    = req.selected_model or "Auto (Best)",
        )
        return start_kaggle_training(kaggle_req, db=db, user=user)

    ds = (
        db.query(Dataset).join(Project)
        .filter(Dataset.id == req.dataset_id, Project.user_id == user.id)
        .first()
    )
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")

    try:
        ml_cost = settings.token_cost_per_train if req.selected_model in ("Auto (Best)", "auto", "") else 50
        token_service.spend(db, user, ml_cost, f"train_{ds.id}")
    except token_service.InsufficientTokens as exc:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, str(exc))

    ds.target_column = req.target_column
    db.commit()

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "log": "Queued...", "model_id": None, "error": None}

    t = threading.Thread(
        target=_run_training_thread,
        kwargs=dict(
            job_id=job_id, ds_path=ds.path, ds_id=ds.id,
            target_column=req.target_column, task_type=req.task_type,
            project_id=ds.project_id, user_id=user.id,
            selected_model=req.selected_model,
            output_format=req.output_format,
            hyperparams=req.hyperparams.model_dump() if req.hyperparams else None,
            preprocessing=req.preprocessing.model_dump() if req.preprocessing else None,
        ),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "running", "message": "Training started"}


@router.get("/available-models")
def available_models():
    from app.ml.automl_engine import ALL_ML_MODELS
    dl_models = {
        "tabular": ["Auto (Best)", "MLP (256-128-64)"],
        "image":   ["Auto (Best)", "ResNet-18", "ResNet-50", "MobileNetV2", "Custom CNN"],
        "text":    ["Auto (Best)", "BiLSTM", "LSTM"],
    }
    return {"ml": ALL_ML_MODELS, "dl": dl_models}


@router.post("/dl/train")
def train_dl(
    req: DLTrainRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ds = (
        db.query(Dataset).join(Project)
        .filter(Dataset.id == req.dataset_id, Project.user_id == user.id)
        .first()
    )
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")

    try:
        dl_costs = {
            "tabular": settings.token_cost_dl_tabular,
            "image":   settings.token_cost_dl_image,
            "text":    settings.token_cost_dl_text,
        }
        dl_cost = dl_costs.get(req.dl_type, settings.token_cost_dl_tabular)
        token_service.spend(db, user, dl_cost, f"dl_{ds.id}")
    except token_service.InsufficientTokens as exc:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, str(exc))

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "log": "Queued...", "model_id": None, "error": None}

    t = threading.Thread(
        target=_run_training_thread,
        args=(job_id, ds.path, ds.id, req.target_column or "", "auto",
              ds.project_id, user.id,
              req.dl_type, req.model_arch, req.epochs, req.img_size,
              req.text_column or ""),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "running", "message": "DL Training started"}


# ── Poll job status ───────────────────────────────────────────────────────

@router.get("/status/{job_id}")
def job_status(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

    resp = {"job_id": job_id, "status": job["status"], "log": job.get("log", "")}

    if job["status"] == "completed" and job.get("model_id"):
        mid = job["model_id"]
        m = (
            db.query(MLModel).join(Project)
            .filter(MLModel.id == mid, Project.user_id == user.id)
            .first()
        )
        if m:
            resp["model"] = {
                "id": m.id, "name": m.name, "framework": m.framework,
                "task_type": m.task_type, "accuracy": m.accuracy,
                "f1": m.f1, "target_column": m.target_column,
            }
            resp["metrics"]          = json.loads(m.metrics_json) if m.metrics_json else {}
            resp["confusion_matrix"] = json.loads(m.confusion_matrix_json) if m.confusion_matrix_json else None
            resp["feature_columns"]  = json.loads(m.feature_columns_json) if m.feature_columns_json else []

    elif job["status"] == "failed":
        resp["error"] = job.get("error", "Unknown error")

    return resp


# ── Models CRUD ───────────────────────────────────────────────────────────

@router.get("/models/project/{project_id}", response_model=list[ModelOut])
def list_models(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return db.query(MLModel).filter(MLModel.project_id == project_id).order_by(MLModel.created_at.desc()).all()


@router.get("/models/{model_id}")
def get_model(
    model_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    m = (
        db.query(MLModel).join(Project)
        .filter(MLModel.id == model_id, Project.user_id == user.id)
        .first()
    )
    if not m:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return {
        "id": m.id, "name": m.name, "framework": m.framework,
        "task_type": m.task_type, "target_column": m.target_column,
        "project_id": m.project_id,
        "feature_columns": json.loads(m.feature_columns_json) if m.feature_columns_json else [],
        "metrics": json.loads(m.metrics_json) if m.metrics_json else {},
        "confusion_matrix": json.loads(m.confusion_matrix_json) if m.confusion_matrix_json else None,
        "training_log": m.training_log,
        "created_at": m.created_at,
    }


@router.get("/models/{model_id}/download")
def download_model(
    model_id: int,
    format: str = "joblib",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from fastapi.responses import FileResponse
    m = (
        db.query(MLModel).join(Project)
        .filter(MLModel.id == model_id, Project.user_id == user.id)
        .first()
    )
    if not m or not Path(m.artifact_path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model artifact missing")

    artifact_path = Path(m.artifact_path)

    if format != "joblib" and artifact_path.suffix == ".joblib":
        try:
            from app.ml.automl_engine import convert_model
            feature_cols = json.loads(m.feature_columns_json) if m.feature_columns_json else []
            artifact_path = Path(convert_model(str(artifact_path), format, feature_cols))
        except Exception as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Conversion failed: {exc}")

    filename = f"model_{m.id}_{m.name.replace(' ', '_')}.{artifact_path.suffix.lstrip('.')}"
    return FileResponse(
        str(artifact_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    req: PredictRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    m = (
        db.query(MLModel).join(Project)
        .filter(MLModel.id == req.model_id, Project.user_id == user.id)
        .first()
    )
    if not m:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    try:
        token_service.spend(db, user, settings.token_cost_per_predict, f"predict_{m.id}")
    except token_service.InsufficientTokens as exc:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, str(exc))
    try:
        result = predict(m.artifact_path, req.features)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Prediction failed: {exc}")
    return PredictResponse(**result)


# ── Image prediction ──────────────────────────────────────────────────────

@router.post("/predict-image")
async def predict_image_endpoint(
    model_id: int = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Run inference on a single image using a trained image classification model."""

    # 1. Load model record and verify ownership
    m = (
        db.query(MLModel).join(Project)
        .filter(MLModel.id == model_id, Project.user_id == user.id)
        .first()
    )
    if not m:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")

    if m.task_type not in ("image_classification", "image"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Model task_type is '{m.task_type}', not an image classification model.",
        )

    artifact_path = Path(m.artifact_path)
    if not artifact_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model artifact file missing")

    # 2. Spend tokens
    try:
        token_service.spend(db, user, settings.token_cost_per_predict, f"predict_img_{m.id}")
    except token_service.InsufficientTokens as exc:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, str(exc))

    # 3. Load checkpoint — format: {'model_state': ..., 'meta': {...}}
    try:
        import io as _io
        import numpy as np
        import torch
        from PIL import Image as PILImage
        from torchvision import transforms
        from app.ml.dl_engine import get_image_model  # ← single source of truth

        contents   = await image.read()
        checkpoint = torch.load(str(artifact_path), map_location="cpu")
        meta        = checkpoint.get("meta", {})
        model_state = checkpoint.get("model_state", checkpoint)

        arch        = meta.get("arch", "resnet18")
        n_classes   = int(meta.get("n_classes", 2))
        class_names = meta.get("class_names", [str(i) for i in range(n_classes)])
        img_size    = int(meta.get("img_size", 128))

    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Failed to load model checkpoint: {exc}")

    # 4. Rebuild architecture using the shared helper in dl_engine
    try:
        net = get_image_model(arch, n_classes, img_size)
        net.load_state_dict(model_state)
        net.eval()

    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Failed to rebuild model architecture: {exc}")

    # 5. Preprocess image — same normalization used during training
    try:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img    = PILImage.open(_io.BytesIO(contents)).convert("RGB")
        tensor = tfm(img).unsqueeze(0)  # shape: (1, 3, H, W)

    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Image processing failed: {exc}")

    # 6. Run inference
    try:
        with torch.no_grad():
            logits = net(tensor)                      # shape: (1, n_classes)
            probs  = torch.softmax(logits, dim=1)     # shape: (1, n_classes)
            probs  = probs.squeeze(0).numpy()         # shape: (n_classes,) — squeeze batch dim only

        top_idx       = int(np.argmax(probs))
        top_class     = class_names[top_idx]
        confidence    = float(probs[top_idx])
        probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        return {
            "prediction":    top_class,
            "confidence":    confidence,
            "probabilities": probabilities,
        }

    except Exception as exc:
        logger.exception("Image inference failed: %s", exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Inference failed: {exc}")