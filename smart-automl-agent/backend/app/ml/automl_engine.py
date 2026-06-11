"""
AutoML training engine.

For the graduation-demo slice this uses scikit-learn directly (fast,
reliable, no heavy installs). The interface is designed so PyCaret can be
slotted in later: see `train_with_pycaret_stub` for the contract.

What this module does:
  * Detect task type (classification vs regression) from the target column
  * Build a preprocessing pipeline (impute + one-hot encode + scale)
  * Fit several candidate models, pick the best by CV score
  * Persist the winning pipeline with joblib
  * Return metrics + confusion matrix for the UI
"""
from __future__ import annotations

import io
import json
import logging
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.core.config import settings
from app.ml.dataset_loader import load_dataframe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
def detect_task_type(y: pd.Series) -> str:
    """Heuristic: small number of unique non-numeric or low-cardinality
    numeric values → classification; otherwise regression."""
    if y.dtype == "O" or str(y.dtype).startswith("category") or y.dtype == "bool":
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= max(20, int(len(y) ** 0.25)) and pd.api.types.is_integer_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
    return "classification"


# Maximum unique values a categorical column may have to get OHE treatment.
# Columns with more unique values get OrdinalEncoder instead, which is
# memory-safe and still informative for tree-based models.
OHE_MAX_CARDINALITY = 50

# If OHE would produce more total columns than this, fall back to OrdinalEncoder
# for ALL categorical columns (hard memory guard).
OHE_MAX_TOTAL_COLS = 500


# ---------------------------------------------------------------------- #
def _sanitize_X(X: pd.DataFrame) -> pd.DataFrame:
    """
    Additional cleaning after ID-drop:
      - Truncate string columns so they never generate >OHE_MAX_CARDINALITY dummies
        (keeps top-N most-frequent values, maps rest to '__other__')
      - Convert bool columns to int (some sklearn versions choke on bool dtype)
    """
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        n_unique = X[col].nunique(dropna=True)
        if n_unique > OHE_MAX_CARDINALITY:
            top = X[col].value_counts().nlargest(OHE_MAX_CARDINALITY).index
            X[col] = X[col].where(X[col].isin(top), other="__other__")
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    return X


def _build_preprocessor(X: pd.DataFrame, scaling: str = "auto", missing_strategy: str = "median") -> ColumnTransformer:
    """
    Memory-safe preprocessor:
      - Numeric  → median impute → StandardScaler
      - Low-card categoricals (≤ OHE_MAX_CARDINALITY unique) → OHE
      - High-card categoricals (> OHE_MAX_CARDINALITY unique) → OrdinalEncoder
        (tree models handle ordinal encoding just fine; linear models get a
        slightly worse representation but don't OOM)

    Total-column guard: if estimated OHE output columns would still exceed
    OHE_MAX_TOTAL_COLS we downgrade everything to OrdinalEncoder.
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Split cat cols by cardinality
    low_card = [c for c in cat_cols if X[c].nunique(dropna=True) <= OHE_MAX_CARDINALITY]
    high_card = [c for c in cat_cols if c not in low_card]

    # Estimate how many OHE columns we'd get
    estimated_ohe_cols = sum(X[c].nunique(dropna=True) for c in low_card)
    if estimated_ohe_cols > OHE_MAX_TOTAL_COLS:
        # Too many even after cardinality split → all ordinal
        high_card = cat_cols
        low_card = []

    # Dynamic imputer strategy
    imp_strat = missing_strategy if missing_strategy in ("mean", "median", "most_frequent") else "median"
    # Dynamic scaler
    if scaling == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        _scaler = MinMaxScaler()
    elif scaling == "none":
        from sklearn.preprocessing import FunctionTransformer
        _scaler = FunctionTransformer()
    else:  # standard or auto
        _scaler = StandardScaler(with_mean=False)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=imp_strat)),
        ("scaler", _scaler),
    ])
    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers: list = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if low_card:
        transformers.append(("cat_ohe", ohe_pipe, low_card))
    if high_card:
        transformers.append(("cat_ord", ord_pipe, high_card))

    if not transformers:
        # Fallback: passthrough everything numeric we can find
        transformers.append(("passthrough", "passthrough", num_cols or list(X.columns)))

    return ColumnTransformer(transformers, remainder="drop")


def _candidate_models(task_type: str, n_rows: int = 0,
                       hyperparams: dict | None = None) -> list[tuple[str, Any]]:
    """Build candidate models with optional hyperparameter overrides."""
    hp    = hyperparams or {}
    n_est = int(hp.get("n_estimators") or (100 if n_rows > 50_000 else 200))
    lr    = float(hp.get("learning_rate") or 0.1)
    depth = hp.get("max_depth")
    mss   = int(hp.get("min_samples_split") or 2)
    knn_k = int(hp.get("n_neighbors") or 5)
    knn_m = hp.get("metric") or "minkowski"
    svm_c = float(hp.get("C") or 1.0)
    svm_k = hp.get("kernel") or "rbf"

    if task_type == "classification":
        return [
            ("Random Forest",       RandomForestClassifier(n_estimators=n_est, max_depth=depth, min_samples_split=mss, random_state=42, n_jobs=-1)),
            ("Logistic Regression", LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)),
            ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=int(depth or 3), random_state=42)),
            ("Decision Tree",       DecisionTreeClassifier(max_depth=depth or 12, min_samples_split=mss, random_state=42)),
            ("Extra Trees",         ExtraTreesClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)),
            ("AdaBoost",            AdaBoostClassifier(n_estimators=50, random_state=42)),
            ("SVM",                 SVC(C=svm_c, kernel=svm_k, probability=True, random_state=42)),
            ("KNN",                 KNeighborsClassifier(n_neighbors=knn_k, metric=knn_m, n_jobs=-1)),
        ]
    else:
        return [
            ("Random Forest",     RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)),
            ("Ridge",             Ridge()),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=int(depth or 3), random_state=42)),
            ("Decision Tree",     DecisionTreeRegressor(max_depth=depth or 12, min_samples_split=mss, random_state=42)),
            ("Extra Trees",       ExtraTreesRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)),
            ("Lasso",             Lasso(max_iter=2000)),
            ("ElasticNet",        ElasticNet(max_iter=2000)),
            ("KNN",               KNeighborsRegressor(n_neighbors=knn_k, metric=knn_m, n_jobs=-1)),
        ]


def train_from_file(
    dataset_path: str | Path,
    target_column: str,
    task_type: str = "auto",
    selected_model: str = "Auto (Best)",
    hyperparams: dict | None = None,
    preprocessing: dict | None = None,
) -> dict[str, Any]:
    """Run the full AutoML cycle and return a results dict.

    Returns keys:
      best_model_name, framework, task_type, metrics, confusion_matrix,
      feature_columns, target_column, artifact_path, log
    """
    log_buf = io.StringIO()
    t0 = time.time()

    def _log(msg: str) -> None:
        print(msg, file=log_buf)
        logger.info(msg)

    _log(f"Loading dataset: {dataset_path}")
    df = load_dataframe(dataset_path)
    _log(f"Loaded {len(df)} rows × {df.shape[1]} cols")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")

    # Drop rows where the target is missing — we can't learn from them.
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("All rows have missing target after dropna.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Drop ID-like columns: unique-valued strings of high cardinality.
    drop_ids: list[str] = []
    for c in X.columns:
        if X[c].dtype == "O" and X[c].nunique() == len(X):
            drop_ids.append(c)
    if drop_ids:
        _log(f"Dropping high-cardinality ID columns: {drop_ids}")
        X = X.drop(columns=drop_ids)

    # Sanitize: cap cardinality, convert bools → prevent OHE memory explosion
    X = _sanitize_X(X)
    _log(f"After sanitize: {X.shape[1]} feature cols, dtypes: "
         f"{dict(X.dtypes.astype(str).value_counts())}")

    feature_columns = X.columns.tolist()
    if not feature_columns:
        raise ValueError("No usable feature columns after preprocessing.")

    # Decide task type
    detected = detect_task_type(y) if task_type == "auto" else task_type

    # ── Target sanity checks ──────────────────────────────────────────────
    n_unique_y = y.nunique(dropna=True)

    # Case 1: user forced classification but target has too many classes →
    # accuracy will be near random. Warn and auto-switch to regression if numeric.
    if detected == "classification" and n_unique_y > 50:
        if pd.api.types.is_numeric_dtype(y):
            _log(
                f"⚠  Target '{target_column}' has {n_unique_y} unique numeric values — "
                f"switching to regression (too many classes for classification)."
            )
            detected = "regression"
        else:
            _log(
                f"⚠  Target '{target_column}' has {n_unique_y} unique text values. "
                f"This is a very high-cardinality classification problem — accuracy may be low. "
                f"Consider grouping values or picking a different target column."
            )

    # Case 2: regression but only 2 unique values → treat as binary classification
    if detected == "regression" and n_unique_y <= 2:
        _log(
            f"⚠  Target '{target_column}' has only {n_unique_y} unique values — "
            f"switching to classification."
        )
        detected = "classification"

    _log(f"Task type: {detected}  |  target unique values: {n_unique_y}")
    # ─────────────────────────────────────────────────────────────────────

    # Encode classification labels for consistency in the report
    class_names: list[str] | None = None
    if detected == "classification":
        class_names = sorted([str(c) for c in y.dropna().unique()])

    # Apply preprocessing config from Guided Mode
    pp = preprocessing or {}
    missing_strategy = pp.get("missing_values", "median")
    scaling_strategy = pp.get("scaling", "auto")
    if missing_strategy == "drop":
        before = len(X)
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        _log(f"Dropped {before - len(X)} rows with missing values. Remaining: {len(X)}")
        missing_strategy = "median"  # for the pipeline
    if missing_strategy == "mode":
        missing_strategy = "most_frequent"
    _log(f"Preprocessing: missing={missing_strategy}, scaling={scaling_strategy}")
    pre = _build_preprocessor(X, scaling=scaling_strategy, missing_strategy=missing_strategy)

    best_name = ""
    best_score = -np.inf
    best_pipe: Pipeline | None = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if detected == "classification" and y.nunique() > 1 else None,
    )

    scoring = "accuracy" if detected == "classification" else "r2"
    cv_folds = 3 if len(X_train) >= 50 else 2

    candidates = _candidate_models(detected, n_rows=len(X_train), hyperparams=hyperparams)

    # Filter to selected model if user chose one
    if selected_model and selected_model not in ("Auto (Best)", "auto", ""):
        candidates = [(n, m) for n, m in candidates if n == selected_model]
        if not candidates:
            raise ValueError(f"Model '{selected_model}' not found.")

    total = len(candidates)
    _log(f"🧪 Testing {total} model(s) with {cv_folds}-fold cross-validation…")

    for idx, (name, model) in enumerate(candidates, 1):
        _log(f"⏳ [{idx}/{total}] Testing {name}…")
        pipe = Pipeline([("pre", pre), ("model", model)])
        try:
            with redirect_stdout(io.StringIO()):
                scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
            mean_score = float(scores.mean())
            _log(f"  ✓ {name:<22} cv-{scoring} = {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_pipe = pipe
                _log(f"  ⭐ New best: {name} ({mean_score:.4f})")
        except Exception as exc:
            _log(f"  ✗ {name:<22} FAILED: {exc}")

    if best_pipe is None:
        raise RuntimeError("All candidate models failed during cross-validation.")

    _log(f"🏆 Best model: {best_name} (cv {scoring}={best_score:.4f})")
    _log(f"🏋️ Fitting {best_name} on full training set…")
    best_pipe.fit(X_train, y_train)
    _log(f"📊 Evaluating on test set…")
    y_pred = best_pipe.predict(X_test)

    # ---------- metrics ----------
    metrics: dict[str, Any] = {"task_type": detected, "best_model": best_name}
    cm: list[list[int]] | None = None

    if detected == "classification":
        avg = "binary" if y.nunique() == 2 else "weighted"
        metrics.update({
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        })
        labels = sorted(pd.unique(pd.concat([y_test.astype(str), pd.Series(y_pred).astype(str)])))
        cm_arr = confusion_matrix(y_test.astype(str), pd.Series(y_pred).astype(str), labels=labels)
        cm = cm_arr.tolist()
        metrics["confusion_labels"] = labels
    else:
        metrics.update({
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(root_mean_squared_error(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        })

    # ---------- persist artifact ----------
    settings.model_path.mkdir(parents=True, exist_ok=True)
    artifact_name = f"model_{uuid.uuid4().hex[:12]}.joblib"
    artifact_path = settings.model_path / artifact_name
    joblib.dump(
        {
            "pipeline": best_pipe,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "task_type": detected,
            "class_names": class_names,
            "best_model_name": best_name,
        },
        artifact_path,
    )
    _log(f"Saved artifact → {artifact_path}")

    elapsed = time.time() - t0
    _log(f"Done in {elapsed:.2f}s")

    return {
        "best_model_name": best_name,
        "framework": "scikit-learn",
        "task_type": detected,
        "metrics": metrics,
        "confusion_matrix": cm,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "artifact_path": str(artifact_path),
        "log": log_buf.getvalue(),
    }


def convert_model(artifact_path: str | Path, output_format: str,
                  feature_columns: list[str]) -> str:
    """Convert saved joblib model to requested format. Returns new path."""
    artifact_path = Path(artifact_path)
    if output_format == "joblib":
        return str(artifact_path)

    bundle = joblib.load(artifact_path)
    pipe   = bundle["pipeline"]
    out    = artifact_path.with_suffix(f".{output_format}")

    if output_format == "pkl":
        import pickle
        with open(out, "wb") as f:
            pickle.dump(bundle, f)

    elif output_format == "onnx":
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            n_features = len(feature_columns)
            init_types = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(pipe, initial_types=init_types)
            with open(out, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except ImportError:
            raise RuntimeError("pip install skl2onnx onnx")

    elif output_format == "pt":
        try:
            import torch
            import torch.nn as nn
            import numpy as np

            # Wrap sklearn model in a simple PyTorch wrapper
            class SklearnWrapper(nn.Module):
                def __init__(self, sklearn_pipeline, n_features):
                    super().__init__()
                    self.pipeline  = sklearn_pipeline
                    self.n_features = n_features
                def forward(self, x):
                    arr  = x.numpy()
                    pred = self.pipeline.predict(arr)
                    return torch.tensor(pred)

            wrapper = SklearnWrapper(pipe, len(feature_columns))
            torch.save({
                "model": wrapper,
                "pipeline": pipe,
                "feature_columns": feature_columns,
                "task_type": bundle.get("task_type"),
                "class_names": bundle.get("class_names"),
            }, out)
        except ImportError:
            raise RuntimeError("pip install torch")
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    return str(out)


# ---------------------------------------------------------------------- #
def predict(model_artifact_path: str, features: dict[str, Any]) -> dict[str, Any]:
    """Load a saved pipeline and predict a single row."""
    bundle = joblib.load(model_artifact_path)
    pipe = bundle["pipeline"]
    feature_columns: list[str] = bundle["feature_columns"]

    row = {c: features.get(c) for c in feature_columns}
    X = pd.DataFrame([row])
    yhat = pipe.predict(X)[0]

    probs: dict[str, float] | None = None
    if bundle.get("task_type") == "classification" and hasattr(pipe.named_steps["model"], "predict_proba"):
        try:
            proba = pipe.predict_proba(X)[0]
            classes = [str(c) for c in pipe.classes_]
            probs = {c: float(p) for c, p in zip(classes, proba)}
        except Exception:
            probs = None

    # Make prediction JSON-safe
    if isinstance(yhat, (np.integer,)):
        yhat = int(yhat)
    elif isinstance(yhat, (np.floating,)):
        yhat = float(yhat)
    else:
        yhat = str(yhat)
    return {"prediction": yhat, "probabilities": probs}


# ---------------------------------------------------------------------- #
def train_with_pycaret_stub(*args, **kwargs):  # pragma: no cover
    """
    Drop-in replacement contract:
        Inputs   : dataset_path, target_column, task_type
        Outputs  : same dict as `train_from_file`

    PyCaret takes 60–90s to import; for the graduation demo we ship the
    sklearn engine and leave this stubbed so a future engineer can swap it
    in by changing one line in `app/api/training.py`.
    """
    raise NotImplementedError("PyCaret backend not enabled in demo mode.")