"""Pydantic schemas exposed by the API."""
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, EmailStr, ConfigDict


# ---------- Auth ----------

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    name: str | None = None
    picture: str | None = None
    is_pro: bool = False
    tokens: int = 0
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class EmailLoginIn(BaseModel):
    """Local email/password login — used as a fallback when Google OAuth
    isn't configured (e.g. during the graduation demo offline test)."""
    email: EmailStr
    password: str


class EmailRegisterIn(EmailLoginIn):
    name: str | None = None


# ---------- Projects ----------

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    task_type: str = "auto"


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str | None
    task_type: str
    status: str
    created_at: datetime
    updated_at: datetime


# ---------- Datasets ----------

class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    filename: str
    file_type: str
    size_bytes: int
    n_rows: int | None
    n_cols: int | None
    target_column: str | None
    created_at: datetime


class DatasetPreview(BaseModel):
    dataset_id: int
    columns: list[dict[str, Any]]   # [{name, dtype, n_missing, sample}]
    head: list[dict[str, Any]]
    n_rows: int
    n_cols: int


# ---------- Training ----------

class HyperParams(BaseModel):
    # Random Forest / Extra Trees
    n_estimators:       Optional[int]   = None
    max_depth:          Optional[int]   = None
    min_samples_split:  Optional[int]   = None
    # KNN
    n_neighbors:        Optional[int]   = None
    metric:             Optional[str]   = None
    # SVM
    C:                  Optional[float] = None
    kernel:             Optional[str]   = None
    # Gradient Boosting
    learning_rate:      Optional[float] = None
    # AdaBoost
    n_estimators_ada:   Optional[int]   = None
    # MLP (DL)
    layers:             Optional[str]   = None   # e.g. "256->128->64"
    epochs:             Optional[int]   = None
    lr:                 Optional[float] = None


class PreprocessingConfig(BaseModel):
    missing_values: Optional[str] = "median"   # drop | mean | median | mode | auto
    scaling:        Optional[str] = "auto"      # standard | minmax | none | auto


class TrainRequest(BaseModel):
    dataset_id:     int
    target_column:  str
    task_type:      str  = "auto"
    selected_model: str  = "Auto (Best)"
    output_format:  str  = "joblib"
    use_kaggle:     bool = False
    hyperparams:    Optional[HyperParams]         = None
    preprocessing:  Optional[PreprocessingConfig] = None


class DLTrainRequest(BaseModel):
    dataset_id: int
    dl_type: str = "tabular"          # tabular | image | text
    target_column: str = ""           # for tabular + text
    text_column: str = ""             # for text classification
    model_arch: str = "resnet18"      # for image: resnet18 | mobilenet | cnn
    epochs: int = 20
    img_size: int = 128               # for image


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    framework: str
    task_type: str
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    target_column: str | None
    created_at: datetime


class TrainingResult(BaseModel):
    model: ModelOut
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]] | None = None
    feature_columns: list[str]
    log: str


# ---------- Predict ----------

class PredictRequest(BaseModel):
    model_id: int
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: Any
    probabilities: dict[str, float] | None = None


# ---------- Chat ----------

class ChatMessageIn(BaseModel):
    chat_id: int | None = None
    content: str
    dataset_id: int | None = None  # optional, for context


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    created_at: datetime


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    created_at: datetime
    updated_at: datetime