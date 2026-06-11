"""Trained model artifact metadata."""
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)  # e.g. "Random Forest"
    framework: Mapped[str] = mapped_column(String(64), default="sklearn")
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)  # joblib pickle path
    task_type: Mapped[str] = mapped_column(String(64), default="classification")
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # full metrics dump
    feature_columns_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confusion_matrix_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="models")
