"""Uploaded dataset metadata."""
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, BigInteger, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), default="csv")  # csv | xlsx | parquet | image_zip
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    n_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_cols: Mapped[int | None] = mapped_column(Integer, nullable=True)
    columns_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list of column metadata
    target_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="datasets")
