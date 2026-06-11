"""User account model."""
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    picture: Mapped[str | None] = mapped_column(String(512), nullable=True)
    google_sub: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    drive_token: Mapped[str | None] = mapped_column(String(2048), nullable=True)  # Google Drive access token
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_pro: Mapped[bool] = mapped_column(Boolean, default=False)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="owner", cascade="all, delete-orphan")
    ledger = relationship("TokenLedger", back_populates="user", cascade="all, delete-orphan")