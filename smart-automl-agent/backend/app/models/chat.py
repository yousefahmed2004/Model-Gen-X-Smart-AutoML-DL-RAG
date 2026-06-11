"""Chat models — SQLAlchemy ORM."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base


class Chat(Base):
    __tablename__ = "chats"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    user_id:    Mapped[int]      = mapped_column(Integer, ForeignKey("users.id"))
    title:      Mapped[str]      = mapped_column(String(200), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    owner    = relationship("User", back_populates="chats")
    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    chat_id:    Mapped[int]      = mapped_column(Integer, ForeignKey("chats.id"))
    role:       Mapped[str]      = mapped_column(String(20))
    content:    Mapped[str]      = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")
