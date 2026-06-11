"""RAG Bot model — full config support."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base


class RAGBot(Base):
    __tablename__ = "rag_bots"
    id:            Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    user_id:       Mapped[int]      = mapped_column(Integer, ForeignKey("users.id"))

    # Identity
    name:          Mapped[str]      = mapped_column(String(120))
    description:   Mapped[str|None] = mapped_column(String(500), nullable=True)
    language:      Mapped[str]      = mapped_column(String(10), default="auto")  # auto|ar|en
    welcome_msg:   Mapped[str]      = mapped_column(String(500), default="Hello! How can I help you?")
    system_prompt: Mapped[str]      = mapped_column(Text, default="You are a helpful assistant.")
    fallback_msg:  Mapped[str]      = mapped_column(String(500), default="I don't have information about that in my documents.")

    # Storage
    docs_dir:      Mapped[str]      = mapped_column(String(512))
    vectordb_path: Mapped[str]      = mapped_column(String(512))
    n_documents:   Mapped[int]      = mapped_column(Integer, default=0)
    n_chunks:      Mapped[int]      = mapped_column(Integer, default=0)
    total_size_mb: Mapped[float]    = mapped_column(default=0.0)

    # Retrieval
    chunk_size:    Mapped[int]      = mapped_column(Integer, default=500)
    chunk_overlap: Mapped[int]      = mapped_column(Integer, default=50)
    top_k:         Mapped[int]      = mapped_column(Integer, default=3)
    similarity_threshold: Mapped[float] = mapped_column(default=0.3)

    # Generation
    temperature:   Mapped[float]    = mapped_column(default=0.7)
    max_tokens:    Mapped[int]      = mapped_column(Integer, default=1024)
    show_sources:  Mapped[bool]     = mapped_column(Boolean, default=False)
    allow_general: Mapped[bool]     = mapped_column(Boolean, default=False)
    output_format: Mapped[str]      = mapped_column(String(20), default="text")  # text|markdown|json

    status:        Mapped[str]      = mapped_column(String(20), default="empty")
    created_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", backref="rag_bots")


class RAGMessage(Base):
    __tablename__ = "rag_messages"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    bot_id:     Mapped[int]      = mapped_column(Integer, ForeignKey("rag_bots.id"))
    user_id:    Mapped[int]      = mapped_column(Integer, ForeignKey("users.id"))
    role:       Mapped[str]      = mapped_column(String(20))
    content:    Mapped[str]      = mapped_column(Text)
    sources:    Mapped[str|None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)