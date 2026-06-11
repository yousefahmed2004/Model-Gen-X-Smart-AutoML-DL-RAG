from __future__ import annotations
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

class Payment(Base):
    __tablename__ = "payments"
    id:           Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    user_id:      Mapped[int]      = mapped_column(Integer, ForeignKey("users.id"))
    amount:       Mapped[int]      = mapped_column(Integer)
    tokens:       Mapped[int]      = mapped_column(Integer)
    phone:        Mapped[str]      = mapped_column(String(20))
    screenshot:   Mapped[str|None] = mapped_column(String(512), nullable=True)
    status:       Mapped[str]      = mapped_column(String(20), default="pending")
    note:         Mapped[str|None] = mapped_column(String(255), nullable=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    approved_at:  Mapped[datetime|None] = mapped_column(DateTime, nullable=True)
    user = relationship("User", backref="payments")
