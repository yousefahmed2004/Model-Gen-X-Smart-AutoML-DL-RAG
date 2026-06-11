"""Chat endpoints — talk to the Gemini AI brain."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.chat import Chat, ChatMessage
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.user import User
from app.schemas.schemas import ChatMessageIn, ChatMessageOut, ChatOut
from app.services.gemini_service import gemini

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _get_dataset_context(db: Session, dataset_id: int | None, user_id: int) -> dict | None:
    """Load dataset metadata for AI context."""
    ds = None
    if dataset_id:
        ds = (
            db.query(Dataset)
            .join(Project)
            .filter(Dataset.id == dataset_id, Project.user_id == user_id)
            .first()
        )
    else:
        # Auto-attach the most recently uploaded dataset
        ds = (
            db.query(Dataset)
            .join(Project)
            .filter(Project.user_id == user_id)
            .order_by(Dataset.id.desc())
            .first()
        )

    if not ds or not ds.columns_json:
        return None

    try:
        columns = json.loads(ds.columns_json)
    except Exception:
        return None

    return {
        "filename":  ds.filename,
        "n_rows":    ds.n_rows,
        "n_cols":    ds.n_cols,
        "columns":   columns,
        "dataset_id": ds.id,
        "target_column": ds.target_column,
    }


@router.get("", response_model=list[ChatOut])
def list_chats(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return (
        db.query(Chat)
        .filter(Chat.user_id == user.id)
        .order_by(Chat.updated_at.desc())
        .all()
    )


@router.post("", response_model=ChatOut)
def new_chat(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    c = Chat(user_id=user.id, title="New chat")
    db.add(c); db.commit(); db.refresh(c)
    return c


@router.get("/{chat_id}/messages", response_model=list[ChatMessageOut])
def list_messages(
    chat_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found")
    return chat.messages


@router.post("/send", response_model=ChatMessageOut)
def send_message(
    msg: ChatMessageIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Resolve or create chat
    if msg.chat_id:
        chat = db.query(Chat).filter(Chat.id == msg.chat_id, Chat.user_id == user.id).first()
        if not chat:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found")
    else:
        chat = Chat(user_id=user.id, title=msg.content[:60])
        db.add(chat); db.commit(); db.refresh(chat)

    # Save user turn
    user_turn = ChatMessage(chat_id=chat.id, role="user", content=msg.content)
    db.add(user_turn); db.commit()

    # Build Gemini history (exclude the message we just saved)
    history = [
        {
            "role": "user" if m.role == "user" else "model",
            "content": m.content,
        }
        for m in chat.messages
        if m.id != user_turn.id
    ]

    # Load dataset context — use explicit dataset_id or auto-attach latest
    ctx = _get_dataset_context(db, msg.dataset_id, user.id)

    # Call Gemini (or fallback)
    reply = gemini.chat(
        history=history,
        user_message=msg.content,
        dataset_context=ctx,
    )

    assistant_turn = ChatMessage(chat_id=chat.id, role="assistant", content=reply)
    db.add(assistant_turn); db.commit(); db.refresh(assistant_turn)
    return assistant_turn


@router.delete("/{chat_id}")
def delete_chat(chat_id: int,
                db:   Session = Depends(get_db),
                user: User    = Depends(get_current_user)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).delete()
    db.delete(chat)
    db.commit()
    return {"message": "Deleted"}