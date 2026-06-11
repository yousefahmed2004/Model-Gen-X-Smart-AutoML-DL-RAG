"""Token accounting service."""
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.user import User
from app.models.token_ledger import TokenLedger


class InsufficientTokens(Exception):
    """Raised when a user tries to spend more tokens than they have."""


def grant(db: Session, user: User, amount: int, reason: str) -> None:
    user.tokens += amount
    db.add(TokenLedger(user_id=user.id, delta=amount, reason=reason))
    db.commit()
    db.refresh(user)


def spend(db: Session, user: User, amount: int, reason: str) -> None:
    if user.tokens < amount:
        raise InsufficientTokens(
            f"Need {amount} tokens but only have {user.tokens}. "
            f"Upgrade to Pro for more."
        )
    user.tokens -= amount
    db.add(TokenLedger(user_id=user.id, delta=-amount, reason=reason))
    db.commit()
    db.refresh(user)


def signup_grant(db: Session, user: User) -> None:
    grant(db, user, settings.free_tokens_on_signup, "signup_bonus")
