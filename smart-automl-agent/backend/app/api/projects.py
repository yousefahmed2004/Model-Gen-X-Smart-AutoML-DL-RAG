"""Projects CRUD."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User
from app.schemas.schemas import ProjectCreate, ProjectOut


router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return (
        db.query(Project)
        .filter(Project.user_id == user.id)
        .order_by(Project.updated_at.desc())
        .all()
    )


@router.post("", response_model=ProjectOut)
def create_project(
    data: ProjectCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    p = Project(user_id=user.id, **data.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    p = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return p


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    p = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    db.delete(p)
    db.commit()
