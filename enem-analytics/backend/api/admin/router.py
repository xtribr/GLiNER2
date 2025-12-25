"""Admin routes for user management"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.config import get_db
from database.models import User
from api.auth.schemas import UserCreate, UserUpdate, UserResponse
from api.auth.service import hash_password, get_user_by_email, get_user_by_inep
from api.auth.dependencies import get_current_admin

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin)
):
    """
    List all registered schools/users.

    Requires admin privileges.
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin)
):
    """
    Get a specific user by ID.

    Requires admin privileges.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )
    return user


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin)
):
    """
    Create a new school/user.

    Requires admin privileges.
    """
    # Check if email already exists
    if get_user_by_email(db, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email já cadastrado"
        )

    # Check if INEP already exists
    if get_user_by_inep(db, user_data.codigo_inep):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código INEP já cadastrado"
        )

    # Create user
    user = User(
        codigo_inep=user_data.codigo_inep,
        nome_escola=user_data.nome_escola,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        is_admin=user_data.is_admin
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin)
):
    """
    Update a school/user.

    Requires admin privileges.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )

    # Update fields if provided
    if user_data.nome_escola is not None:
        user.nome_escola = user_data.nome_escola

    if user_data.email is not None:
        existing = get_user_by_email(db, user_data.email)
        if existing and existing.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email já cadastrado por outro usuário"
            )
        user.email = user_data.email

    if user_data.password is not None:
        user.password_hash = hash_password(user_data.password)

    if user_data.is_active is not None:
        user.is_active = user_data.is_active

    if user_data.is_admin is not None:
        user.is_admin = user_data.is_admin

    db.commit()
    db.refresh(user)

    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    """
    Deactivate a school/user (soft delete).

    Requires admin privileges.
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não é possível desativar seu próprio usuário"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )

    user.is_active = False
    db.commit()

    return None


@router.get("/stats")
async def get_admin_stats(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin)
):
    """
    Get admin statistics.

    Requires admin privileges.
    """
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    admin_users = db.query(User).filter(User.is_admin == True).count()

    return {
        "total_users": total_users,
        "active_users": active_users,
        "inactive_users": total_users - active_users,
        "admin_users": admin_users
    }
