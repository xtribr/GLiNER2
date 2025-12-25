"""Authentication dependencies for FastAPI"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database.config import get_db
from database.models import User
from .service import decode_token, get_user_by_id

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    token_data = decode_token(token)

    if token_data is None:
        raise credentials_exception

    user = get_user_by_id(db, token_data.user_id)

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuário desativado"
        )

    return user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get the current user and verify admin status"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso restrito a administradores"
        )
    return current_user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> User | None:
    """Get current user if authenticated, None otherwise"""
    if credentials is None:
        return None

    token_data = decode_token(credentials.credentials)
    if token_data is None:
        return None

    user = get_user_by_id(db, token_data.user_id)
    if user is None or not user.is_active:
        return None

    return user
