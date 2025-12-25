"""Authentication service with JWT and password hashing"""
import os
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database.models import User
from .schemas import TokenData

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(user: User) -> str:
    """Create a JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": str(user.id),
        "codigo_inep": user.codigo_inep,
        "is_admin": user.is_admin,
        "exp": expire
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        codigo_inep = payload.get("codigo_inep")
        is_admin = payload.get("is_admin", False)

        if user_id is None:
            return None

        return TokenData(user_id=user_id, codigo_inep=codigo_inep, is_admin=is_admin)
    except JWTError:
        return None


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get a user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_inep(db: Session, codigo_inep: str) -> Optional[User]:
    """Get a user by INEP code"""
    return db.query(User).filter(User.codigo_inep == codigo_inep).first()


def create_user(db: Session, codigo_inep: str, nome_escola: str, email: str,
                password: str, is_admin: bool = False) -> User:
    """Create a new user"""
    user = User(
        codigo_inep=codigo_inep,
        nome_escola=nome_escola,
        email=email,
        password_hash=hash_password(password),
        is_admin=is_admin
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
