"""Database models"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from .config import Base


class User(Base):
    """User model for school authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    codigo_inep = Column(String(8), unique=True, nullable=False, index=True)
    nome_escola = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.email} ({self.codigo_inep})>"
