"""Pydantic schemas for authentication"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Base user schema"""
    codigo_inep: str
    nome_escola: str
    email: EmailStr


class UserCreate(UserBase):
    """Schema for creating a user"""
    password: str
    is_admin: bool = False


class UserUpdate(BaseModel):
    """Schema for updating a user"""
    nome_escola: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user response"""
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Schema for login request"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenData(BaseModel):
    """Schema for token payload"""
    user_id: int
    codigo_inep: str
    is_admin: bool
