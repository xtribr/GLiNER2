"""Supabase authentication service for the FastAPI backend."""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Lazy-loaded Supabase client
_supabase_client = None


def get_supabase():
    """Get Supabase client (lazy-loaded singleton)."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError(
                "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY"
            )
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client


@dataclass
class UserProfile:
    """User profile data from Supabase."""
    id: str  # UUID from Supabase Auth
    email: str
    codigo_inep: str
    nome_escola: str
    is_admin: bool = False
    is_active: bool = True


@dataclass
class Lead:
    """Lead do cadastro público (origem='cadastro_publico')."""
    id: str
    nome_escola: str
    codigo_inep: str
    nome_contato: str
    cargo: str
    telefone: str
    email: str
    email_verified: bool
    created_at: str


def verify_token(access_token: str) -> Optional[dict]:
    """
    Verify a Supabase access token.

    Args:
        access_token: JWT token from Supabase Auth

    Returns:
        User data if valid, None otherwise
    """
    try:
        supabase = get_supabase()
        user_response = supabase.auth.get_user(access_token)

        if not user_response or not user_response.user:
            return None

        return {
            "id": user_response.user.id,
            "email": user_response.user.email,
            "user_metadata": user_response.user.user_metadata
        }
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        return None


def get_claims_from_token(access_token: str) -> dict:
    """Read JWT claims without verification for fallback metadata extraction."""
    try:
        return jwt.get_unverified_claims(access_token)
    except JWTError as e:
        logger.warning(f"Failed to decode access token claims: {e}")
        return {}


def get_auth_user_email(user_id: str) -> str:
    """Fetch the canonical email from Supabase Auth admin APIs."""
    try:
        supabase = get_supabase()
        user_response = supabase.auth.admin.get_user_by_id(user_id)
        if user_response and getattr(user_response, "user", None):
            return user_response.user.email or ""
    except Exception as e:
        logger.warning(f"Failed to fetch auth user email for {user_id}: {e}")
    return ""


def resolve_user_email(
    user_id: str,
    *,
    profile_email: str = "",
    token_email: str = "",
    access_token: Optional[str] = None,
) -> str:
    """Resolve email from the most reliable available sources."""
    if token_email:
        return token_email

    if access_token:
        claims = get_claims_from_token(access_token)
        claim_email = claims.get("email")
        if isinstance(claim_email, str) and claim_email:
            return claim_email

    if profile_email:
        return profile_email

    return get_auth_user_email(user_id)


def get_user_profile(user_id: str) -> Optional[UserProfile]:
    """
    Get user profile from profiles table.

    Args:
        user_id: Supabase Auth user UUID

    Returns:
        UserProfile if found, None otherwise
    """
    try:
        supabase = get_supabase()
        result = supabase.table("profiles").select("*").eq("id", user_id).single().execute()

        if not result.data:
            return None

        return UserProfile(
            id=result.data["id"],
            email=resolve_user_email(
                result.data["id"],
                profile_email=result.data.get("email", ""),
            ),
            codigo_inep=result.data["codigo_inep"],
            nome_escola=result.data["nome_escola"],
            is_admin=result.data.get("is_admin", False),
            is_active=result.data.get("is_active", True)
        )
    except Exception as e:
        logger.error(f"Failed to get profile for {user_id}: {e}")
        return None


def authenticate_with_token(access_token: str) -> Optional[UserProfile]:
    """
    Authenticate user with Supabase access token.

    Args:
        access_token: JWT from Supabase Auth

    Returns:
        UserProfile if valid, None otherwise
    """
    # Verify token with Supabase
    user_data = verify_token(access_token)
    if not user_data:
        return None

    # Get profile data
    profile = get_user_profile(user_data["id"])
    if profile:
        profile.email = resolve_user_email(
            user_data["id"],
            profile_email=profile.email,
            token_email=user_data.get("email", ""),
            access_token=access_token,
        )
        return profile

    # Sem linha em profiles → acesso NEGADO (fail-closed).
    # Segurança: is_admin e codigo_inep devem vir SEMPRE da tabela profiles
    # (server-controlled). O user_metadata é gravável pelo próprio usuário
    # (supabase.auth.updateUser), então derivar is_admin/codigo_inep dele
    # permitiria escalonamento de privilégio / acesso a outra escola. Todo
    # usuário legítimo tem perfil (criado no cadastro self-service e no
    # create_admin), então a ausência de perfil é tratada como não-autenticado.
    logger.warning(
        f"Token válido sem perfil em profiles — acesso negado: {user_data['id']}"
    )
    return None


def create_profile(
    user_id: str,
    codigo_inep: str,
    nome_escola: str,
    is_admin: bool = False
) -> Optional[UserProfile]:
    """
    Create a profile for a Supabase Auth user.

    Args:
        user_id: Supabase Auth user UUID
        codigo_inep: School INEP code
        nome_escola: School name
        is_admin: Admin flag

    Returns:
        Created UserProfile
    """
    try:
        supabase = get_supabase()
        result = supabase.table("profiles").insert({
            "id": user_id,
            "codigo_inep": codigo_inep,
            "nome_escola": nome_escola,
            "is_admin": is_admin
        }).execute()

        if not result.data:
            return None

        return UserProfile(
            id=result.data[0]["id"],
            email="",
            codigo_inep=result.data[0]["codigo_inep"],
            nome_escola=result.data[0]["nome_escola"],
            is_admin=result.data[0].get("is_admin", False)
        )
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        return None


def list_all_profiles(skip: int = 0, limit: int = 100) -> list[UserProfile]:
    """
    List all user profiles (admin function).

    Args:
        skip: Number of records to skip
        limit: Maximum records to return

    Returns:
        List of UserProfile objects
    """
    try:
        supabase = get_supabase()
        result = supabase.table("profiles").select("*").range(skip, skip + limit - 1).execute()

        return [
            UserProfile(
                id=row["id"],
                email=resolve_user_email(
                    row["id"],
                    profile_email=row.get("email", ""),
                ),
                codigo_inep=row["codigo_inep"],
                nome_escola=row["nome_escola"],
                is_admin=row.get("is_admin", False),
                is_active=row.get("is_active", True)
            )
            for row in result.data
        ]
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        return []


def list_leads(limit: int = 1000) -> list[Lead]:
    """
    Lista os leads do cadastro público (origem='cadastro_publico'),
    ativos e mais recentes primeiro. Admin-only (chamado atrás de get_current_admin).
    """
    try:
        supabase = get_supabase()
        result = (
            supabase.table("profiles")
            .select("*")
            .eq("origem", "cadastro_publico")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [
            Lead(
                id=row["id"],
                nome_escola=row.get("nome_escola", "") or "",
                codigo_inep=row.get("codigo_inep", "") or "",
                nome_contato=row.get("nome_contato", "") or "",
                cargo=row.get("cargo", "") or "",
                telefone=row.get("telefone", "") or "",
                email=resolve_user_email(row["id"], profile_email=row.get("email", "")),
                email_verified=row.get("email_verified", False),
                created_at=row.get("created_at", "") or "",
            )
            for row in result.data
        ]
    except Exception as e:
        logger.error(f"Failed to list leads: {e}")
        return []


def update_profile(user_id: str, **kwargs) -> Optional[UserProfile]:
    """
    Update a user profile.

    Args:
        user_id: Supabase Auth user UUID
        **kwargs: Fields to update

    Returns:
        Updated UserProfile
    """
    try:
        supabase = get_supabase()

        # Filter allowed fields
        allowed_fields = {"codigo_inep", "nome_escola", "is_admin", "is_active"}
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not update_data:
            return get_user_profile(user_id)

        result = supabase.table("profiles").update(update_data).eq("id", user_id).execute()

        if not result.data:
            return None

        return get_user_profile(user_id)
    except Exception as e:
        logger.error(f"Failed to update profile {user_id}: {e}")
        return None
