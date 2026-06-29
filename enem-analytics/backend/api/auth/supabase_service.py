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


# PostgREST caps a single response (Supabase default max-rows ≈ 1000), so the
# admin user list must page through ALL profiles in batches. Otherwise schools
# beyond the first page silently vanish from "Usuários" while still appearing in
# "Leads" (which reads the same table with its own 1000-row, date-ordered query).
# Order by (created_at desc, id) for a TOTAL order so batch windows never skip or
# duplicate rows that share a created_at timestamp at a page boundary.
_PROFILE_PAGE_SIZE = 1000
_PROFILE_FETCH_ALL_CAP = 20000  # safety stop; logged if hit (never silently truncate)


def _build_auth_email_map() -> dict:
    """Bulk {user_id: email} from Supabase Auth, paginated.

    Resolves emails for the whole admin list in a handful of requests instead of
    one admin HTTP call per row. `profiles.email` is not populated at signup, so
    a per-row fallback to the Auth admin API turns a full-table user list into an
    N+1 of blocking calls (slow / gateway-timeout / rate-limit). Best-effort:
    returns {} on error, and callers fall back to profiles.email or "".
    """
    emails: dict = {}
    try:
        supabase = get_supabase()
        page = 1
        per_page = 1000
        while True:
            users = supabase.auth.admin.list_users(page=page, per_page=per_page) or []
            for u in users:
                uid = getattr(u, "id", None)
                email = getattr(u, "email", None)
                if uid and email:
                    emails[uid] = email
            if not users:  # empty page → exhausted (robust to server per_page caps)
                break
            page += 1
            if page > 100:  # safety: ~100k auth users
                logger.warning("_build_auth_email_map stopped at page 100; some emails unresolved")
                break
    except Exception as e:  # noqa: BLE001 - email map is best-effort
        logger.warning(f"Failed to build auth email map: {e}")
    return emails


def list_all_profiles(skip: int = 0, limit: Optional[int] = None) -> list[UserProfile]:
    """
    List user profiles (admin function).

    Args:
        skip: Number of records to skip.
        limit: Maximum records to return. When None (default), returns the FULL
            table, paging through it in batches — the per-request row cap would
            otherwise truncate the list and hide schools that exist as leads.

    Returns:
        List of UserProfile objects, ordered by (created_at desc, id).
    """
    try:
        supabase = get_supabase()

        def _page(start: int, count: int) -> list[dict]:
            return (
                supabase.table("profiles")
                .select("*")
                .order("created_at", desc=True)
                .order("id")
                .range(start, start + count - 1)
                .execute()
                .data
                or []
            )

        rows: list[dict] = []
        if limit is not None:
            rows = _page(skip, limit)
        else:
            offset = skip
            while True:
                page = _page(offset, _PROFILE_PAGE_SIZE)
                rows.extend(page)
                offset += len(page)
                if len(page) < _PROFILE_PAGE_SIZE:
                    break  # short page → last page
                if offset >= _PROFILE_FETCH_ALL_CAP:
                    # Only warn if rows genuinely remain beyond the cap (avoids a
                    # false positive when the table size is an exact multiple).
                    if _page(offset, 1):
                        logger.warning(
                            "list_all_profiles hit safety cap of %s rows; some profiles "
                            "were not returned. Raise _PROFILE_FETCH_ALL_CAP.",
                            _PROFILE_FETCH_ALL_CAP,
                        )
                    break

        # Resolve emails in ONE bulk pass — never one Auth call per row.
        email_map = _build_auth_email_map()

        profiles: list[UserProfile] = []
        for row in rows:
            # One malformed row must not blank the entire admin list (was the
            # prior all-or-nothing failure mode).
            try:
                uid = row["id"]
                profiles.append(
                    UserProfile(
                        id=uid,
                        email=(row.get("email") or "") or email_map.get(uid, ""),
                        codigo_inep=row["codigo_inep"],
                        nome_escola=row["nome_escola"],
                        is_admin=row.get("is_admin", False),
                        is_active=row.get("is_active", True),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Skipping malformed profile row %s: %s", row.get("id"), exc)
        return profiles
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

        # Never truncate leads silently — if the cap is reached, older leads are
        # being dropped from the panel and the limit must be raised / paginated.
        if len(result.data or []) >= limit:
            logger.warning("list_leads hit the %s-row cap; older leads may be omitted.", limit)

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
