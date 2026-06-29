"""Admin routes for user management - Supabase version"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from api.auth.supabase_dependencies import get_current_admin, UserProfile
from api.auth.supabase_service import get_supabase, get_user_profile, list_all_profiles, list_leads, update_profile

router = APIRouter(prefix="/api/admin", tags=["Admin"])


class UserCreate(BaseModel):
    codigo_inep: str
    nome_escola: str
    email: EmailStr
    password: str
    is_admin: bool = False


class UserUpdate(BaseModel):
    nome_escola: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = None  # For password changes


@router.get("/users")
async def list_users(
    skip: int = 0,
    limit: Optional[int] = None,
    _: UserProfile = Depends(get_current_admin)
):
    """
    List all registered schools/users.

    Returns the full set by default (limit omitted) so every registered school —
    including those captured as public leads — appears. Pass `limit` for a
    single bounded page. Requires admin privileges.
    """
    profiles = list_all_profiles(skip=skip, limit=limit)
    return [
        {
            "id": p.id,
            "codigo_inep": p.codigo_inep,
            "nome_escola": p.nome_escola,
            "email": p.email,
            "is_admin": p.is_admin,
            "is_active": p.is_active,
        }
        for p in profiles
    ]


@router.get("/leads")
async def list_leads_endpoint(_: UserProfile = Depends(get_current_admin)):
    """
    Lista os leads do cadastro público (origem='cadastro_publico'),
    mais recentes primeiro, com os dados de contato. Admin only.
    """
    leads = list_leads()
    return [
        {
            "id": lead.id,
            "nome_escola": lead.nome_escola,
            "codigo_inep": lead.codigo_inep,
            "nome_contato": lead.nome_contato,
            "cargo": lead.cargo,
            "telefone": lead.telefone,
            "email": lead.email,
            "email_verified": lead.email_verified,
            "created_at": lead.created_at,
        }
        for lead in leads
    ]


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    _: UserProfile = Depends(get_current_admin)
):
    """
    Get a specific user by ID.
    Requires admin privileges.
    """
    profile = get_user_profile(user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )

    return {
        "id": profile.id,
        "codigo_inep": profile.codigo_inep,
        "nome_escola": profile.nome_escola,
        "email": profile.email,
        "is_admin": profile.is_admin,
        "is_active": profile.is_active,
    }


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    _: UserProfile = Depends(get_current_admin)
):
    """
    Create a new school/user.
    Requires admin privileges.
    """
    supabase = get_supabase()

    # Check if INEP already exists
    existing = supabase.table("profiles").select("id").eq("codigo_inep", user_data.codigo_inep).execute()
    if existing.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código INEP já cadastrado"
        )

    # Create user in Supabase Auth
    try:
        auth_response = supabase.auth.admin.create_user({
            "email": user_data.email,
            "password": user_data.password,
            "email_confirm": True,
            "user_metadata": {
                "codigo_inep": user_data.codigo_inep,
                "nome_escola": user_data.nome_escola
            }
        })

        if not auth_response.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Erro ao criar usuário"
            )

        user_id = auth_response.user.id

        # Create profile
        profile_result = supabase.table("profiles").insert({
            "id": user_id,
            "codigo_inep": user_data.codigo_inep,
            "nome_escola": user_data.nome_escola,
            "is_admin": user_data.is_admin
        }).execute()

        created_profile = get_user_profile(user_id)
        if created_profile:
            return {
                "id": created_profile.id,
                "codigo_inep": created_profile.codigo_inep,
                "nome_escola": created_profile.nome_escola,
                "email": created_profile.email,
                "is_admin": created_profile.is_admin,
                "is_active": created_profile.is_active,
            }

        return {"id": user_id, "email": user_data.email}

    except Exception as e:
        if "already registered" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email já cadastrado"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/users/{user_id}")
async def update_user_endpoint(
    user_id: str,
    user_data: UserUpdate,
    _: UserProfile = Depends(get_current_admin)
):
    """
    Update a school/user.
    Requires admin privileges.
    """
    supabase = get_supabase()

    # Update password in Supabase Auth if provided
    if user_data.password:
        try:
            supabase.auth.admin.update_user_by_id(
                user_id,
                {"password": user_data.password}
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Erro ao atualizar senha: {str(e)}"
            )

    # Update profile fields
    updated = update_profile(
        user_id,
        nome_escola=user_data.nome_escola,
        is_active=user_data.is_active,
        is_admin=user_data.is_admin
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )

    return {
        "id": updated.id,
        "codigo_inep": updated.codigo_inep,
        "nome_escola": updated.nome_escola,
        "email": updated.email,
        "is_admin": updated.is_admin,
        "is_active": updated.is_active,
    }


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: UserProfile = Depends(get_current_admin)
):
    """
    Remove DEFINITIVAMENTE a escola/usuário: apaga o perfil e o usuário de auth,
    liberando o código INEP e o e-mail para um novo cadastro. Requer admin.
    (Para apenas desativar sem apagar, use PUT /users/{id} com is_active=false.)
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não é possível excluir seu próprio usuário"
        )

    if not get_user_profile(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado"
        )

    supabase = get_supabase()
    supabase.table("profiles").delete().eq("id", user_id).execute()
    try:
        supabase.auth.admin.delete_user(user_id)
    except Exception:  # noqa: BLE001 - perfil já removido; usuário de auth pode não existir
        pass

    return None


@router.get("/stats")
async def get_admin_stats(
    _: UserProfile = Depends(get_current_admin)
):
    """
    Get admin statistics.
    Requires admin privileges.
    """
    supabase = get_supabase()

    # Use exact COUNT head-queries instead of pulling rows: PostgREST silently
    # caps a row response at ~1000, which would undercount once profiles exceed
    # that (the same truncation class as the leads-vs-users bug). Counts are
    # accurate at any table size and pull no rows.
    def _count(query) -> int:
        return query.limit(1).execute().count or 0

    total_users = _count(supabase.table("profiles").select("id", count="exact"))
    active_users = _count(
        supabase.table("profiles").select("id", count="exact").eq("is_active", True)
    )
    admin_users = _count(
        supabase.table("profiles").select("id", count="exact").eq("is_admin", True)
    )

    return {
        "total_users": total_users,
        "active_users": active_users,
        "inactive_users": total_users - active_users,
        "admin_users": admin_users
    }
