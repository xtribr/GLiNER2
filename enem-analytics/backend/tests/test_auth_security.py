"""Segurança da autenticação — is_admin nunca vem do user_metadata.

O user_metadata do Supabase é gravável pelo próprio usuário; derivar is_admin
dele permitiria escalonamento de privilégio. authenticate_with_token deve negar
acesso (None) quando não há linha em `profiles`, mesmo com is_admin no metadata.
"""

import sys

import pytest
from fastapi import HTTPException


def test_token_sem_perfil_nega_acesso_mesmo_com_admin_no_metadata(monkeypatch):
    # Purga módulos de auth falsos (poluição do test_gliner_insights).
    for name in (
        "api.auth", "api.auth.supabase_service",
        "api.auth.supabase_dependencies", "api.auth.authorization",
    ):
        sys.modules.pop(name, None)

    import api.auth.supabase_service as svc

    # Token válido, com is_admin=True no metadata, mas SEM perfil em profiles.
    monkeypatch.setattr(
        svc,
        "verify_token",
        lambda _t: {
            "id": "user-sem-perfil",
            "email": "atacante@example.com",
            "user_metadata": {"is_admin": True, "codigo_inep": "99999999"},
        },
    )
    monkeypatch.setattr(svc, "get_user_profile", lambda _uid: None)

    result = svc.authenticate_with_token("token-falso")

    # Fail-closed: sem perfil → não autenticado (jamais admin via metadata).
    assert result is None


def test_admin_pode_acessar_qualquer_escola():
    from api.auth.authorization import ensure_school_access
    from api.auth.supabase_service import UserProfile

    admin = UserProfile(
        id="admin",
        email="admin@example.com",
        codigo_inep="",
        nome_escola="XTRI",
        is_admin=True,
    )

    assert ensure_school_access(admin, "31350664") is admin


def test_usuario_escola_pode_acessar_a_propria_escola():
    from api.auth.authorization import ensure_school_access
    from api.auth.supabase_service import UserProfile

    school_user = UserProfile(
        id="school-user",
        email="school@example.com",
        codigo_inep="31350664",
        nome_escola="Escola",
    )

    assert ensure_school_access(school_user, "31350664") is school_user


def test_usuario_escola_nao_pode_acessar_outra_escola():
    from api.auth.authorization import ensure_school_access
    from api.auth.supabase_service import UserProfile

    school_user = UserProfile(
        id="school-user",
        email="school@example.com",
        codigo_inep="31350664",
        nome_escola="Escola",
    )

    with pytest.raises(HTTPException) as exc_info:
        ensure_school_access(school_user, "23246847")

    assert exc_info.value.status_code == 403
