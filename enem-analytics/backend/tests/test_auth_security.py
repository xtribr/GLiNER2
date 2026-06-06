"""Segurança da autenticação — is_admin nunca vem do user_metadata.

O user_metadata do Supabase é gravável pelo próprio usuário; derivar is_admin
dele permitiria escalonamento de privilégio. authenticate_with_token deve negar
acesso (None) quando não há linha em `profiles`, mesmo com is_admin no metadata.
"""

import sys


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
