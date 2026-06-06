"""Fase 1 — vitrine pública.

Garante que os endpoints de leitura (ranking/lista/busca/stats/resumo) são
acessíveis SEM token, enquanto os endpoints gated (compare, skills agregados)
continuam exigindo autenticação. A camada de dados é mockada — não toca Supabase.
"""

import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # test_gliner_insights stuba módulos de auth em sys.modules (via setdefault, sem
    # teardown e sem get_optional_user). Purga-os para importar os módulos REAIS,
    # garantindo o comportamento de auth correto (get_current_admin levanta 403).
    for _name in (
        "api.main", "api.routes.schools",
        "api.auth.supabase_dependencies", "api.auth.authorization", "api.auth",
    ):
        sys.modules.pop(_name, None)

    import api.main as main_module
    from data import supabase_store

    # Evita conexão real ao Supabase no startup (lifespan).
    monkeypatch.setattr(main_module, "init_supabase", lambda: None)

    # Mocka a camada de dados usada pelos handlers públicos.
    monkeypatch.setattr(
        supabase_store, "list_schools",
        lambda **kw: [{"codigo_inep": "11111111", "nome_escola": "Escola Teste", "uf": "RN"}],
    )
    monkeypatch.setattr(
        supabase_store, "get_top_schools",
        lambda **kw: [{"codigo_inep": "11111111", "nome_escola": "Escola Teste", "nota_media": 700.0, "ranking": 1}],
    )
    monkeypatch.setattr(
        supabase_store, "search_schools",
        lambda **kw: [{"codigo_inep": "11111111", "nome_escola": "Escola Teste"}],
    )
    monkeypatch.setattr(
        supabase_store, "get_school_detail",
        lambda inep: {
            "codigo_inep": inep,
            "nome_escola": "Escola Teste",
            "uf": "RN",
            "tipo_escola": "Privada",
            "historico": [{
                "ano": 2024, "nota_media": 700.0, "ranking_brasil": 1,
                "nota_cn": 700, "nota_ch": 700, "nota_lc": 700, "nota_mt": 700, "nota_redacao": 700,
            }],
        },
    )
    monkeypatch.setattr(supabase_store, "get_stats", lambda: {"total_schools": 1})

    return TestClient(main_module.app)


PUBLIC_PATHS = [
    "/api/stats",
    "/api/schools/?limit=3",
    "/api/schools/top?limit=3",
    "/api/schools/search?q=col",
    "/api/schools/11111111/summary",
]

GATED_PATHS = [
    "/api/schools/compare/11111111/22222222",
    "/api/schools/skills/all",
]


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_public_endpoint_sem_token_retorna_200(client, path):
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} deveria ser público (200), veio {resp.status_code}"


@pytest.mark.parametrize("path", GATED_PATHS)
def test_endpoint_gated_exige_auth(client, path):
    resp = client.get(path)
    assert resp.status_code in (401, 403), f"{path} deveria exigir auth, veio {resp.status_code}"
