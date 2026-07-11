"""Fase 3 — cadastro self-service (POST /api/auth/signup).

Testa o comportamento do endpoint com o Supabase mockado (não toca DB/Auth real):
honeypot, aceite de termos, e-mail descartável, escola duplicada e o caminho feliz
(perfil criado com is_admin=False e origem='cadastro_publico').
"""

import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


# --------- Fake Supabase client (mínimo necessário pro signup) ---------
class _Resp:
    def __init__(self, data):
        self.data = data


class _Tbl:
    def __init__(self, parent):
        self.parent = parent
        self._op = "select"
        self._row = None

    def select(self, *a, **k):
        self._op = "select"
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def insert(self, row):
        self._op = "insert"
        self._row = row
        return self

    def update(self, row):
        self._op = "update"
        self._row = row
        return self

    def delete(self):
        self._op = "delete"
        return self

    def execute(self):
        if self._op == "select":
            return _Resp(self.parent.existing)
        if self._op == "insert":
            self.parent.inserts.append(self._row)
            return _Resp([self._row])
        if self._op == "delete":
            self.parent.profile_deleted = True
            return _Resp(self.parent.existing)
        return _Resp([{}])


class _Admin:
    def __init__(self, parent):
        self.parent = parent

    def create_user(self, payload):
        self.parent.created = payload
        return SimpleNamespace(user=SimpleNamespace(id="uuid-test-1234"))

    def delete_user(self, user_id):
        self.parent.deleted = user_id


class _Auth:
    def __init__(self, parent):
        self.admin = _Admin(parent)


class FakeSupabase:
    def __init__(self, existing=None):
        self.existing = existing or []
        self.inserts = []
        self.created = None
        self.deleted = None
        self.profile_deleted = False
        self.auth = _Auth(self)

    def table(self, _name):
        return _Tbl(self)


VALID = {
    "codigo_inep": "31350664",
    "nome_escola": "Colegio Bernoulli",
    "nome_contato": "Maria da Silva",
    "cargo": "Diretor(a)",
    "email": "maria.teste@gmail.com",
    "telefone": "(31) 98888-7777",
    "senha": "senha123",
    "aceite_termos": True,
    "website": "",
}


@pytest.fixture
def ctx(monkeypatch):
    # Purga módulos de auth falsos (poluição do test_gliner_insights) para
    # carregar os reais, igual ao test_public_endpoints.
    for name in (
        "api.main", "api.routes.schools", "api.auth.signup",
        "api.auth.supabase_dependencies", "api.auth.authorization", "api.auth",
    ):
        sys.modules.pop(name, None)

    import api.main as main_module
    import api.auth.signup as signup_module

    monkeypatch.setattr(main_module, "init_supabase", lambda: None)
    monkeypatch.setattr(signup_module, "_dns_resolver", None)  # pula checagem de MX (sem rede)
    monkeypatch.setattr(signup_module, "_send_welcome_email", lambda *a, **k: None)
    signup_module._signup_hits.clear()

    return TestClient(main_module.app), signup_module, monkeypatch


def test_signup_sucesso(ctx):
    client, signup_module, mp = ctx
    fake = FakeSupabase(existing=[])
    mp.setattr(signup_module, "get_supabase", lambda: fake)

    resp = client.post("/api/auth/signup", json=VALID)

    assert resp.status_code == 201, resp.text
    assert resp.json()["success"] is True
    assert fake.created is not None  # create_user no Supabase Auth foi chamado
    assert fake.inserts, "perfil deveria ter sido inserido"
    profile = fake.inserts[0]
    assert profile["is_admin"] is False  # nunca admin pelo cadastro público
    assert profile["origem"] == "cadastro_publico"
    assert profile["codigo_inep"] == "31350664"


def test_signup_honeypot_rejeitado(ctx):
    client, signup_module, mp = ctx
    mp.setattr(signup_module, "get_supabase", lambda: FakeSupabase())
    payload = {**VALID, "website": "http://spam"}
    resp = client.post("/api/auth/signup", json=payload)
    assert resp.status_code == 400


def test_signup_termos_obrigatorios(ctx):
    client, signup_module, mp = ctx
    mp.setattr(signup_module, "get_supabase", lambda: FakeSupabase())
    payload = {**VALID, "aceite_termos": False}
    resp = client.post("/api/auth/signup", json=payload)
    assert resp.status_code == 400


def test_signup_email_descartavel(ctx):
    client, signup_module, mp = ctx
    mp.setattr(signup_module, "get_supabase", lambda: FakeSupabase())
    payload = {**VALID, "email": "fulano@mailinator.com"}
    resp = client.post("/api/auth/signup", json=payload)
    assert resp.status_code == 400


def test_signup_escola_ativa_duplicada(ctx):
    client, signup_module, mp = ctx
    fake = FakeSupabase(existing=[{"id": "ja-existe", "is_active": True}])
    mp.setattr(signup_module, "get_supabase", lambda: fake)
    resp = client.post("/api/auth/signup", json=VALID)
    assert resp.status_code == 409
    assert fake.created is None  # não cria auth user se a escola já existe


def test_signup_escola_inativa_e_limpa_e_recadastra(ctx):
    client, signup_module, mp = ctx
    fake = FakeSupabase(existing=[{"id": "ja-existe", "is_active": False}])
    mp.setattr(signup_module, "get_supabase", lambda: fake)

    resp = client.post("/api/auth/signup", json=VALID)

    assert resp.status_code == 201, resp.text
    assert fake.profile_deleted is True
    assert fake.deleted == "ja-existe"
    assert fake.created is not None
