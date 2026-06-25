"""Contratos da listagem administrativa de leads."""


class _Resp:
    def __init__(self, data):
        self.data = data


class _Tbl:
    def __init__(self, rows):
        self.rows = rows
        self.eq_calls = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, field, value):
        self.eq_calls.append((field, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _Resp(self.rows)


class _Supabase:
    def __init__(self, rows):
        self.table_name = None
        self.table_obj = _Tbl(rows)

    def table(self, name):
        self.table_name = name
        return self.table_obj


def test_list_leads_filtra_apenas_cadastros_publicos_ativos(monkeypatch):
    from api.auth import supabase_service

    fake = _Supabase(rows=[{
        "id": "lead-1",
        "nome_escola": "Escola Teste",
        "codigo_inep": "11111111",
        "nome_contato": "Contato",
        "cargo": "Diretor",
        "telefone": "",
        "email": "contato@example.com",
        "email_verified": False,
        "created_at": "2026-06-22 23:25:26+00",
    }])
    monkeypatch.setattr(supabase_service, "get_supabase", lambda: fake)

    leads = supabase_service.list_leads()

    assert fake.table_name == "profiles"
    assert ("origem", "cadastro_publico") in fake.table_obj.eq_calls
    assert ("is_active", True) in fake.table_obj.eq_calls
    assert len(leads) == 1
