"""Tests for list_all_profiles batching + bulk email resolution + row resilience.

Covers the leads-vs-users fix: the admin user list must return EVERY profile
(not a 100/1000-row window), resolve emails in one bulk Auth pass (no per-row
N+1), and never let a single malformed row blank the whole list.
"""

import os
import sys
import types

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# Other tests stub api.auth.* in sys.modules; force the REAL module here.
for _m in [m for m in list(sys.modules) if m == "api.auth" or m.startswith("api.auth.")]:
    del sys.modules[_m]

from api.auth import supabase_service as svc  # noqa: E402


class _Resp:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class _Query:
    """Chainable fake for select/order/eq/range/limit/execute (+ count='exact')."""

    def __init__(self, rows):
        self._rows = rows
        self._eq = {}
        self._range = None
        self._count_mode = False

    def select(self, *_a, **k):
        self._count_mode = k.get("count") == "exact"
        return self

    def order(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._eq[col] = val
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def limit(self, _n):
        return self

    def execute(self):
        rows = [r for r in self._rows if all(r.get(k) == v for k, v in self._eq.items())]
        if self._count_mode:
            return _Resp(data=[], count=len(rows))
        if self._range is not None:
            start, end = self._range
            return _Resp(data=rows[start : end + 1])
        return _Resp(data=rows)


class _AuthAdmin:
    def __init__(self, users):
        self._users = users

    def list_users(self, page=None, per_page=None):
        page = page or 1
        per_page = per_page or 50
        start = (page - 1) * per_page
        return self._users[start : start + per_page]


class _FakeSupabase:
    def __init__(self, rows, auth_users):
        self._rows = rows
        self.auth = types.SimpleNamespace(admin=_AuthAdmin(auth_users))

    def table(self, name):
        assert name == "profiles"
        return _Query(self._rows)


def _profile_row(i, **over):
    row = {
        "id": f"u{i}",
        "codigo_inep": f"{10000000 + i}",
        "nome_escola": f"Escola {i}",
        "email": "",  # signup never stores email → forces the bulk map path
        "is_admin": False,
        "is_active": True,
        "created_at": f"2026-01-01T00:00:{i % 60:02d}Z",
    }
    row.update(over)
    return row


def test_list_all_profiles_returns_every_row_beyond_the_page_cap(monkeypatch):
    # 2300 rows → exercises 3 pages (1000 + 1000 + 300); the old 100/1000 cap
    # would have hidden the tail (exactly the leads-vs-users symptom).
    rows = [_profile_row(i) for i in range(2300)]
    auth_users = [types.SimpleNamespace(id=f"u{i}", email=f"e{i}@x.com") for i in range(2300)]
    monkeypatch.setattr(svc, "get_supabase", lambda: _FakeSupabase(rows, auth_users))

    result = svc.list_all_profiles()

    assert len(result) == 2300
    assert {p.codigo_inep for p in result} == {f"{10000000 + i}" for i in range(2300)}
    # email came from the bulk Auth map, not a per-row lookup
    assert result[0].email == "e0@x.com"


def test_list_all_profiles_skips_malformed_row_without_blanking_list(monkeypatch):
    good = [_profile_row(i) for i in range(3)]
    bad = {"id": "bad", "nome_escola": "No INEP key"}  # missing codigo_inep
    rows = good[:2] + [bad] + good[2:]
    monkeypatch.setattr(svc, "get_supabase", lambda: _FakeSupabase(rows, []))

    result = svc.list_all_profiles()

    # 3 good rows survive; the malformed one is skipped, not all-or-nothing []
    assert len(result) == 3
    assert all(p.codigo_inep for p in result)


def test_list_all_profiles_bounded_limit_still_works(monkeypatch):
    rows = [_profile_row(i) for i in range(50)]
    monkeypatch.setattr(svc, "get_supabase", lambda: _FakeSupabase(rows, []))

    result = svc.list_all_profiles(skip=0, limit=10)

    assert len(result) == 10


def test_bulk_email_map_falls_back_to_profile_email(monkeypatch):
    # No matching auth user for u0, but profiles.email is set → use it.
    rows = [_profile_row(0, email="stored@x.com")]
    monkeypatch.setattr(svc, "get_supabase", lambda: _FakeSupabase(rows, []))

    result = svc.list_all_profiles()

    assert result[0].email == "stored@x.com"
