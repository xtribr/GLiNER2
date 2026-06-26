"""Cadastro self-service (vitrine pública / funil de leads) — modelo A.

Cria o usuário no Supabase Auth (service key) com a senha informada e o perfil
em `profiles` (is_admin=False), liberando acesso imediato. O e-mail de
boas-vindas (confirmação) é NÃO-bloqueante. Higiene: honeypot, e-mails
descartáveis, MX (se dnspython instalado) e rate-limit por IP.
"""

import logging
import os
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field

from .supabase_service import get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Cadastro"])

# Segredo para assinar o token de confirmação de e-mail (server-side).
CONFIRM_SECRET = os.getenv("SUPABASE_SERVICE_KEY", "dev-secret")

# E-mails descartáveis (lista curta; ampliável).
DISPOSABLE_DOMAINS = {
    "mailinator.com", "tempmail.com", "10minutemail.com", "guerrillamail.com",
    "yopmail.com", "trashmail.com", "getnada.com", "throwawaymail.com",
    "temp-mail.org", "fakeinbox.com", "sharklasers.com", "maildrop.cc",
    "dispostable.com", "mailnesia.com",
}

# Checagem de MX só se dnspython estiver disponível (opcional, sem nova dependência).
try:  # pragma: no cover - depende do ambiente
    import dns.resolver as _dns_resolver
except Exception:  # noqa: BLE001
    _dns_resolver = None

# Rate-limit simples em memória, por IP (reseta no restart do processo).
_RATE_WINDOW_SEC = 3600
_RATE_MAX = 5
_signup_hits: Dict[str, List[float]] = {}


class SignupRequest(BaseModel):
    codigo_inep: str = Field(..., min_length=1)
    nome_escola: str = Field(..., min_length=1)
    nome_contato: str = Field(..., min_length=2)
    cargo: str = Field(..., min_length=1)
    email: EmailStr
    telefone: str = Field(..., min_length=8)
    senha: str = Field(..., min_length=6)
    aceite_termos: bool
    website: str = ""  # honeypot — deve chegar vazio


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _signup_hits.get(ip, []) if now - t < _RATE_WINDOW_SEC]
    if len(hits) >= _RATE_MAX:
        _signup_hits[ip] = hits
        return True
    hits.append(now)
    _signup_hits[ip] = hits
    return False


def _email_domain_ok(email: str) -> bool:
    domain = email.rsplit("@", 1)[-1].lower()
    if domain in DISPOSABLE_DOMAINS:
        return False
    if _dns_resolver is not None:
        try:
            answers = _dns_resolver.resolve(domain, "MX")
            return len(answers) > 0
        except Exception:  # noqa: BLE001 - domínio sem MX / inexistente
            return False
    return True


def _make_confirm_token(user_id: str) -> str:
    return jwt.encode(
        {"sub": user_id, "purpose": "email_confirm"}, CONFIRM_SECRET, algorithm="HS256"
    )


def _send_welcome_email(email: str, nome_escola: str, confirm_url: str) -> None:
    """Envia o e-mail de boas-vindas via Resend. NÃO-bloqueante."""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.info("RESEND_API_KEY ausente — pulando welcome email (acesso já liberado).")
        return
    try:
        import resend

        resend.api_key = api_key
        resend.Emails.send(
            {
                "from": "X-TRI Escolas <contato@xtri.online>",
                "to": [email],
                "subject": "Bem-vindo ao Ranking ENEM XTRI",
                "html": (
                    f"<p>Olá! O acesso da <strong>{nome_escola}</strong> ao Ranking ENEM XTRI "
                    f"já está ativo.</p>"
                    f"<p>Para confirmar seu e-mail, clique aqui: "
                    f"<a href=\"{confirm_url}\">confirmar e-mail</a>.</p>"
                    f"<p>Se você não solicitou este cadastro, ignore esta mensagem.</p>"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - e-mail nunca bloqueia o cadastro
        logger.warning(f"Falha ao enviar welcome email para {email}: {exc}")


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, request: Request):
    """Cadastro self-service — cria a conta e libera acesso imediato (modelo A)."""
    # Honeypot anti-bot
    if payload.website:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cadastro inválido.")

    if not payload.aceite_termos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="É necessário aceitar os Termos de Uso.",
        )

    if _rate_limited(_client_ip(request)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Muitas tentativas. Tente novamente mais tarde.",
        )

    if not _email_domain_ok(str(payload.email)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use um e-mail válido e não descartável.",
        )

    supabase = get_supabase()

    # 1 escola : 1 perfil — mas um cadastro ANTERIOR removido/desativado não pode
    # bloquear um novo cadastro (auto-cura). Só perfis ATIVOS barram.
    existing = (
        supabase.table("profiles")
        .select("id, is_active")
        .eq("codigo_inep", payload.codigo_inep)
        .limit(1)
        .execute()
    )
    if existing.data:
        prev = existing.data[0]
        if prev.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Esta escola já está cadastrada. Faça login ou fale com o suporte.",
            )
        # Resíduo de um cadastro removido/inativo → limpa (perfil + usuário de auth)
        # para liberar o código INEP e evitar conflito de unicidade.
        try:
            supabase.table("profiles").delete().eq("id", prev["id"]).execute()
            supabase.auth.admin.delete_user(prev["id"])
        except Exception as exc:  # noqa: BLE001 - melhor-esforço; segue o cadastro
            logger.warning(f"Falha ao limpar cadastro inativo {prev['id']}: {exc}")

    # Cria usuário no Supabase Auth (acesso imediato → email_confirm=True).
    # NÃO gravamos is_admin no user_metadata (is_admin vem só da tabela profiles).
    try:
        auth_response = supabase.auth.admin.create_user(
            {
                "email": str(payload.email),
                "password": payload.senha,
                "email_confirm": True,
                "user_metadata": {
                    "codigo_inep": payload.codigo_inep,
                    "nome_escola": payload.nome_escola,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        if "already" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Este e-mail já está cadastrado. Faça login.",
            )
        logger.error(f"Erro ao criar usuário no Supabase Auth: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao criar a conta.",
        )

    if not auth_response.user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao criar a conta.",
        )

    user_id = auth_response.user.id

    # Perfil (lead) — is_admin SEMPRE False.
    try:
        supabase.table("profiles").insert(
            {
                "id": user_id,
                "codigo_inep": payload.codigo_inep,
                "nome_escola": payload.nome_escola,
                "is_admin": False,
                "is_active": True,
                "nome_contato": payload.nome_contato,
                "cargo": payload.cargo,
                "telefone": payload.telefone,
                "email_verified": False,
                "origem": "cadastro_publico",
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Erro ao criar perfil (rollback do auth user {user_id}): {exc}")
        try:
            supabase.auth.admin.delete_user(user_id)
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao criar o perfil da escola.",
        )

    # Welcome email (não-bloqueante).
    token = _make_confirm_token(user_id)
    confirm_url = f"{str(request.base_url).rstrip('/')}/api/auth/confirm?token={token}"
    _send_welcome_email(str(payload.email), payload.nome_escola, confirm_url)

    return {"success": True, "codigo_inep": payload.codigo_inep, "email": str(payload.email)}


@router.get("/confirm")
async def confirm_email(token: str):
    """Confirma o e-mail (flag não-bloqueante) e redireciona para a vitrine."""
    try:
        claims = jwt.decode(token, CONFIRM_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido ou expirado."
        )

    if claims.get("purpose") != "email_confirm" or not claims.get("sub"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido.")

    try:
        supabase = get_supabase()
        supabase.table("profiles").update({"email_verified": True}).eq(
            "id", claims["sub"]
        ).execute()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Erro ao confirmar e-mail: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erro ao confirmar e-mail."
        )

    frontend = os.getenv("FRONTEND_PUBLIC_URL", "https://rankingenem.com")
    return RedirectResponse(url=f"{frontend}/?email_confirmado=1")
