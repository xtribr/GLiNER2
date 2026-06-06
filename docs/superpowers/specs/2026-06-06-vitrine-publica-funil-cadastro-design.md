# Spec — Vitrine pública + Funil de captação de leads (rankingenem.com)

**Data:** 2026-06-06
**Status:** Aprovação pendente do usuário
**Origem:** Inverter o modelo do produto (B2B fechado → funil de lead aberto), espelhando a estrutura do Radar ENEM (Bernoulli), mantendo a identidade visual XTRI.

---

## 1. Contexto e objetivo

Hoje o `rankingenem.com` é um produto **fechado**: o `ClientLayout` redireciona qualquer visitante sem sessão para `/login`, o dashboard de ranking (que já existe e é rico) fica trancado, e cada escola só enxerga a si mesma. Os endpoints de ranking são **admin-only**.

O objetivo é **inverter** isso para um funil de captação de leads de escolas:

- **Vitrine pública** (ranking + lista + resumo) como isca/SEO — sem login.
- **Cadastro self-service sem atrito** que captura o lead.
- **Análise profunda gated** — a escola, após cadastro, vê o painel completo da própria escola.

A identidade XTRI (navy `#061927`, cyan `#28B7ED`, laranja `#FF4B2E`, logo X) é mantida.

---

## 2. Decisões travadas (input do brainstorming)

1. **Modelo invertido:** vitrine pública (isca) + cadastro self-service + análise gated. Identidade XTRI mantida.
2. **Escopo (Funil 1):** público (deslogado) = ranking + lista + **resumo** por escola (com teaser borrado em parte do ranking). Escola logada = **modelo B** (trancada só no próprio Painel + Roadmap). Admin = tudo (inalterado).
3. **Auth — modelo A:** cadastro **com senha na hora** → acesso imediato. E-mail = boas-vindas **não-bloqueante** (não está no caminho crítico de acesso).
4. **Lead capturado no submit:** nome do contato, escola (typeahead INEP), cargo (dropdown), e-mail, celular, senha. Usuário nasce amarrado ao `codigo_inep`, `is_admin=false`.
5. **Roles:** binário admin/escola — **sem RBAC novo**. Comparar, Tendências, Oráculo, Habilidades permanecem **admin-only** (já é assim hoje).
6. **Segurança (nível A):** 1 conta por e-mail · 1 escola por perfil · Termos de Uso (com nota de que os dados derivam de microdados **públicos** do INEP) · audit/rate-limit. **Sem CPF/CNPJ.** Verificação forte (domínio/aprovação) fica engatilhada para fase futura.
7. **Higiene de lead (nível A):** formato + checagem de MX + blocklist de descartáveis + honeypot + rate-limit (invisível) · e-mail de confirmação **não-bloqueante** → flag `email_verificado` · lead scoring no painel admin. Telefone validado no follow-up comercial.

---

## 3. Modelo de produto — matriz de acesso

Três públicos. A regra de isolamento por escola (`ensure_school_access`) **já existe e é enforçada de verdade** no backend.

| Recurso | Deslogado (público) | Escola logada (modelo B) | Admin |
|---|---|---|---|
| Vitrine `/` (ranking + stats) | ✅ vê (com teaser) | ↪︎ redirecionado p/ próprio painel | ✅ ranking completo |
| Lista `/schools` + filtros | ✅ | ↪︎ redirecionado | ✅ |
| Resumo por escola (qualquer) | ✅ (só resumo) | ↪︎ redirecionado | ✅ |
| **Cadastro** `/cadastro` | ✅ | — | — |
| Painel profundo da **própria** escola | 🔒 → cadastro | ✅ | ✅ (qualquer) |
| Painel profundo de **outra** escola | 🔒 | ⛔ 403 | ✅ |
| Comparar, Tendências, Oráculo, Habilidades | 🔒 | ⛔ | ✅ |
| Admin / Usuários / Leads | 🔒 | ⛔ | ✅ |

> **Nota modelo B:** uma escola logada **não** navega a vitrine pública — ao acessar `/`, `/schools` etc. é levada ao próprio painel. O visitante deslogado é que vê a vitrine.

---

## 4. Arquitetura — duas camadas

```
┌─────────────────────────── CAMADA PÚBLICA (sem auth) ───────────────────────────┐
│  Frontend: / (vitrine), /schools (lista+filtros), /schools/[inep] (RESUMO), /cadastro │
│  Backend:  GET ranking, lista, busca, stats, resumo por escola  (público)            │
│  Dados:    schools / enem_results / school_skills  → RLS SELECT já é USING (true)     │
└──────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── CAMADA GATED (auth) ─────────────────────────────────┐
│  Frontend: /schools/[inep] (PROFUNDO), /roadmap, /compare, /trends, /admin/*         │
│  Backend:  predictions, diagnosis, recommendations, clusters, tri_lists, compare,    │
│            skills agregados, admin/*  → get_authorized_school_user / get_current_admin│
└──────────────────────────────────────────────────────────────────────────────────┘
```

Princípio central: **a RLS já libera leitura pública** dos dados de ranking; o que tranca hoje é o **guard da API** (`get_current_admin`) e o **redirect do frontend**. Abrir a vitrine = criar uma camada de leitura pública + ajustar o gating do frontend, **sem reescrever** o que existe.

---

## 5. Backend — mudanças

### 5.1 Camada pública de leitura (Fase 1)
Hoje (verificado em `api/routes/schools.py`):
- `GET /schools/` (lista), `GET /schools/top`, `GET /schools/search` → `get_current_admin` (admin-only).
- `GET /schools/{inep}` (SchoolDetail), `/history`, `/skills` → `get_authorized_school_user`.
- `GET /schools/compare/{a}/{b}`, `/skills/worst`, `/skills/all` → `get_current_admin`.
- Endpoint de stats/overview (consumido por `api.getStats`) → confirmar guard e relaxar.

Mudanças:
- Expor versões **públicas** (sem token) de: ranking/top, lista+filtros, busca, stats. Abordagem preferida: reutilizar o `get_optional_user` (já existe em `supabase_dependencies.py`, hoje sem uso) → endpoint público que opcionalmente enriquece se logado. Alternativa: router `/api/public/*`.
- Novo `GET /schools/{inep}/summary` (ou `/api/public/schools/{inep}`) retornando **apenas o resumo** (posição, média TRI, notas por área, série) — **não** os campos profundos. O `GET /schools/{inep}` profundo continua `get_authorized_school_user`.
- **Compare, skills agregados, geração em lote, tri_lists mgmt, admin/* permanecem `get_current_admin`.** Sem mudança.
- Auditar a shape de cada resposta pública para não vazar campo gated.

### 5.2 Endpoint de cadastro self-service (Fase 3)
`POST /api/auth/signup` (ou `/api/cadastro`):
1. Valida payload (Pydantic): `nome_contato`, `codigo_inep`, `cargo`, `email` (EmailStr), `telefone`, `senha` (mín. 6), `aceite_termos` (true), honeypot vazio.
2. Higiene: honeypot preenchido → rejeita; MX do domínio do e-mail inexistente → rejeita; domínio descartável (blocklist) → rejeita; rate-limit por IP.
3. Cria usuário no Supabase via **service key** (mesmo padrão do `create_admin.py`) com a senha; cria/atualiza `profiles` com `codigo_inep`, `is_admin=false`, `is_active=true`, campos de lead, `email_verified=false`, `origem='cadastro_publico'`.
4. Dispara e-mail de boas-vindas com **link de confirmação** via Resend (`from: XTRI <...@xtri.online>`), **não-bloqueante**.
5. Retorna sucesso → o frontend já autentica e leva ao painel da escola (acesso imediato).

`GET /api/auth/confirm?token=…` → marca `email_verified=true`. Não afeta acesso.

### 5.3 Leads para o admin (Fase 4)
`GET /api/admin/leads` (`get_current_admin`): lista perfis `origem='cadastro_publico'` com sinais de qualidade: `email_verified`, domínio institucional (heurística: não é gmail/hotmail/outlook/yahoo), formato de telefone válido, nº de cadastros do mesmo IP.

---

## 6. Frontend — mudanças

### 6.1 Gating (`ClientLayout.tsx`) — Fase 2
Refatorar a regra de redirect (hoje: sem sessão → `/login` para tudo):
- Definir prefixos **públicos**: `/`, `/schools`, `/cadastro`, `/login`.
- Sem sessão + rota pública → renderiza (não redireciona).
- Sem sessão + rota gated → `/login` (ou `/cadastro`).
- Sessão + escola (não-admin) → trava no próprio painel (regra atual mantida) e redireciona da vitrine (modelo B).
- Sessão + admin → tudo.

### 6.2 Vitrine `/` — Fase 2
- Reusar os componentes do dashboard atual (`page.tsx`: hero navy, stats, ranking Top N, médias por área) adaptando para **público**.
- Adicionar: CTA "Acessar análise gratuita" → `/cadastro`; prova social; cards de feature (Evolução histórica · Comparativo por área · Oráculo IA); **teaser** (parte do ranking borrada com CTA de cadastro); banner promo; mapa (placeholder na v1).
- Admin logado em `/` vê ranking completo (sem teaser) + sidebar.
- Referência visual: `.superpowers/brainstorm/.../vitrine-publica.html`.

### 6.3 Funil de cadastro `/cadastro` — Fase 3
Wizard de 3 passos (URL com `?step=escola|conta|pronto`):
- **Escola:** busca typeahead (reusa `GET /schools/search` público) → seleciona → avança.
- **Conta:** escola travada no topo; campos nome/cargo(dropdown)/e-mail/celular/senha; honeypot oculto; aceite de Termos; aviso suave "use o e-mail institucional"; botão "Criar conta e acessar".
- **Pronto:** acesso liberado → "Ir para o painel da minha escola"; nota do e-mail de boas-vindas (opcional, acesso já ativo).
- Referência visual: `.superpowers/brainstorm/.../cadastro-funil.html`.

### 6.4 Página por escola — Fase 2/3
- Pública (resumo): qualquer um vê posição, média TRI, notas por área.
- Profunda: só logado (própria escola) ou admin — conteúdo atual.

---

## 7. Modelo de dados — lead (migration 006)

Colunas **aditivas** em `public.profiles` (não quebra o existente):

| Coluna | Tipo | Uso |
|---|---|---|
| `nome_contato` | text | nome da pessoa (lead) |
| `cargo` | text | dropdown (Diretor/Coordenador/Professor/Secretaria/Outro) |
| `telefone` | text | WhatsApp do lead |
| `email_verified` | boolean default false | flag não-bloqueante |
| `origem` | text default 'cadastro_publico' | distingue lead de admin |

RLS: a `profiles_insert_policy` atual é admin-only — o signup insere via **service key** (bypassa RLS), igual ao `create_admin.py`. Sem mudança de policy. A `profiles_select_policy` (own OR admin) já protege os dados de contato (escola só vê o próprio perfil; admin vê todos para o painel de leads).

---

## 8. Higiene & segurança (nível A)

- **Formato e-mail:** Pydantic `EmailStr` (já há `pydantic[email]`).
- **MX/DNS:** checar registro MX do domínio (lib leve, ex. `dnspython`) — rejeita domínio sem e-mail.
- **Blocklist descartáveis:** lista estática de domínios temp-mail.
- **Honeypot:** campo oculto; preenchido → descarta (anti-bot, igual Bernoulli).
- **Rate-limit:** por IP no endpoint de signup.
- **Termos de Uso:** aceite obrigatório; texto deixa claro que os dados derivam de microdados públicos do INEP (escudo jurídico). *(Validar redação com jurídico.)*
- **Confirmação de e-mail:** link não-bloqueante → `email_verified`. Acesso nunca depende disso.

---

## 9. Fases e critérios de sucesso (verificáveis)

| Fase | Entrega | Critério de sucesso (checável) |
|---|---|---|
| **1. Backend público** | endpoints públicos de ranking/lista/busca/stats/resumo | `curl` sem token nos endpoints públicos → 200; compare/admin sem admin → 401/403; escola não acessa outra escola (403); `pytest` verde |
| **2. Frontend público** | gating refatorado + vitrine + página de resumo | deslogado vê `/` e `/schools`; escola logada trava no próprio painel; admin intacto; `pnpm build` passa |
| **3. Funil de cadastro** | `POST /signup` + higiene + wizard + welcome Resend | fluxo escola→conta→pronto cria conta + acesso imediato; honeypot/MX/descartável/duplicado rejeitados; e-mail chega; e2e passa |
| **4. Admin de leads** | `GET /admin/leads` + página com scoring | admin vê leads com selos (verificado/institucional/telefone) |

Cada fase é deployável isoladamente (Coolify dispara em push para `main`); o site permanece no ar entre fases.

---

## 10. Plano de testes

**Backend (pytest):**
- Público: ranking/lista/busca/stats/resumo retornam 200 **sem** token.
- Autorização: compare e admin/* → 403 sem admin; escola → 200 na própria, 403 em outra (`ensure_school_access`).
- Signup: cria user+profile, amarra `codigo_inep`, `is_admin=false`; honeypot preenchido → 400; e-mail descartável/MX inválido → 400; e-mail duplicado → 409; rate-limit dispara.
- Confirmação seta `email_verified`.

**Frontend (Playwright e2e):**
- Deslogado: vitrine carrega, ranking visível, cadastro ponta-a-ponta → painel.
- Escola logada: travada no próprio painel; `/compare`, `/trends`, outra escola → bloqueados.
- Admin: acesso completo.

Meta de cobertura mantida conforme `pytest.ini` do projeto.

---

## 11. Fora de escopo (YAGNI)

- CPF/CNPJ e verificação na Receita.
- OTP de WhatsApp/SMS; verificação de telefone no cadastro.
- Confirmação de e-mail **bloqueante**.
- RBAC granular / novos papéis.
- Tiers premium / paywall.
- Mapa de calor interativo real (v1 usa placeholder).

---

## 12. Riscos & mitigações

| Risco | Mitigação |
|---|---|
| Endpoint público vazar campo gated | Auditar shape de cada resposta; criar `/summary` dedicado em vez de expor o detail profundo |
| Abuso do signup (massa/fake) | Honeypot + rate-limit + higiene; lead scoring; verificação forte engatilhada p/ fase futura |
| Refactor do `ClientLayout` quebrar acesso de escola/admin | Cobrir os 3 públicos com e2e antes do deploy da Fase 2 |
| `/` renderizar diferente por papel (vitrine vs admin vs redirect) | Regras explícitas na matriz §3/§6.1; testar os 3 caminhos |
| Deliverability do welcome (público escolar) | E-mail é não-bloqueante por design (modelo A); confirmação só marca flag |
| Mudança em produção | Fases isoladas e deployáveis; cada uma com critério de sucesso |

---

## 13. Referências

- Mockups validados: `.superpowers/brainstorm/24346-1780748787/content/vitrine-publica.html` e `cadastro-funil.html`.
- Código-chave verificado: `api/auth/authorization.py` (`ensure_school_access`), `api/auth/supabase_dependencies.py` (`get_current_admin`, `get_optional_user`), `api/routes/schools.py` (guards), `scripts/migrations/003_optimize_rls_performance.sql` (RLS `USING (true)`), `frontend/src/components/layout/ClientLayout.tsx` (gating), `api/routes/contact.py` (padrão Resend).
