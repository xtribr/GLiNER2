# AGENTS.md

Guia para Codex (codex.ai/code) e demais agentes trabalharem neste repositório. Conteúdo espelha `CLAUDE.md` — se um for atualizado, atualizar o outro.

## Project Overview

**Ranking ENEM Analytics** é a plataforma da XTRI para análise de desempenho escolar no ENEM. Produto em produção em `rankingenem.com`.

- Frontend: Next.js 16 (App Router) + React 19 em `enem-analytics/frontend/`
- Backend: FastAPI em `enem-analytics/backend/`
- Auth + DB: Supabase Postgres (RLS habilitado)
- ML: scikit-learn + XGBoost (`backend/ml/`) servindo predições e clusters
- Deploy: Coolify (backend + frontend separados); ver `enem-analytics/README.md`

Fluxos principais: ranking de escolas, comparativo entre escolas, predições/projeções, diagnóstico, roadmap por escola, oráculo, painel admin.

## Layout do repositório

```
enem-analytics/                   # PRODUTO ATIVO
├── backend/
│   ├── api/                      # FastAPI: main.py + auth/ + admin/ + routes/
│   │   └── routes/               # schools, predictions, diagnosis, clusters,
│   │                             # recommendations, tri_lists, gliner_insights,
│   │                             # oracle, contact
│   ├── ml/                       # prediction_model, clustering_model,
│   │                             # diagnosis_engine, recommendation_engine,
│   │                             # preprocessor, gliner_processor (offline)
│   ├── data/                     # CSVs pré-computados servidos pela API
│   ├── scripts/                  # ETL, migrations SQL, treino, admin
│   ├── tests/                    # pytest (cobertura ainda abaixo de 80%)
│   ├── requirements-prod.txt     # único arquivo usado no Dockerfile
│   └── Dockerfile
├── frontend/
│   ├── src/app/                  # Next.js App Router (admin, compare, oraculo,
│   │                             # schools, skills, trends, login)
│   ├── src/components/           # compare/, gliner/, layout/, predictions/, ui/
│   ├── src/lib/                  # api, supabase, auth-context, enem-cycle
│   └── Dockerfile
├── docs/plans/
└── README.md                     # deploy Coolify, env vars, fluxos

microdados-2024/                  # microdados INEP brutos (gitignored)
microdados-2025/                  # staging para ingestão 2025 (README docs)
gliner2/                          # tooling OFFLINE (não roda na API) — ver §GLiNER2
scripts/                          # scripts soltos (legacy/oracle training)
```

Tudo na **raiz** que não seja `enem-analytics/`, `microdados-*/` ou `gliner2/` deve ser tratado como **legacy** (ver §Legacy).

## Comandos comuns

### Backend (FastAPI)

```bash
cd enem-analytics/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-prod.txt
uvicorn api.main:app --reload --port 8000

# Tests
pytest                            # ou: python -m pytest tests/
```

Variáveis obrigatórias: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`. Opcionais: `PIONEER_API_KEY`, `RESEND_API_KEY`.

### Frontend (Next.js)

```bash
cd enem-analytics/frontend
pnpm install                      # pnpm pinado em 8.15.9 (lockfile v6)
pnpm dev                          # localhost:3000
pnpm build && pnpm start
pnpm test                         # Vitest unitário
pnpm test:e2e                     # Playwright e2e (requer serviços locais)
```

Variáveis: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`. `NEXT_PUBLIC_*` é injetado em build — mudou, redeploy.

### Ingestão de dados ENEM

**Caminho oficial e único:** `enem-analytics/backend/scripts/update_enem_year.py`.

```bash
cd enem-analytics/backend
python scripts/update_enem_year.py \
  --year 2025 \
  --input ../../microdados-2025/microdados_enem_2025.zip \
  --input-format inep_raw \
  --env local --dry-run \
  --censo-file data/censo_escolas_2024.csv
# revisar relatório, depois trocar --dry-run por --apply
```

Aplica migration `backend/scripts/migrations/005_enem_results_atomic_import.sql` antes do primeiro `--apply`.

Existe `scripts/import_enem_year.py` no mesmo diretório — **deprecated**, não usar.

Antes de retreinar os modelos, valide a proveniência das fontes reais:

```bash
cd enem-analytics/backend
python scripts/retrain_prediction_models.py --validate-sources-only
```

O manifesto inclui ano, tamanho, SHA-256 e uso efetivo de cada fonte. O treino
é bloqueado se fontes usadas nas features tiverem anos incompatíveis.

## Autenticação

- Login: frontend → Supabase Auth direto.
- API: valida `Authorization: Bearer <access_token>` do Supabase. Sem JWT próprio.
- Perfil autenticado: `GET /api/auth/me`.
- Bootstrap do primeiro admin: `python scripts/create_admin.py <email> <senha> "<nome>"`.

## Endpoints principais

```
GET  /api/auth/me
GET  /api/schools                 GET /api/schools/{codigo_inep}
GET  /api/predictions/{codigo_inep}
GET  /api/diagnosis/{codigo_inep}
GET  /api/clusters/{codigo_inep}/cluster
GET  /api/recommendations/{codigo_inep}
GET  /api/gliner/...              (insights pré-computados)
GET  /api/admin/users             POST /api/admin/users (admin only)
GET  /api/admin/stats
```

## Convenções

- **Commits:** `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:` (ver `git log`).
- **Arquivos grandes:** alvo 200–400 linhas. Vários arquivos hoje violam isso (`update_enem_year.py` 1441, `gliner_insights.py` 1087, `predictions.py` 965, `GLiNERInsights.tsx` 1606, páginas `compare`/`schools/[codigo_inep]`). Decompor antes de adicionar features novas — ver `KARPATHY_PLAN.md`.
- **Supabase:** RLS habilitado em todas as tabelas. Mudanças de schema vão em `backend/scripts/migrations/NNN_*.sql` (numeração crescente).
- **CORS:** lista explícita em `api/main.py`. Novo domínio → editar lá.
- **Singleton dos modelos ML:** carregados uma vez em `ml/`. Não instanciar por requisição.

## GLiNER2 (`gliner2/` na raiz)

**Não é dependência da API em produção.** `requirements-prod.txt` não instala `gliner2` nem `gliner`. O Dockerfile só copia esse arquivo.

Uso real:
- Tooling offline para **regenerar** os CSVs `backend/data/conteudos_tri_gliner.csv` e os caches `gliner_cache*.json`.
- Consumidores: `enem-analytics/backend/reextract_gliner.py` e scripts de treino (`train_gliner_enem.py`, `finetune_gliner_enem.py`, `continue_training.py`, `test_finetuned_model.py`, `evaluate_model.py`, `reprocess_with_local_model.py`).

A rota `routes/gliner_insights.py` lê o CSV pré-computado — nunca invoca o modelo.

Para rodar o tooling offline:
```bash
pip install -e .                  # instala gliner2 a partir da raiz
python enem-analytics/backend/reextract_gliner.py
```

## Legacy

Arquivos na raiz que **não são referenciados** por `enem-analytics/`:

- Scripts: `enem_extractor.py`, `enem_extractor_completo.py`, `enem_ingestor_2025.py`, `enem_comparador_anual.py`, `enem_schema_analysis.py`, `extract_enem_data.py`, `migrate_to_supabase.py`.
- Bancos/CSVs: `enem_2024.db`, `enem_2024_completo.csv`, `enem_2018_2024_completo.csv`, `rankings_enem_2024_raw.json`.
- SQL: `enem_migration_2025.sql` (schema SQLite legado).
- Docs: `analysis_and_improvements.md` (descreve a biblioteca GLiNER2, não o produto), `PREPARACAO_ENEM_2025.md` (descreve fluxo SQLite obsoleto — produção é Supabase).

Não editar/usar sem confirmar com o usuário. Plano de remoção em `KARPATHY_PLAN.md`.

## Operacional

- Mudou env do backend → redeploy backend.
- Mudou `NEXT_PUBLIC_*` → redeploy frontend.
- Webhook Coolify dispara em push para `main`.
- Ingestão de novo ano de microdados: ver §Ingestão de dados ENEM acima.

## Antes de submeter mudanças

1. Backend: `pytest` passa.
2. Frontend: `pnpm build` passa; UI testada no navegador para mudanças visuais.
3. Sem segredos no diff (`.env`, service keys).
4. PR foca em **um alvo** — nada de commits "UI + backend + RPC" misturados (anti-padrão observado em commits passados).
