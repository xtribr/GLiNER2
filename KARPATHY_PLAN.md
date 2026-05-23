# Karpathy Plan — Ranking ENEM

**Data:** 2026-05-23
**Escopo:** Aplicar os 4 princípios de Karpathy ao monorepo `RANKING ENEM/`.

---

## 0. Diagnóstico Inicial

O repositório tem **duas camadas misturadas**:

1. **Raiz** — biblioteca legada GLiNER2 + scripts soltos de ETL ENEM (`enem_extractor.py`, `enem_extractor_completo.py`, `enem_ingestor_2025.py`, `enem_comparador_anual.py`, `extract_enem_data.py`, `migrate_to_supabase.py`, `enem_2024.db`, dois CSVs grandes).
2. **`enem-analytics/`** — produto real em produção: Next.js 14 + FastAPI + Supabase, deploy no Coolify (rankingenem.com).

`CLAUDE.md`, `AGENTS.md` e `analysis_and_improvements.md` descrevem **apenas a biblioteca GLiNER2**, não o produto. Quem chega no repo é induzido ao erro.

**Sintomas concretos:**

| Arquivo | Linhas | Sintoma |
|---|---|---|
| `frontend/src/components/gliner/GLiNERInsights.tsx` | 1606 | Componente monolítico |
| `backend/scripts/update_enem_year.py` | 1441 | Script de manutenção virou framework |
| `backend/api/routes/gliner_insights.py` | 1087 | Rota gigante, mistura ML + serialização |
| `backend/api/routes/predictions.py` | 965 | Rota gigante |
| `frontend/src/app/compare/page.tsx` | 907 | Página com lógica + UI + estado |
| `frontend/src/app/schools/[codigo_inep]/page.tsx` | 839 | Idem |
| `backend/ml/preprocessor.py` | 771 | Pré-processamento sem fronteira clara |
| `backend/ml/prediction_model.py` | 759 | Modelo + I/O + features no mesmo módulo |
| `backend/scripts/` | 30 scripts | Sprawl: muitos one-shots persistidos |
| `backend/tests/` | 4 arquivos | Cobertura claramente abaixo de 80% |

Untracked sugerem exploração em andamento: `.agents/`, `.codex/`, `microdados de 2005 a 2015/`.

Histórico recente mostra direção certa (consolidação Supabase, remoção de `sys.path` hacks, fixes de XSS). Vamos seguir o vetor.

---

## 1. Think Before Coding — perguntas a resolver antes de qualquer refactor

Não toque em código até decidir, com o usuário:

- **Q1.** A biblioteca `gliner2/` ainda é usada pelo produto, ou virou peso morto? (O backend importa de `ml/gliner_processor.py` — checar se `gliner2/` é dependência viva).
- **Q2.** Os scripts/CSV/SQLite na raiz (`enem_2024.db`, `enem_2018_2024_completo.csv`, `enem_extractor*.py`, `enem_ingestor_2025.py`, `enem_comparador_anual.py`, `migrate_to_supabase.py`) ainda são executados, ou são histórico já migrado para `enem-analytics/backend/scripts/`?
- **Q3.** `microdados de 2005 a 2015/` (untracked, 11 anos de dados) entra no pipeline ou é arquivo morto?
- **Q4.** Qual é o **único caminho oficial** para ingerir um novo ano de microdados? Hoje existem candidatos: `enem_ingestor_2025.py`, `enem-analytics/backend/scripts/update_enem_year.py` (1441 linhas), `enem-analytics/backend/scripts/import_enem_year.py`. Escolher um, deletar os outros.

**Regra:** sem responder Q1–Q4, qualquer "limpeza" vira chute.

---

## 2. Simplicity First — alvos óbvios de redução

Sem features novas. Sem abstração nova. Só apagar o que claramente não serve e cortar arquivos gigantes em pedaços do tamanho certo.

### 2.1 Apagar (após confirmar Q1–Q4)
- Raiz: scripts e bancos legados duplicados pelo `enem-analytics/backend/`.
- `analysis_and_improvements.md` (descreve GLiNER2, não o produto).
- Scripts em `backend/scripts/` marcados pelo time como one-shot já executado (lista a produzir).
- Pastas `.agents/` e `.codex/` se forem rascunho local — ou mover para `.gitignore`.

### 2.2 Decompor (sem reescrever — extrair, não refatorar)
Quebrar em arquivos de 200–400 linhas conforme a regra do repo, **um arquivo gigante por vez**, com testes antes e depois:

1. `GLiNERInsights.tsx` (1606) → separar: fetch hook, visualizações por aba, container.
2. `update_enem_year.py` (1441) → separar: download, validação, transformação, upsert, CLI.
3. `routes/gliner_insights.py` (1087) → rota fina + serviço em `ml/` ou `services/`.
4. `routes/predictions.py` (965) → idem.
5. `compare/page.tsx` (907) + `schools/[codigo_inep]/page.tsx` (839) → extrair seções já existentes em `components/compare/` para fora da página.

### 2.3 Não fazer agora
- Nada de "melhorar adjacente". Cada PR toca um alvo.
- Nada de design system novo, nada de novo state management, nada de migrar framework.

---

## 3. Surgical Changes — disciplina de PR

Cada PR deve obedecer:

- Um alvo (uma pergunta de §1 OU um arquivo gigante de §2.2).
- Toda linha alterada rastreável ao objetivo do PR.
- Estilo existente mantido (mesmo padrão FastAPI/Next.js já adotado).
- Imports/funções órfãos **criados pelo PR** são removidos; código morto pré-existente vira issue, não deleção silenciosa.
- Commit message no padrão atual: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:` (consistente com histórico recente).

**Anti-padrão a evitar:** PRs como o de Dec/Jan misturando "UI improvements + backend fixes + dashboard stats RPC" (commits `682a8d0`, `15ef077`). Esses são impossíveis de revisar e de reverter.

---

## 4. Goal-Driven Execution — plano executável

Cada passo tem critério de sucesso verificável. Loop até cumprir.

### Fase A — Verdade documental (1 PR)
1. Reescrever `CLAUDE.md` e `AGENTS.md` para descreverem o produto atual (`enem-analytics/`), com seções: stack, comandos, deploy, fluxos.
   → **Verificar:** `grep -r GLiNER2 CLAUDE.md AGENTS.md` retorna apenas menção de uso interno, não como descrição do projeto.
2. Marcar `analysis_and_improvements.md` como histórico ou apagar.
   → **Verificar:** removido ou movido para `docs/legacy/`.

### Fase B — Decidir e remover ambiguidade (depende de Q1–Q4) (1 PR)
1. Eleger **um** caminho oficial de ingestão de microdados.
   → **Verificar:** `README.md` aponta para um único script; os outros foram removidos ou explicitamente marcados como `deprecated_` no nome.
2. Eleger **uma** localização para código de ETL (raiz vs. `enem-analytics/backend/scripts/`).
   → **Verificar:** raiz não tem mais scripts ENEM ativos.
3. Decidir destino de `microdados de 2005 a 2015/`: ou entra no pipeline, ou vai para `.gitignore`.
   → **Verificar:** `git status` limpo nessa pasta.

### Fase C — Rede de proteção antes de cortar gigantes (1 PR por módulo)
Karpathy: "Refactor X → Ensure tests pass before and after". Hoje não temos garantia. Antes de cortar cada arquivo gigante:

1. Escrever testes de caracterização para o módulo alvo (input/output observável).
   → **Verificar:** `pytest backend/tests/test_<alvo>.py` passa contra o código atual.
2. Só então decompor.
   → **Verificar:** mesmos testes passam após decomposição, sem alteração.

Ordem sugerida (maior dor primeiro):
- C1. `update_enem_year.py` — é o pipeline de dados, falha aqui afunda o produto.
- C2. `routes/predictions.py` — endpoint mais consultado.
- C3. `GLiNERInsights.tsx` — componente que mais cresce.
- C4. `routes/gliner_insights.py`.
- C5. `compare/page.tsx` e `schools/[codigo_inep]/page.tsx`.

### Fase D — Cobertura mínima (contínua durante C)
Meta: 80% conforme regra do repo. Hoje: 4 arquivos de teste no backend, nenhum no frontend visível.

→ **Verificar:** `pytest --cov=backend` mostra ≥ 80% nas rotas e em `ml/`.
→ **Verificar:** `pnpm test` rodando em CI no frontend (mesmo que com cobertura inicial baixa, começar a medir).

---

## 5. O que **não** está neste plano (intencional)

- Trocar Next.js/FastAPI/Supabase — não está quebrado.
- Introduzir microsserviços, GraphQL, monorepo tool (Nx/Turborepo) — sem requisito.
- Reescrever ML pipeline — `prediction_model.py` funciona; cortar em pedaços é Fase C, não reescrita.
- Tocar em `tutorial/`, `gliner2/`, `xtri-claude-agents/` enquanto Q1 não for respondida.
- Otimização de performance sem profiling — sem evidência de gargalo.

---

## 6. Próxima ação concreta

Responder Q1–Q4 da Seção 1. Tudo depende disso. Sem respostas, o trabalho seguinte é chute caro.
