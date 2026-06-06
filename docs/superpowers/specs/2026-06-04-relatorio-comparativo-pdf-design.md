# Relatório Comparativo em PDF — Redesenho exaustivo (formato "XTRI / SISU")

**Data:** 2026-06-04
**Página afetada:** `enem-analytics/frontend/src/app/compare/page.tsx`
**Status:** ✅ **APROVADO** — formato validado em PDF real (protótipo) em 2026-06-04.

## 0. Protótipo validado (2026-06-04)

O formato foi prototipado como **PDF real** (HTML → Chrome headless) e **aprovado pelo usuário**. Artefato: `.superpowers/report-prototype/report.html` — o HTML/CSS é a **base direta do `ReportDocument`** de produção. Validado:

- **2 páginas A4 densas, sem espaço em branco**; banner de seção azul `#2ba9df` + header de tabela coral `#ff6b5c` + zebra; **logo XTRI** no header.
- **Análise em TEXTO** (não só tabelas): Resumo Executivo + **um parágrafo analítico por área** explicando *onde / quanto / por quê / o que fazer*.
- **As 5 notas (CN/CH/LC/MT/RED) como eixo central** — seção "Comparação Detalhada — as 5 notas, uma a uma".
- **12 recomendações** priorizadas (A·Alta/Média/Baixa, B, Ambas, Benchmark), cada uma ligada a um gap real.
- Densidade alta (margens/fontes compactas) para não gerar páginas semivazias — requisito explícito do usuário.
- (Opcional, fora do protótipo de 2 págs: blocos dedicados de habilidades, cluster e conceitos GLiNER → expandiriam para 3ª página; decidir na implementação.)

## 1. Problema

O PDF atual (`ExecutiveReportGenerator.tsx`, jsPDF) é pobre: sem gráficos, recomendações genéricas fixas, sem acentos, poucos dados. O usuário quer um relatório **exaustivo** — cruzar **todas** as análises que a plataforma faz por escola (diagnóstico, ganho por TRI, conteúdos, habilidades, conceitos, cluster, recomendações), comparando as 2 escolas, no formato denso do `relatorio-monitoramento-sisu-direito.pdf`.

## 2. Direção visual (aprovada — manter)

- **Logo XTRI** no header (`public/logo-xtri.png`).
- Banners de seção azul-ciano `#2ba9df`; headers de tabela coral `#ff6b5c`; linhas zebradas `#f4f7fa`. Sub-seções de diagnóstico coloridas por área.
- Denso, em tabelas, multipágina, com gráficos.

## 3. Abordagem técnica

**HTML → PDF via `html2pdf.js`** (já é dependência), renderizando um componente React dedicado e capturando-o. Reusa os gráficos `recharts`; acentos nativos. Substitui o `ExecutiveReportGenerator` (jsPDF).

## 4. Estrutura (11 seções, multipágina)

Header: logo XTRI + "Relatório Comparativo de Escolas" + "Análise gerada em DD/MM/AAAA, HH:MM | ENEM 2024".

1. **Escolas Comparadas** — nome, INEP, UF/cidade, tipo, porte, anos, **cluster/perfil** (`getSchoolCluster`), média, ranking BR/UF, saúde geral (`overall_health`).
2. **Veredito** — líder, vantagem na média, áreas vencidas, gap de ranking, **domínio TRI geral** (`overall_tri_mastery` A×B).
3. **Análise por Área** — Área · Nota A/B · **Média Nacional** · **Média Pares** · **Gap nacional A/B** · **Z-score A/B** · Status A/B · Vence. (via `getDiagnosis` de cada escola — `AreaAnalysis`).
4. **Ganho / Projeção por TRI** — por área: score atual, **tendência** (annual_change, r²), **ganho potencial**, cenários (conservador/realista/otimista), **predição oficial** (Δ esperado). (`getAreaProjection` A&B / `getPredictionComparison`).
5. **Conteúdos TRI a focar** — por área e escola: conteúdos/skills com maior `gap` e seu `tri_score` (`stretch_content` das projeções + `getTriContent`/`getTriSkills`). É o "o que estudar" do SISU.
6. **Habilidades** — fortes e fracas de cada escola (`getSchoolSkills`, `getWorstSkills`), `desempenho_habilidades` histórico.
7. **Conceitos & Redação (GLiNER)** — conceitos prioritários por área (`getGlinerConceptAnalysis`); **redação por competência C1–C5** escola×nacional com status (`getGlinerStudyFocus`).
8. **Recomendações & Quick Wins** — recomendações acionáveis e ganhos rápidos por escola (`getRecommendations`, `getQuickWins`, `getImprovementPotential`).
9. **Panorama Visual (gráficos)** — barras por área (A×B), radar das 5 áreas, evolução histórica (linhas), barras de ganho projetado.
10. **Histórico Ano a Ano** — ranking BR/UF, média e notas por área por ano (A e B).
11. **Diagnóstico por Área (síntese acionável)** — sub-seções coloridas por área, juntando status + gap + ganho + conteúdo + recomendação numa leitura única (espírito "Plano de Aprovação").

## 5. Matriz de dados — real / calculável / fora

| Categoria | Origem | Status |
|-----------|--------|--------|
| Identificação, porte, anos, ranking BR/UF, notas/ano por área | `SchoolHistory`, `getSchool` | ✅ real |
| **Z-score, média nacional, média pares, gap por área, status, priority** | `getDiagnosis` (A e B) · `AreaAnalysis` | ✅ real |
| **Ganho potencial, cenários, tendência (r²), conteúdos a focar (stretch)** | `getAreaProjection` (A e B) · `TRIAreaProjection` | ✅ real |
| Predição comparada / Δ esperado por área | `getPredictionComparison` (A e B) | ✅ real |
| Domínio TRI geral, weak skills | `getTRIAnalysis` (A e B) | ✅ real |
| Conteúdos / skills TRI por área | `getTriContent`, `getTriSkills`, `stretch_content` | ✅ real |
| Habilidades fortes/fracas | `getSchoolSkills`, `getWorstSkills` | ✅ real |
| Conceitos GLiNER; redação C1–C5 (escola×nacional) | `getGlinerConceptAnalysis`, `getGlinerStudyFocus` | ✅ real |
| Cluster/perfil | `getSchoolCluster` | ✅ real |
| Recomendações, quick wins, potencial | `getRecommendations`, `getQuickWins`, `getImprovementPotential` | ✅ real |
| Tendência, volatilidade, CV, percentil de ranking | calculados (histórico/ranking) | 🧮 calculável |
| **Percentil exato por área** | — (sem distribuição nacional por área) | ❌ fora → use z-score + gap_to_national, que existem |

## 6. Componentes

**Criar:**
- `components/compare/report/ReportDocument.tsx` — as 11 seções (HTML/CSS estilo SISU) reusando os charts.
- `components/compare/report/reportMetrics.ts` — funções puras (tendência, volatilidade, CV, percentil de ranking, montagem de linhas). **Testáveis.**
- `components/compare/report/useReportData.ts` — orquestra a **coleta paralela** de todos os endpoints para A e B (ver §7).
- `components/compare/report/generateReportPdf.ts` — render off-screen → `html2pdf` → blob/download.

**Modificar:** `compare/page.tsx` `handleExportPdf` → usar o novo fluxo.
**Aposentar:** `ExecutiveReportGenerator.tsx` (jsPDF) e `PDFExportModal.tsx` (dead code).

## 7. Fluxo, performance e erros

- **Coleta pesada:** o relatório exige ~12–18 chamadas (várias por escola: diagnóstico, projeção por área, predição, skills, gliner, cluster, recomendações). Buscar **em paralelo** (`Promise.all`) por escola e exibir progresso ("Gerando relatório — coletando análises…"). Reusar o que a página já carregou (diagnóstico/histórico) para não rebuscar.
- **Degradação graciosa:** se um endpoint falhar/estiver indisponível para uma escola, a seção correspondente é **omitida** com nota, sem quebrar o PDF.
- Render off-screen → aguardar charts montarem → `html2pdf` (multipágina, `page-break` por seção) → download (mantém fallback atual).
- Erros gerais: `try/catch` + mensagem (já existe).

## 8. Testes

- **Unit:** `reportMetrics.ts` (puro) — tendência, volatilidade, CV, percentil; casos: histórico curto/ausente, empate. *(frontend só tem Playwright hoje; decidir vitest vs e2e.)*
- **Manual/visual:** gerar com 2 escolas reais no preview; validar todas as seções, acentos, gráficos, quebras de página, e a degradação quando falta dado.

## 9. Fora de escopo (v1)

- Percentil exato por área e média estadual por área (não existem — cobertos por z-score/gap/ranking_uf).
- Customização do relatório pelo usuário; outros idiomas.
- Nome do arquivo com escolas reais — **melhoria trivial, entra junto**.

## 10. Faseamento sugerido (para o plano de implementação)

Dado o tamanho, o `writing-plans` deve quebrar em fases verificáveis:
- **Fase 1** — esqueleto: `ReportDocument` + html2pdf + seções 1-3 + 9-10 (identificação, veredito, área com z-score/nacional, gráficos, histórico) reusando dados já carregados.
- **Fase 2** — ganho/projeção (4-5) + predição.
- **Fase 3** — habilidades, conceitos/redação, recomendações (6-8) + síntese acionável (11).

Cada fase gera um PDF válido e mais rico que a anterior.

## 11. Open questions (não bloqueiam)

- **OQ-1:** Endpoint exato de média nacional por área agregada (a home consome) vs usar `national_avg` do `getDiagnosis` — confirmar na impl.
- **OQ-2:** Custo real das ~12–18 chamadas; se lento, mover seções 4-8 para "anexo" carregado sob demanda ou cache.
