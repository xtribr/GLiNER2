# Relatório Comparativo Premium (PDF) — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Substituir o `ExecutiveReportGenerator.tsx` (jsPDF, fraco) por um relatório comparativo premium multipágina, gerado a partir de um componente React renderizado para PDF via `html2pdf.js`, com nomes reais, análise textual por área, projeções TRI, redação C1–C5, recomendações priorizadas, gráficos (barras/radar/linha) e histórico.

**Architecture:** Um componente off-screen `ReportDocument` (porte direto do protótipo validado `.superpowers/report-prototype/report.html`) é alimentado por um hook `useReportData` que coleta em paralelo os ~17 endpoints da `api.ts` para as duas escolas. A lógica pura (cálculos + geração das narrativas) vive em módulos testáveis (`reportMetrics.ts`, `reportNarrative.ts`). `generateReportPdf.ts` renderiza o componente fora da tela e chama `html2pdf` para baixar. Seções degradam graciosamente quando falta dado.

**Tech Stack:** Next.js 16 / React 19, TypeScript, `html2pdf.js` (já instalado), `recharts` (já instalado) ou SVG inline, **vitest** (a adicionar p/ unit tests), Tailwind para o resto do app (o relatório usa CSS próprio escopado).

**Spec:** `docs/superpowers/specs/2026-06-04-relatorio-comparativo-pdf-design.md` (aprovada; reconfirmada 2026-06-08).
**Base de markup/CSS validada:** `.superpowers/report-prototype/report.html` (508KB PDF validado pelo usuário).

---

## Estrutura de arquivos

**Criar:**
- `enem-analytics/frontend/src/components/compare/report/types.ts` — shape agregado `ReportData` + sub-tipos por seção.
- `enem-analytics/frontend/src/components/compare/report/reportMetrics.ts` — funções puras de cálculo (gap, áreas vencidas, z-score, tendência, percentil ranking, linhas de tabela). **Testável.**
- `enem-analytics/frontend/src/components/compare/report/reportNarrative.ts` — geração das strings de narrativa (resumo executivo, parágrafo por área) a partir dos dados. **Testável.**
- `enem-analytics/frontend/src/components/compare/report/reportTheme.ts` — constantes de cor/estilo (azul `#2ba9df`, coral `#ff6b5c`, etc.).
- `enem-analytics/frontend/src/components/compare/report/ReportDocument.tsx` — as 11 seções (porte de `report.html`) + radar + badges de status.
- `enem-analytics/frontend/src/components/compare/report/ReportDocument.css` — CSS escopado (porte do `<style>` do protótipo).
- `enem-analytics/frontend/src/components/compare/report/useReportData.ts` — coleta paralela A&B + flags de disponibilidade.
- `enem-analytics/frontend/src/components/compare/report/generateReportPdf.ts` — render off-screen → `html2pdf` → download.
- `enem-analytics/frontend/src/components/compare/report/reportMetrics.test.ts` — unit tests.
- `enem-analytics/frontend/src/components/compare/report/reportNarrative.test.ts` — unit tests.
- `enem-analytics/frontend/public/logo-xtri.png` — copiar de `.superpowers/report-prototype/logo-xtri.png`.
- `enem-analytics/frontend/vitest.config.ts` — config de unit tests.

**Modificar:**
- `enem-analytics/frontend/src/app/compare/page.tsx` — `handleExportPdf` passa a usar o novo fluxo, com **nomes reais** (`school1Name`/`school2Name` em vez de `displayLabel1/2`); estado de progresso.
- `enem-analytics/frontend/package.json` — script `test:unit` + devDeps vitest.

**Aposentar (remover ao fim da Fase 3):**
- `enem-analytics/frontend/src/components/compare/ExecutiveReportGenerator.tsx` (jsPDF).
- `enem-analytics/frontend/src/components/compare/PDFExportModal.tsx` (não usado pela página; confirmar antes de remover).

---

## FASE 0 — Setup e scaffolding

### Task 0.1: Adicionar vitest para unit tests

**Files:**
- Create: `enem-analytics/frontend/vitest.config.ts`
- Modify: `enem-analytics/frontend/package.json`

- [ ] **Step 1: Instalar devDeps**

Run:
```bash
cd enem-analytics/frontend && pnpm add -D vitest@^2 @vitest/coverage-v8
```

- [ ] **Step 2: Criar config**

`vitest.config.ts`:
```ts
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    coverage: { provider: 'v8', include: ['src/components/compare/report/**'] },
  },
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
});
```

- [ ] **Step 3: Adicionar script**

Em `package.json` `"scripts"`, adicionar: `"test:unit": "vitest run"`.

- [ ] **Step 4: Verificar runner**

Run: `pnpm test:unit`
Expected: "No test files found" (ou 0 testes) — runner funciona, sem erro de config.

- [ ] **Step 5: Commit**

```bash
git add enem-analytics/frontend/vitest.config.ts enem-analytics/frontend/package.json enem-analytics/frontend/pnpm-lock.yaml
git commit -m "chore(frontend): add vitest for unit tests"
```

### Task 0.2: Copiar logo e portar CSS do protótipo

**Files:**
- Create: `enem-analytics/frontend/public/logo-xtri.png`
- Create: `enem-analytics/frontend/src/components/compare/report/ReportDocument.css`

- [ ] **Step 1: Copiar logo**

Run:
```bash
cp "/Volumes/KINGSTON/apps/RANKING ENEM/.superpowers/report-prototype/logo-xtri.png" \
   "/Volumes/KINGSTON/apps/RANKING ENEM/enem-analytics/frontend/public/logo-xtri.png"
```

- [ ] **Step 2: Portar o `<style>` do protótipo**

Copiar o conteúdo de `<style>` de `.superpowers/report-prototype/report.html` (linhas 6–32) para `ReportDocument.css`, prefixando todos os seletores com `.xtri-report ` para escopar (ex.: `.xtri-report .sec { ... }`, `.xtri-report table { ... }`). Adicionar ao fim os estilos dos extras:
```css
.xtri-report .statusbadge{display:inline-block;padding:1px 6px;border-radius:999px;font-size:7.5px;font-weight:700;margin-left:4px}
.xtri-report .st-exc{background:#dcfce7;color:#15803d}
.xtri-report .st-good{background:#dbeafe;color:#1d4ed8}
.xtri-report .st-att{background:#fef3c7;color:#b45309}
.xtri-report .st-crit{background:#fee2e2;color:#b91c1c}
```

- [ ] **Step 3: Commit**

```bash
git add enem-analytics/frontend/public/logo-xtri.png enem-analytics/frontend/src/components/compare/report/ReportDocument.css
git commit -m "chore(report): add xtri logo and scoped report css from validated prototype"
```

### Task 0.3: Tipos agregados do relatório

**Files:**
- Create: `enem-analytics/frontend/src/components/compare/report/types.ts`

- [ ] **Step 1: Definir tipos** (reusando os já existentes em `@/lib/api`)

```ts
import type {
  DiagnosisComparisonResult, SchoolHistory,
} from '@/lib/api';

export interface ReportSchoolMeta {
  codigo_inep: string;
  nome_escola: string;        // NOME REAL (não "Escola 1/2")
  uf: string | null;
  cidade?: string | null;
  tipo_escola: string | null;
  porte_label?: string | null;
  nota_media: number | null;
  ranking_brasil: number | null;
  ranking_uf: number | null;
  overall_health: 'excellent' | 'good' | 'needs_attention' | 'critical' | null;
}

export interface ComparisonYearRow {
  ano: number;
  a_media: number | null; a_rank: number | null;
  b_media: number | null; b_rank: number | null;
}

// Seções avançadas (Fases 2-3) são opcionais — degradação graciosa.
export interface ReportData {
  generatedAt: Date;
  baseYear: number;
  schoolA: ReportSchoolMeta;
  schoolB: ReportSchoolMeta;
  diagnosis: DiagnosisComparisonResult;     // area_comparison + status
  history: ComparisonYearRow[];
  // Fase 2+:
  projection?: ProjectionRow[];
  redacaoCompetencias?: RedacaoCompRow[];
  recommendations?: RecommendationRow[];
  skills?: { a: SkillRow[]; b: SkillRow[] };
}

export interface ProjectionRow {
  area: string; area_name: string;
  a_current: number | null; a_projected: number | null; a_gain: number | null;
  b_current: number | null; b_projected: number | null; b_gain: number | null;
  focus_content: string;       // conteúdo de maior gap
}
export interface RedacaoCompRow {
  comp: 'C1'|'C2'|'C3'|'C4'|'C5'; label: string;
  a: number | null; b: number | null; nacional: number | null;
  reading: string;
}
export interface RecommendationRow {
  scope: 'A'|'B'|'Ambas'|'Benchmark'; priority: 'Alta'|'Média'|'Baixa'|'—';
  action: string; impact: string;
}
export interface SkillRow { area: string; skill: string; kind: 'forte'|'fraca'; }
```

- [ ] **Step 2: Verificar tipos compilam**

Run: `cd enem-analytics/frontend && npx tsc --noEmit`
Expected: sem erros novos nesse arquivo.

- [ ] **Step 3: Commit**

```bash
git add enem-analytics/frontend/src/components/compare/report/types.ts
git commit -m "feat(report): aggregate ReportData types"
```

---

## FASE 1 — Núcleo testável + pipeline + esqueleto (seções 1–3, 9, 10)

> Reusa **somente** dados já carregados pela página (`compareSchools`, `compareDiagnosis`, `getSchoolHistory` ×2). Produz um PDF já melhor que o atual: nomes reais, KPIs, análise por área com status, gráficos, histórico.

### Task 1.1: `reportMetrics.ts` — cálculos puros (TDD)

**Files:**
- Create: `enem-analytics/frontend/src/components/compare/report/reportMetrics.ts`
- Test: `enem-analytics/frontend/src/components/compare/report/reportMetrics.test.ts`

- [ ] **Step 1: Escrever testes (RED)**

```ts
import { describe, it, expect } from 'vitest';
import { areasWon, biggestGapArea, rankingGap, trendOverYears, statusClass, fmt } from './reportMetrics';
import type { DiagnosisComparisonResult } from '@/lib/api';

const area = (name: string, a: number, b: number, sa: any='good', sb: any='good') => ({
  area: name.slice(0,2).toUpperCase(), area_name: name,
  school_1_score: a, school_2_score: b, difference: a - b,
  school_1_status: sa, school_2_status: sb,
});
const diag = { area_comparison: [
  area('Redação', 671, 883), area('Matemática', 690, 863),
  area('Ciências Humanas', 720, 700), area('Linguagens', 620, 671),
  area('Ciências da Natureza', 615, 695),
]} as unknown as DiagnosisComparisonResult;

describe('areasWon', () => {
  it('conta vitórias de cada escola e empates', () => {
    expect(areasWon(diag)).toEqual({ a: 1, b: 4, ties: 0 });
  });
});
describe('biggestGapArea', () => {
  it('retorna a área de maior |diferença| com o vencedor', () => {
    const r = biggestGapArea(diag);
    expect(r.area_name).toBe('Matemática'); // |690-863|=173 é o maior
    expect(r.winner).toBe('B');
    expect(r.gap).toBe(173);
  });
});
describe('rankingGap', () => {
  it('valor absoluto da diferença de ranking', () => {
    expect(rankingGap(604, 2)).toBe(602);
    expect(rankingGap(null, 2)).toBeNull();
  });
});
describe('trendOverYears', () => {
  it('último menos primeiro ano válido', () => {
    expect(trendOverYears([631, 638, null, 654, 663])).toBe(32);
    expect(trendOverYears([null])).toBeNull();
  });
});
describe('statusClass', () => {
  it('mapeia status para classe CSS', () => {
    expect(statusClass('excellent')).toBe('st-exc');
    expect(statusClass('critical')).toBe('st-crit');
  });
});
describe('fmt', () => {
  it('formata nota pt-BR com 1 casa', () => {
    expect(fmt(663.23)).toBe('663,2');
    expect(fmt(null)).toBe('—');
  });
});
```

- [ ] **Step 2: Rodar — falha**

Run: `cd enem-analytics/frontend && pnpm test:unit src/components/compare/report/reportMetrics.test.ts`
Expected: FAIL ("não exportado").

- [ ] **Step 3: Implementar**

```ts
import type { DiagnosisComparisonResult, DiagnosisComparisonArea } from '@/lib/api';

export type Winner = 'A' | 'B' | 'tie';
type Status = 'excellent' | 'good' | 'needs_attention' | 'critical';

export function areasWon(diag: DiagnosisComparisonResult) {
  let a = 0, b = 0, ties = 0;
  for (const ar of diag.area_comparison) {
    if (ar.difference > 0) a++;
    else if (ar.difference < 0) b++;
    else ties++;
  }
  return { a, b, ties };
}

export function biggestGapArea(diag: DiagnosisComparisonResult) {
  const top = [...diag.area_comparison].sort(
    (x, y) => Math.abs(y.difference) - Math.abs(x.difference),
  )[0];
  return {
    area_name: top.area_name,
    gap: Math.abs(top.difference),
    winner: (top.difference > 0 ? 'A' : top.difference < 0 ? 'B' : 'tie') as Winner,
  };
}

export function rankingGap(a: number | null, b: number | null): number | null {
  if (a == null || b == null) return null;
  return Math.abs(a - b);
}

export function trendOverYears(notas: (number | null)[]): number | null {
  const valid = notas.filter((n): n is number => n != null);
  if (valid.length < 2) return null;
  return Math.round((valid[valid.length - 1] - valid[0]) * 10) / 10;
}

export function statusClass(s: Status | null): string {
  return { excellent: 'st-exc', good: 'st-good', needs_attention: 'st-att', critical: 'st-crit' }[s ?? 'good'] ?? 'st-good';
}
export function statusLabel(s: Status | null): string {
  return { excellent: 'Excelente', good: 'Bom', needs_attention: 'Atenção', critical: 'Crítico' }[s ?? 'good'] ?? '—';
}

export function fmt(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—';
  return n.toLocaleString('pt-BR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

export function winnerOfArea(ar: DiagnosisComparisonArea): Winner {
  return ar.difference > 0 ? 'A' : ar.difference < 0 ? 'B' : 'tie';
}
```

- [ ] **Step 4: Rodar — passa**

Run: `pnpm test:unit src/components/compare/report/reportMetrics.test.ts`
Expected: PASS (todos).

- [ ] **Step 5: Commit**

```bash
git add enem-analytics/frontend/src/components/compare/report/reportMetrics.ts enem-analytics/frontend/src/components/compare/report/reportMetrics.test.ts
git commit -m "feat(report): pure comparison metrics with tests"
```

### Task 1.2: `reportNarrative.ts` — geração de texto data-driven (TDD)

**Files:**
- Create: `enem-analytics/frontend/src/components/compare/report/reportNarrative.ts`
- Test: `enem-analytics/frontend/src/components/compare/report/reportNarrative.test.ts`

> Substitui as narrativas **hand-written** do protótipo por geração templada a partir dos dados, mantendo o tom analítico (quem lidera, por quanto, onde se concentra, o que fazer).

- [ ] **Step 1: Escrever testes (RED)**

```ts
import { describe, it, expect } from 'vitest';
import { executiveSummary, areaParagraph } from './reportNarrative';
import type { ReportData } from './types';

const base = {
  schoolA: { nome_escola: 'FARIAS BRITO', nota_media: 663.2, ranking_brasil: 604 },
  schoolB: { nome_escola: 'CHRISTUS', nota_media: 762.4, ranking_brasil: 2 },
  diagnosis: { area_comparison: [
    { area:'RED', area_name:'Redação', school_1_score:671, school_2_score:883, difference:-212, school_1_status:'good', school_2_status:'excellent' },
    { area:'CH', area_name:'Ciências Humanas', school_1_score:720, school_2_score:700, difference:20, school_1_status:'excellent', school_2_status:'good' },
  ] },
} as unknown as ReportData;

describe('executiveSummary', () => {
  it('nomeia o líder, a vantagem e as áreas vencidas', () => {
    const s = executiveSummary(base);
    expect(s).toContain('CHRISTUS');
    expect(s).toContain('99,2');       // |663.2-762.4|
    expect(s).toMatch(/Redação/);      // maior gap mencionado
  });
});
describe('areaParagraph', () => {
  it('descreve a área e o vencedor com a diferença', () => {
    const p = areaParagraph(base.diagnosis.area_comparison[0], base);
    expect(p).toMatch(/212/);
    expect(p).toContain('CHRISTUS');   // vencedor da Redação (B)
  });
});
```

- [ ] **Step 2: Rodar — falha** (`pnpm test:unit ...reportNarrative.test.ts`) → FAIL.

- [ ] **Step 3: Implementar** (`reportNarrative.ts`)

```ts
import type { ReportData } from './types';
import type { DiagnosisComparisonArea } from '@/lib/api';
import { areasWon, biggestGapArea, fmt } from './reportMetrics';

export function executiveSummary(d: ReportData): string {
  const a = d.schoolA, b = d.schoolB;
  const leader = (a.nota_media ?? 0) >= (b.nota_media ?? 0) ? a : b;
  const diff = Math.abs((a.nota_media ?? 0) - (b.nota_media ?? 0));
  const won = areasWon(d.diagnosis);
  const wonLeader = leader === a ? won.a : won.b;
  const big = biggestGapArea(d.diagnosis);
  return `${leader.nome_escola} (média ${fmt(leader.nota_media)} · #${leader.ranking_brasil ?? '—'} no Brasil) ` +
    `lidera por ${fmt(diff)} pontos na média geral, vencendo em ${wonLeader} das ${d.diagnosis.area_comparison.length} áreas. ` +
    `A maior diferença está em ${big.area_name} (${fmt(big.gap)} pts). ` +
    `As demais áreas são mais equilibradas; a leitura detalhada por área aponta onde concentrar esforço.`;
}

export function areaParagraph(ar: DiagnosisComparisonArea, d: ReportData): string {
  const winnerName = ar.difference > 0 ? d.schoolA.nome_escola : ar.difference < 0 ? d.schoolB.nome_escola : null;
  const gap = Math.abs(ar.difference);
  if (!winnerName) return `${ar.area_name}: empate técnico (${fmt(ar.school_1_score)} × ${fmt(ar.school_2_score)}). Disputa aberta.`;
  return `Em ${ar.area_name}, ${winnerName} vence por ${fmt(gap)} pts ` +
    `(${fmt(ar.school_1_score)} × ${fmt(ar.school_2_score)}). ` +
    `Onde agir: priorizar a escola com menor nota nesta área para reduzir a distância geral.`;
}
```

- [ ] **Step 4: Rodar — passa.** **Step 5: Commit** `feat(report): data-driven narrative generation with tests`.

### Task 1.3: `reportTheme.ts` — constantes de estilo

**Files:** Create `.../report/reportTheme.ts`

- [ ] **Step 1: Implementar**
```ts
export const REPORT_COLORS = {
  blue: '#2563eb', green: '#16a34a', cyan: '#2ba9df', coral: '#ff6b5c',
  ink: '#0f172a', muted: '#94a3b8', zebra: '#f4f7fa',
} as const;
export const AREA_ORDER = ['LC', 'CH', 'CN', 'MT', 'RED'] as const;
```
- [ ] **Step 2: Commit** `feat(report): theme constants`.

### Task 1.4: `generateReportPdf.ts` — render off-screen → html2pdf

**Files:** Create `.../report/generateReportPdf.ts`

- [ ] **Step 1: Implementar** (mirror do download robusto do gerador atual)

```ts
import { createRoot } from 'react-dom/client';
import React from 'react';
import type { ReportData } from './types';
import ReportDocument from './ReportDocument';

export interface GeneratedReportFile { filename: string; }

export async function generateReportPdf(data: ReportData): Promise<GeneratedReportFile> {
  const html2pdf = (await import('html2pdf.js')).default;
  const container = document.createElement('div');
  container.style.position = 'fixed';
  container.style.left = '-10000px';
  container.style.top = '0';
  container.style.width = '210mm';
  document.body.appendChild(container);

  const root = createRoot(container);
  await new Promise<void>((resolve) => {
    root.render(React.createElement(ReportDocument, { data, onReady: resolve }));
  });
  // garante layout/charts montados
  await new Promise((r) => setTimeout(r, 400));

  const filename = `XTRI_Relatorio_${slug(data.schoolA.nome_escola)}_vs_${slug(data.schoolB.nome_escola)}.pdf`;
  await html2pdf().set({
    margin: 0,
    filename,
    image: { type: 'jpeg', quality: 0.96 },
    html2canvas: { scale: 2, useCORS: true, logging: false },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
    pagebreak: { mode: ['css', 'legacy'] },
  }).from(container.firstElementChild as HTMLElement).save();

  root.unmount();
  document.body.removeChild(container);
  return { filename };
}

function slug(s: string): string {
  return s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-zA-Z0-9]+/g, '_').slice(0, 18);
}
```

> Nota: `ReportDocument` deve aceitar `onReady?: () => void` e chamá-lo em `useEffect(()=>{ onReady?.(); },[])`.

- [ ] **Step 2: Commit** `feat(report): off-screen html2pdf pipeline`.

### Task 1.5: `useReportData.ts` — coleta (Fase 1: reusa o que a página já tem)

**Files:** Create `.../report/useReportData.ts`

- [ ] **Step 1: Implementar** (Fase 1 — monta `ReportData` a partir dos dados já carregados; sem fetch novo)

```ts
import type { ReportData, ComparisonYearRow, ReportSchoolMeta } from './types';
import type { DiagnosisComparisonResult, SchoolHistory } from '@/lib/api';

interface BuildArgs {
  diagnosis: DiagnosisComparisonResult;
  history1?: SchoolHistory; history2?: SchoolHistory;
  comparison?: { comparison: { ano: number; escola1: { nota_media: number|null; ranking: number|null }|null; escola2: { nota_media: number|null; ranking: number|null }|null }[] };
  nameA: string; nameB: string;
  ufA?: string|null; ufB?: string|null;
}

export function buildPhase1ReportData(args: BuildArgs): ReportData {
  const { diagnosis, history1, history2, comparison } = args;
  const lastA = history1?.history?.at(-1);
  const lastB = history2?.history?.at(-1);
  const meta = (info: DiagnosisComparisonResult['school_1'], name: string, uf: string|null|undefined, last: any): ReportSchoolMeta => ({
    codigo_inep: info.codigo_inep,
    nome_escola: name,                          // NOME REAL
    uf: uf ?? info.info.localizacao ?? null,
    cidade: info.info.localizacao ?? null,
    tipo_escola: info.info.tipo_escola,
    porte_label: info.info.porte != null ? String(info.info.porte) : null,
    nota_media: last?.nota_media ?? null,
    ranking_brasil: last?.ranking_brasil ?? null,
    ranking_uf: last?.ranking_uf ?? null,
    overall_health: info.overall_health ?? null,
  });
  const history: ComparisonYearRow[] = (comparison?.comparison ?? []).map((y) => ({
    ano: y.ano,
    a_media: y.escola1?.nota_media ?? null, a_rank: y.escola1?.ranking ?? null,
    b_media: y.escola2?.nota_media ?? null, b_rank: y.escola2?.ranking ?? null,
  }));
  return {
    generatedAt: new Date(),
    baseYear: diagnosis.school_1.info.ano,
    schoolA: meta(diagnosis.school_1, args.nameA, args.ufA, lastA),
    schoolB: meta(diagnosis.school_2, args.nameB, args.ufB, lastB),
    diagnosis,
    history,
  };
}
```

- [ ] **Step 2: Commit** `feat(report): phase-1 report data assembly`.

### Task 1.6: `ReportDocument.tsx` — esqueleto + seções 1–3, 9, 10

**Files:** Create `.../report/ReportDocument.tsx`

> **Porte direto** de `.superpowers/report-prototype/report.html`. Cada `<div class="sec">` do protótipo vira um sub-bloco JSX; literais hardcoded são substituídos por bindings. Importar `./ReportDocument.css`.

- [ ] **Step 1: Implementar o componente** com a estrutura abaixo (Fase 1 cobre Header, Resumo Executivo, Escolas Comparadas + KPIs, Comparação Detalhada por área **com badge de status**, Visão Gráfica, Histórico). Seções avançadas (4–8, 11) ficam para Fases 2–3 e são renderizadas condicionalmente quando os dados existem.

```tsx
'use client';
import { useEffect } from 'react';
import './ReportDocument.css';
import type { ReportData } from './types';
import { areasWon, biggestGapArea, rankingGap, statusClass, statusLabel, fmt, winnerOfArea } from './reportMetrics';
import { executiveSummary, areaParagraph } from './reportNarrative';
import { AreaBars, EvolutionLine, AreaRadar } from './ReportCharts';

interface Props { data: ReportData; onReady?: () => void; }

export default function ReportDocument({ data: d, onReady }: Props) {
  useEffect(() => { onReady?.(); }, [onReady]);
  const won = areasWon(d.diagnosis);
  const big = biggestGapArea(d.diagnosis);
  const date = d.generatedAt.toLocaleDateString('pt-BR');

  return (
    <div className="xtri-report">
      <div className="header">
        <img src="/logo-xtri.png" alt="X-TRI" />
        <div>
          <h1>Relatório Comparativo de Escolas</h1>
          <div className="meta">Análise gerada em {date} | Base ENEM {d.baseYear} | X-TRI Escolas</div>
        </div>
      </div>

      <div className="sec">Resumo Executivo</div>
      <p className="an">{executiveSummary(d)}</p>

      <div className="sec">Escolas Comparadas</div>
      <table>
        <thead><tr><th style={{ width: '24%' }}>Campo</th><th>Escola A — {d.schoolA.nome_escola}</th><th>Escola B — {d.schoolB.nome_escola}</th></tr></thead>
        <tbody>
          <tr><td>INEP · UF · Cidade</td><td>{d.schoolA.codigo_inep} · {d.schoolA.uf} · {d.schoolA.cidade}</td><td>{d.schoolB.codigo_inep} · {d.schoolB.uf} · {d.schoolB.cidade}</td></tr>
          <tr><td>Tipo · Porte</td><td>{d.schoolA.tipo_escola} · {d.schoolA.porte_label}</td><td>{d.schoolB.tipo_escola} · {d.schoolB.porte_label}</td></tr>
          <tr><td>Média geral (TRI)</td><td className="a">{fmt(d.schoolA.nota_media)}</td><td className="b">{fmt(d.schoolB.nota_media)}</td></tr>
          <tr><td>Ranking Brasil · UF</td><td>#{d.schoolA.ranking_brasil} · #{d.schoolA.ranking_uf}</td><td>#{d.schoolB.ranking_brasil} · #{d.schoolB.ranking_uf}</td></tr>
        </tbody>
      </table>
      <div className="kpis">
        <div className="kpi"><div className="v" style={{ color: '#16a34a' }}>{fmt(Math.abs((d.schoolA.nota_media??0)-(d.schoolB.nota_media??0)))}</div><div className="l">Vantagem média</div></div>
        <div className="kpi"><div className="v">{won.a} × {won.b}</div><div className="l">Áreas A × B</div></div>
        <div className="kpi"><div className="v" style={{ color: '#ff6b5c' }}>{fmt(big.gap)}</div><div className="l">Maior gap ({big.area_name})</div></div>
        <div className="kpi"><div className="v">{rankingGap(d.schoolA.ranking_brasil, d.schoolB.ranking_brasil) ?? '—'}</div><div className="l">Gap de ranking</div></div>
      </div>

      <div className="sec">Comparação Detalhada — as 5 notas, uma a uma</div>
      {d.diagnosis.area_comparison.map((ar) => {
        const w = winnerOfArea(ar);
        const cls = w === 'A' ? 'win-a' : w === 'B' ? 'gap' : '';
        return (
          <div className={`areablock ${cls}`} key={ar.area}>
            <div className="areahead">
              <span className="t">{ar.area_name}
                <span className={`statusbadge ${statusClass(ar.school_1_status)}`}>A {statusLabel(ar.school_1_status)}</span>
                <span className={`statusbadge ${statusClass(ar.school_2_status)}`}>B {statusLabel(ar.school_2_status)}</span>
              </span>
              <span className="n"><span className="a">A {fmt(ar.school_1_score)}</span> · <span className="b">B {fmt(ar.school_2_score)}</span></span>
            </div>
            <p className="an">{areaParagraph(ar, d)}</p>
          </div>
        );
      })}

      <div className="sec">Visão Gráfica</div>
      <div className="grid2">
        <div><div className="cap">Notas por área — A (azul) × B (verde)</div><AreaBars diagnosis={d.diagnosis} /></div>
        <div><div className="cap">Radar das 5 áreas</div><AreaRadar diagnosis={d.diagnosis} /></div>
      </div>
      <div className="grid2">
        <div><div className="cap">Evolução da média</div><EvolutionLine history={d.history} /></div>
      </div>

      <div className="sec">Histórico Ano a Ano</div>
      <table>
        <thead><tr><th>Ano</th><th>Média A</th><th>Rank A</th><th>Média B</th><th>Rank B</th><th>Distância (B−A)</th></tr></thead>
        <tbody>
          {d.history.map((y) => (
            <tr key={y.ano}><td>{y.ano}</td><td className="a">{fmt(y.a_media)}</td><td>#{y.a_rank}</td><td className="b">{fmt(y.b_media)}</td><td>#{y.b_rank}</td>
              <td>{y.a_media != null && y.b_media != null ? fmt(y.b_media - y.a_media) : '—'}</td></tr>
          ))}
        </tbody>
      </table>

      {/* Seções 4–8 e 11 entram nas Fases 2–3 (render condicional) */}

      <div className="foot"><span>X-TRI Escolas · rankingenem.com</span><span>Base ENEM {d.baseYear}</span></div>
    </div>
  );
}
```

- [ ] **Step 2: Criar `ReportCharts.tsx`** com `AreaBars`, `EvolutionLine`, `AreaRadar`.

> Decisão: usar **SVG inline** (como o protótipo) para garantir render síncrono no html2pdf, evitando timing de animação do recharts. `AreaBars` e `EvolutionLine` portam o SVG das linhas 97–120 do protótipo, parametrizados; `AreaRadar` desenha um pentágono (extra do mockup). Cada um recebe os dados e normaliza para a viewBox.

```tsx
import type { DiagnosisComparisonResult } from '@/lib/api';
import type { ComparisonYearRow } from './types';

export function AreaBars({ diagnosis }: { diagnosis: DiagnosisComparisonResult }) {
  const areas = diagnosis.area_comparison;
  const max = Math.max(...areas.flatMap(a => [a.school_1_score, a.school_2_score]), 1);
  const h = (v: number) => (v / max) * 96;
  return (
    <svg viewBox="0 0 300 130" style={{ width: '100%', height: 120 }}>
      <line x1="28" y1="108" x2="298" y2="108" stroke="#e2e8f0" />
      {areas.map((a, i) => {
        const x = 38 + i * 54;
        return (
          <g key={a.area}>
            <rect x={x} y={108 - h(a.school_1_score)} width="15" height={h(a.school_1_score)} fill="#2563eb" />
            <rect x={x + 17} y={108 - h(a.school_2_score)} width="15" height={h(a.school_2_score)} fill="#16a34a" />
            <text x={x + 16} y="120" fontSize="8" fill="#64748b" textAnchor="middle">{a.area}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function EvolutionLine({ history }: { history: ComparisonYearRow[] }) {
  const ys = history.map(h => [h.a_media, h.b_media]).flat().filter((n): n is number => n != null);
  const min = Math.min(...ys, 0), max = Math.max(...ys, 1);
  const norm = (v: number) => 105 - ((v - min) / (max - min || 1)) * 90;
  const xs = (i: number) => 20 + (i * 276) / Math.max(history.length - 1, 1);
  const line = (key: 'a_media'|'b_media') => history.map((h, i) => h[key] != null ? `${xs(i)},${norm(h[key]!)}` : '').filter(Boolean).join(' ');
  return (
    <svg viewBox="0 0 300 120" style={{ width: '100%', height: 120 }}>
      <line x1="20" y1="105" x2="298" y2="105" stroke="#e2e8f0" />
      <polyline points={line('a_media')} fill="none" stroke="#2563eb" strokeWidth="2.2" />
      <polyline points={line('b_media')} fill="none" stroke="#16a34a" strokeWidth="2.2" />
      {history[0] && <text x="20" y="116" fontSize="7.5" fill="#94a3b8">{history[0].ano}</text>}
      {history.at(-1) && <text x="270" y="116" fontSize="7.5" fill="#94a3b8">{history.at(-1)!.ano}</text>}
    </svg>
  );
}

export function AreaRadar({ diagnosis }: { diagnosis: DiagnosisComparisonResult }) {
  const areas = diagnosis.area_comparison.slice(0, 5);
  const max = Math.max(...areas.flatMap(a => [a.school_1_score, a.school_2_score]), 1);
  const pt = (i: number, v: number, r = 95) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / areas.length;
    const rad = (v / max) * r;
    return `${150 + rad * Math.cos(ang)},${110 + rad * Math.sin(ang)}`;
  };
  const poly = (key: 'school_1_score'|'school_2_score') => areas.map((a, i) => pt(i, a[key])).join(' ');
  return (
    <svg viewBox="0 0 300 220" style={{ width: '100%', height: 200 }}>
      <polygon points={areas.map((_, i) => pt(i, max)).join(' ')} fill="none" stroke="#e2e8f0" />
      <polygon points={poly('school_1_score')} fill="rgba(37,99,235,.15)" stroke="#2563eb" strokeWidth="2" />
      <polygon points={poly('school_2_score')} fill="rgba(22,163,74,.13)" stroke="#16a34a" strokeWidth="2" />
      {areas.map((a, i) => { const [x, y] = pt(i, max * 1.12).split(','); return <text key={a.area} x={x} y={y} fontSize="8" fill="#64748b" textAnchor="middle">{a.area}</text>; })}
    </svg>
  );
}
```

- [ ] **Step 3: tsc check** — `npx tsc --noEmit` sem erros novos. **Step 4: Commit** `feat(report): ReportDocument skeleton + charts (phase 1 sections)`.

### Task 1.7: Ligar a página ao novo fluxo (nomes reais)

**Files:** Modify `enem-analytics/frontend/src/app/compare/page.tsx`

- [ ] **Step 1: Substituir `handleExportPdf`** para usar `buildPhase1ReportData` + `generateReportPdf`, passando **`school1Name`/`school2Name`** (reais) em vez de `displayLabel1/2`.

```tsx
// imports
import { buildPhase1ReportData } from '@/components/compare/report/useReportData';
import { generateReportPdf } from '@/components/compare/report/generateReportPdf';
// ...
const handleExportPdf = async () => {
  if (!school1 || !school2 || !diagnosisComparison) {
    alert('Selecione duas escolas e aguarde o diagnóstico carregar.');
    return;
  }
  setIsPdfExporting(true);
  try {
    const reportData = buildPhase1ReportData({
      diagnosis: diagnosisComparison,
      history1, history2, comparison,
      nameA: school1Name, nameB: school2Name,           // NOMES REAIS
      ufA: comparison?.escola1?.uf, ufB: comparison?.escola2?.uf,
    });
    await generateReportPdf(reportData);
    setPdfExportSuccess(true);
    setTimeout(() => setPdfExportSuccess(false), 2000);
  } catch (e) {
    console.error('Erro ao exportar PDF:', e);
    alert('Erro ao gerar o relatório. Tente novamente.');
  } finally {
    setIsPdfExporting(false);
  }
};
```

- [ ] **Step 2: Remover** o import e o uso de `generateExecutiveReport` na página (não apagar o arquivo ainda — só desligar).

- [ ] **Step 3: Verificação manual (build + visual)**

Run: `cd enem-analytics/frontend && pnpm build`
Expected: build passa.
Depois: `pnpm dev`, abrir `/compare`, logar como admin, escolher FARIAS BRITO vs ARI DE SÁ, clicar exportar → PDF baixa com nomes reais, KPIs, análise por área com badges, gráficos e histórico, **com acentos**.

- [ ] **Step 4: Commit** `feat(compare): wire new premium report (phase 1) with real school names`.

---

## FASE 2 — Projeção/ganho por TRI + conteúdos a focar (seções 4–5)

### Task 2.1: Documentar shapes de `getAreaProjection` / `getPredictionComparison`

**Files:** Modify `.../report/types.ts` (preencher `ProjectionRow` se necessário)

- [ ] **Step 1:** Ler em `enem-analytics/frontend/src/lib/api.ts` as assinaturas/tipos de `getAreaProjection` e `getPredictionComparison` (e os tipos `TRIAreaProjection`, `PredictionComparison`). Anotar campos reais: score atual, projetado, ganho, `stretch_content`/conteúdo de maior gap.
- [ ] **Step 2:** Ajustar `ProjectionRow` em `types.ts` para casar com os campos reais. **Step 3: Commit** `chore(report): confirm projection data shapes`.

### Task 2.2: `reportMetrics` — montagem das linhas de projeção (TDD)

**Files:** Modify `reportMetrics.ts` + `reportMetrics.test.ts`

- [ ] **Step 1: Teste (RED)** — `buildProjectionRows(projA, projB)` retorna 5 linhas com `a_current→a_projected (+gain)`, idem B, e `focus_content` por área. (escrever caso com 2 áreas e asserts em ganho e ordenação por maior ganho de A).
- [ ] **Step 2: Falha → Step 3: Implementar `buildProjectionRows` → Step 4: Passa → Step 5: Commit** `feat(report): projection rows builder with tests`.

### Task 2.3: `useReportData` busca projeção (paralelo) + seção no documento

**Files:** Modify `useReportData.ts`, `ReportDocument.tsx`

- [ ] **Step 1:** Criar `useReportData()` (hook) que, dadas as duas escolas, faz `Promise.all([api.getAreaProjection(a), api.getAreaProjection(b), api.getPredictionComparison(a,b)])` e popula `projection`. Reusar dados da Fase 1. Expor `{ data, isLoading, progress }`.
- [ ] **Step 2:** Em `ReportDocument`, renderizar a seção **"Ganho Projetado por TRI (ENEM {ano+1}) & Conteúdos a Focar"** (porte da tabela linhas 124–135 do protótipo) **somente se** `d.projection`. Parágrafo via `reportNarrative.projectionParagraph`.
- [ ] **Step 3:** Página passa a usar o hook (com indicador "coletando análises…"). **Step 4: build + visual. Step 5: Commit** `feat(report): TRI projection section (phase 2)`.

---

## FASE 3 — Redação C1–C5, habilidades, recomendações, síntese (seções 6–8, 11)

### Task 3.1: `useReportData` — coleta avançada paralela (graceful)

**Files:** Modify `useReportData.ts`

- [ ] **Step 1:** Acrescentar ao `Promise.allSettled` (não `all` — degradação graciosa): `getGlinerStudyFocus(a)`, `getGlinerStudyFocus(b)` (redação C1–C5 + foco), `getSchoolSkills`/`getWorstSkills` (a,b), `getRecommendations`/`getQuickWins` (a,b), `getSchoolCluster` (a,b). Cada falha → seção omitida.
- [ ] **Step 2:** Mapear o resultado de `study-focus` (tipo já existe em api.ts, ramo `kind:'redacao_competencias'`) para `RedacaoCompRow[]`.
- [ ] **Step 3: Commit** `feat(report): advanced parallel data collection (allSettled)`.

### Task 3.2: `reportMetrics`/`reportNarrative` — C1–C5 + recomendações (TDD)

**Files:** Modify metrics/narrative + tests

- [ ] **Step 1: Testes (RED)** para `buildRedacaoRows(studyA, studyB)` (5 linhas C1–C5 com diff B−A e "leitura" do maior gargalo) e `buildRecommendations(d)` (lista priorizada A·Alta/Média/Baixa, B, Ambas, Benchmark ligadas aos maiores gaps).
- [ ] **Step 2–4: Falha → implementar → passa. Step 5: Commit** `feat(report): redação C1-C5 and recommendations builders with tests`.

### Task 3.3: Seções no documento (6, 7, 8, 11)

**Files:** Modify `ReportDocument.tsx`

- [ ] **Step 1:** Renderizar condicionalmente: **Redação por Competência (C1–C5)** (tabela linhas 137–148 do protótipo), **Habilidades** (fortes/fracas A e B), **Recomendações & Próximos Passos** (tabela priorizada linhas 150–167), **Diagnóstico por Área (síntese acionável)** (sub-blocos coloridos juntando status+gap+ganho+conteúdo+recomendação). Todas envoltas por `{d.redacaoCompetencias && (...)}` etc.
- [ ] **Step 2: build + visual** (verificar quebras de página `page-break-inside:avoid` nos `.areablock`/tabelas). **Step 3: Commit** `feat(report): redação/skills/recommendations/synthesis sections (phase 3)`.

### Task 3.4: Aposentar o gerador jsPDF e o modal não usado

**Files:** Delete `ExecutiveReportGenerator.tsx`, `PDFExportModal.tsx`; Modify imports/`index.ts`

- [ ] **Step 1:** Confirmar que nada além da página importa `generateExecutiveReport`/`PDFExportModal` (`grep -rn "ExecutiveReportGenerator\|PDFExportModal" src`).
- [ ] **Step 2:** Remover os dois arquivos e quaisquer imports órfãos; ajustar `components/compare/index.ts` se exportar algum deles.
- [ ] **Step 3:** `npx tsc --noEmit` + `pnpm build` passam. **Step 4: Commit** `refactor(compare): retire jsPDF report generator and unused PDF modal`.

---

## FASE 4 — Verificação final

### Task 4.1: Verificação visual com 2 escolas reais

- [ ] **Step 1:** `pnpm dev`; `/compare` como admin; gerar relatório FARIAS BRITO × ARI DE SÁ.
- [ ] **Step 2:** Abrir o PDF e validar checklist: nomes reais; acentos corretos em todos os textos; 11 seções presentes (ou degradadas com nota se faltou dado); gráficos (barras/radar/linha) legíveis; quebras de página sem cortar blocos; rodapé/branding X-TRI; recomendações ligadas aos gaps reais.
- [ ] **Step 3:** Testar **degradação graciosa**: simular falha de `getGlinerStudyFocus` (ex.: escola sem dado) → o PDF ainda gera, sem a seção de redação, com nota.

### Task 4.2: Suite e build

- [ ] **Step 1:** `pnpm test:unit` (todas as fns puras) — PASS, cobertura do diretório `report/` ≥ 80%.
- [ ] **Step 2:** `pnpm build` — PASS.
- [ ] **Step 3:** Commit final / abrir PR.

---

## Self-review (cobertura vs spec)

| Seção da spec (§4) | Coberta por |
|---|---|
| 1. Escolas Comparadas (+cluster/saúde) | Task 1.6 (cluster entra na 3.1) |
| 2. Veredito (KPIs) | Task 1.6 |
| 3. Análise por Área (z-score/status) | Task 1.6 (+badges) |
| 4. Ganho/Projeção TRI | Tasks 2.1–2.3 |
| 5. Conteúdos TRI a focar | Task 2.3 (coluna focus_content) |
| 6. Habilidades | Tasks 3.1, 3.3 |
| 7. Conceitos & Redação C1–C5 | Tasks 3.1–3.3 |
| 8. Recomendações & Quick Wins | Tasks 3.2–3.3 |
| 9. Panorama Visual (gráficos) | Task 1.6 (AreaBars/Radar/Evolution) |
| 10. Histórico ano a ano | Task 1.6 |
| 11. Síntese acionável por área | Task 3.3 |
| Extras (radar, badges status) | Task 1.6 |
| Aposentar jsPDF | Task 3.4 |
| Nomes reais | Task 1.7 |

**Notas de risco conhecidas:** `getGlinerStudyFocus` instancia modelos ML por requisição (latência) → mitigado por `allSettled` + seções GLiNER ao final + indicador de progresso. Cluster (`getSchoolCluster`) é desejável mas opcional; entra na coleta da 3.1 e só preenche a linha "perfil" se vier.
