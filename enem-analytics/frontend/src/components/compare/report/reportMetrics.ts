import type { DiagnosisComparisonResult, DiagnosisComparisonArea, TRIAreaProjection } from '@/lib/api';
import type { ProjectionRow, ProjectionCell, ProjectionFocusItem } from './types';

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

function nullCell(): ProjectionCell {
  return {
    current: null, recommended: null, potential_gain: null,
    scenarios: null, official_next: null, official_change: null,
    risk_level: null, trend_dir: null, trend_annual: null,
  };
}

function toCell(p: TRIAreaProjection): ProjectionCell {
  return {
    current: p.current_score,
    recommended: p.projection.recommended,
    potential_gain: p.projection.potential_gain,
    scenarios: {
      conservative: p.projection.scenarios.conservative,
      realistic: p.projection.scenarios.realistic,
      optimistic: p.projection.scenarios.optimistic,
    },
    official_next: p.official_prediction.display_score,
    official_change: p.official_prediction.display_expected_change,
    risk_level: p.official_prediction.risk_level,
    trend_dir: p.historical_analysis.trend.direction,
    trend_annual: p.historical_analysis.trend.annual_change,
  };
}

function toFocus(p: TRIAreaProjection): ProjectionFocusItem[] {
  return [...p.stretch_content.items]
    .sort((a, b) => b.gap - a.gap)
    .slice(0, 3)
    .map(({ skill, gap }) => ({ skill, gap }));
}

export function buildProjectionRows(
  projA: TRIAreaProjection[],
  projB: TRIAreaProjection[],
): ProjectionRow[] {
  if (projA.length === 0 && projB.length === 0) return [];

  const mapA = new Map(projA.map(p => [p.area, p]));
  const mapB = new Map(projB.map(p => [p.area, p]));

  // Union: A's areas first, then B-only areas
  const areas: string[] = [
    ...projA.map(p => p.area),
    ...projB.filter(p => !mapA.has(p.area)).map(p => p.area),
  ];

  return areas.map(area => {
    const a = mapA.get(area);
    const b = mapB.get(area);
    const meta = a ?? b!;
    return {
      area: meta.area,
      area_name: meta.area_name,
      target_year: a?.projection.target_year ?? b?.projection.target_year ?? 0,
      a: a ? toCell(a) : nullCell(),
      b: b ? toCell(b) : nullCell(),
      a_focus: a ? toFocus(a) : [],
      b_focus: b ? toFocus(b) : [],
    };
  });
}
