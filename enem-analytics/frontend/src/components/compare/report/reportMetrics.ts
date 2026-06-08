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
