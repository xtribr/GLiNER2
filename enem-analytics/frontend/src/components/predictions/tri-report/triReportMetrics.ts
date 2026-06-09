import type { TRIProjectionInsight } from '@/lib/api';

/** Cor do nível de domínio TRI (0–1), em 5 faixas — espelha MasteryGauge.getColor de TRIAnalysis. */
export function masteryColor(level: number): string {
  if (level >= 0.8) return '#22c55e';
  if (level >= 0.6) return '#84cc16';
  if (level >= 0.4) return '#eab308';
  if (level >= 0.2) return '#f97316';
  return '#ef4444';
}

/** Nível 0–1 → porcentagem inteira (0.836 → 84). */
export function pct(level: number | null | undefined): number {
  return Math.round((level ?? 0) * 100);
}

export function trendLabel(dir: string): string {
  if (dir === 'ascending') return '↑ Crescente';
  if (dir === 'descending') return '↓ Decrescente';
  if (dir === 'stable') return '→ Estável';
  return '→ Dados insuf.';
}

export function trendColor(dir: string): string {
  if (dir === 'ascending') return '#16a34a';
  if (dir === 'descending') return '#dc2626';
  return '#6b7280';
}

/** Classe CSS do card de insight pela severidade. */
export function insightClass(type: TRIProjectionInsight['type']): string {
  return { positive: 'ins-pos', warning: 'ins-warn', info: 'ins-info', neutral: 'ins-neutral' }[type] ?? 'ins-neutral';
}

/** Delta assinado, 1 casa pt-BR, sinal Unicode: +16,7 / −5,0 / — para null. */
export function signed(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—';
  const r = Math.round(n * 10) / 10;
  const sign = r >= 0 ? '+' : '−';
  return `${sign}${Math.abs(r).toLocaleString('pt-BR', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}`;
}
