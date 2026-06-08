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
