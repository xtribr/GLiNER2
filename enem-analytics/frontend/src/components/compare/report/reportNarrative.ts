import type { ReportData, ProjectionRow } from './types';
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

export function projectionParagraph(rows: ProjectionRow[]): string {
  if (rows.length === 0) return '';

  // Find the area with biggest projected gain considering both schools
  let bestArea = rows[0];
  let bestGain = Math.max(rows[0].a.potential_gain ?? 0, rows[0].b.potential_gain ?? 0);
  for (const row of rows) {
    const gain = Math.max(row.a.potential_gain ?? 0, row.b.potential_gain ?? 0);
    if (gain > bestGain) { bestGain = gain; bestArea = row; }
  }

  // School with more upside: sum of potential_gain across all areas
  const totalGainA = rows.reduce((s, r) => s + (r.a.potential_gain ?? 0), 0);
  const totalGainB = rows.reduce((s, r) => s + (r.b.potential_gain ?? 0), 0);
  const moreUpside = totalGainA >= totalGainB ? 'A' : 'B';

  const targetYear = bestArea.target_year;

  return `Para o ENEM ${targetYear}, a área com maior potencial de ganho projetado é ` +
    `${bestArea.area_name} (até +${fmt(bestGain)} pts no cenário otimista). ` +
    `A Escola ${moreUpside} apresenta maior margem de evolução agregada ` +
    `(+${fmt(moreUpside === 'A' ? totalGainA : totalGainB)} pts nas áreas analisadas). ` +
    `As predições oficiais e os conteúdos a focar por área estão detalhados na tabela abaixo.`;
}
