import type { ReportData, ProjectionRow } from './types';
import type { DiagnosisComparisonArea } from '@/lib/api';
import { areasWon, biggestGapArea, fmt, overallFromDiagnosis } from './reportMetrics';

export function executiveSummary(d: ReportData): string {
  // Líder e vantagem vêm das notas de área do diagnóstico (sempre presentes),
  // não de nota_media (history), que pode ser null e produzir um falso "lidera por ~600 pts".
  const overall = overallFromDiagnosis(d.diagnosis);
  const big = biggestGapArea(d.diagnosis);

  if (overall.leader === 'tie' || overall.advantage == null) {
    return `As duas escolas têm desempenho médio equivalente nas áreas avaliadas. ` +
      `A maior diferença pontual está em ${big.area_name} (${fmt(big.gap)} pts). ` +
      `A leitura detalhada por área aponta onde concentrar esforço.`;
  }

  const isA = overall.leader === 'A';
  const leaderName = isA ? d.schoolA.nome_escola : d.schoolB.nome_escola;
  const leaderMean = isA ? overall.aMean : overall.bMean;
  const leaderRank = isA ? d.schoolA.ranking_brasil : d.schoolB.ranking_brasil;
  const won = areasWon(d.diagnosis);
  const wonLeader = isA ? won.a : won.b;
  return `${leaderName} (média ${fmt(leaderMean)} · #${leaderRank ?? '—'} no Brasil) ` +
    `lidera por ${fmt(overall.advantage)} pontos na média geral, vencendo em ${wonLeader} das ${d.diagnosis.area_comparison.length} áreas. ` +
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
