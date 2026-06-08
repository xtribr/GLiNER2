import type { DiagnosisComparisonResult, DiagnosisComparisonArea, TRIAreaProjection } from '@/lib/api';
import type { ProjectionRow, ProjectionCell, ProjectionFocusItem, ReportData, RedacaoCompRow, SkillRow, RecommendationRow } from './types';

// ── Local type aliases derived from api.ts return shapes ─────────────────────

type RedacaoFocusArea = {
  area: 'REDACAO';
  area_name: 'Redação';
  color: string;
  kind: 'redacao_competencias';
  n_redacoes: number;
  gargalo_principal: 'C1' | 'C2' | 'C3' | 'C4' | 'C5' | null;
  competencias: {
    competencia: 'C1' | 'C2' | 'C3' | 'C4' | 'C5';
    descricao: string;
    media_escola: number | null;
    media_nacional: number;
    gap: number | null;
    status: 'verde' | 'amarelo' | 'vermelho' | 'indisponivel';
  }[];
};

export type StudyFocusResult = {
  codigo_inep: string;
  focus_areas: (
    | {
        area: string; area_name: string; color: string;
        current_score: number; target_range: [number, number]; level: string;
        study_sequence: {
          concept: string; frequency: number; avg_difficulty: number;
          semantic_fields: string[]; lexical_fields: string[]; related_processes: string[];
          priority: 'high' | 'medium' | 'low'; estimated_impact: number;
        }[];
        total_concepts: number; estimated_total_impact: number;
      }
    | RedacaoFocusArea
  )[];
  total_estimated_improvement: number;
  study_plan: {
    phase_1: { name: string; description: string; concepts_count: number };
    phase_2: { name: string; description: string; concepts_count: number };
    phase_3: { name: string; description: string; concepts_count: number };
  };
};

export type SchoolSkillsResult = {
  codigo_inep: string;
  ano: number;
  total_skills: number;
  worst_overall: {
    area: string;
    skill_num: number;
    performance: number;
    national_avg: number | null;
    diff: number | null;
    descricao: string;
    status: 'above' | 'below' | 'equal';
  }[];
  by_area: Record<string, {
    skill_num: number;
    performance: number;
    national_avg: number | null;
    diff: number | null;
    descricao: string;
    status: 'above' | 'below' | 'equal';
  }[]>;
};

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

// ── buildRedacaoRows ──────────────────────────────────────────────────────────

function findRedacao(study: StudyFocusResult | null): RedacaoFocusArea | null {
  if (!study) return null;
  const found = study.focus_areas.find(
    (f): f is RedacaoFocusArea => 'kind' in f && f.kind === 'redacao_competencias',
  );
  return found ?? null;
}

export function buildRedacaoRows(
  studyA: StudyFocusResult | null,
  studyB: StudyFocusResult | null,
): RedacaoCompRow[] {
  const redA = findRedacao(studyA);
  const redB = findRedacao(studyB);
  if (!redA && !redB) return [];

  const comps = (['C1', 'C2', 'C3', 'C4', 'C5'] as const);

  return comps.map((comp): RedacaoCompRow => {
    const cA = redA?.competencias.find(c => c.competencia === comp) ?? null;
    const cB = redB?.competencias.find(c => c.competencia === comp) ?? null;
    const meta = cA ?? cB!;
    const aNota = cA?.media_escola ?? null;
    const bNota = cB?.media_escola ?? null;
    const nacional = meta.media_nacional;

    let reading = '';
    if (redA?.gargalo_principal === comp && redB?.gargalo_principal === comp) {
      reading = 'Maior gargalo de ambas as escolas';
    } else if (redA?.gargalo_principal === comp) {
      reading = 'Maior gargalo da Escola A';
    } else if (redB?.gargalo_principal === comp) {
      reading = 'Maior gargalo da Escola B';
    } else if (aNota != null && bNota != null && aNota >= nacional && bNota >= nacional) {
      reading = 'Ambas acima da média nacional';
    }

    return { comp, label: meta.descricao, a: aNota, b: bNota, nacional, reading };
  });
}

// ── buildSkillRows ────────────────────────────────────────────────────────────

export function buildSkillRows(skills: SchoolSkillsResult | null): SkillRow[] {
  if (!skills) return [];

  const fracas: SkillRow[] = skills.worst_overall
    .filter(s => s.status === 'below')
    .slice(0, 3)
    .map(s => ({ area: s.area, skill: s.descricao, kind: 'fraca' as const }));

  const seenDescricao = new Set(fracas.map(f => f.skill));

  const allAbove = Object.entries(skills.by_area).flatMap(([area, items]) =>
    items
      .filter(s => s.status === 'above')
      .map(s => ({ area, skill: s.descricao, performance: s.performance })),
  );

  const fortes: SkillRow[] = [...allAbove]
    .sort((x, y) => y.performance - x.performance)
    .filter(s => !seenDescricao.has(s.skill))
    .slice(0, 3)
    .map(s => ({ area: s.area, skill: s.skill, kind: 'forte' as const }));

  return [...fracas, ...fortes];
}

// ── buildRecommendations ──────────────────────────────────────────────────────

export function buildRecommendations(d: ReportData): RecommendationRow[] {
  const aMedia = d.schoolA.nota_media ?? 0;
  const bMedia = d.schoolB.nota_media ?? 0;

  const leaderScope = aMedia >= bMedia ? 'A' : 'B';
  const laggardScope = leaderScope === 'A' ? 'B' : 'A';
  const leaderName = leaderScope === 'A' ? d.schoolA.nome_escola : d.schoolB.nome_escola;
  const laggardName = laggardScope === 'A' ? d.schoolA.nome_escola : d.schoolB.nome_escola;

  // Areas where laggard loses (difference sign tells us: positive = A wins, negative = B wins)
  const laggardLoses = d.diagnosis.area_comparison.filter(ar =>
    laggardScope === 'B' ? ar.difference > 0 : ar.difference < 0,
  );

  // Sort by absolute gap descending to determine top-2
  const sortedByGap = [...laggardLoses].sort(
    (x, y) => Math.abs(y.difference) - Math.abs(x.difference),
  );

  const projMap = new Map(
    (d.projection ?? []).map(p => [p.area, p]),
  );

  const laggardAreaRecs: RecommendationRow[] = sortedByGap.map((ar, idx) => {
    const priority = idx < 2 ? 'Alta' : 'Média';
    const projRow = projMap.get(ar.area);
    const projCell = projRow ? (laggardScope === 'B' ? projRow.b : projRow.a) : null;
    const gainStr = projCell?.potential_gain != null
      ? `+${projCell.potential_gain} pts em ${ar.area_name}`
      : `reduz gap de ${Math.abs(ar.difference)} pts`;
    return {
      scope: laggardScope,
      priority,
      action: `Reforço focado em ${ar.area_name}`,
      impact: gainStr,
    };
  });

  // Redação recommendation for laggard's worst competência
  const redacaoRecs: RecommendationRow[] = [];
  if (d.redacaoCompetencias && d.redacaoCompetencias.length > 0) {
    const notaField = laggardScope === 'A' ? 'a' : 'b';
    // Find the competência where laggard is furthest below nacional
    const worst = [...d.redacaoCompetencias]
      .filter(r => r[notaField] != null)
      .sort((x, y) => {
        const gapX = (x[notaField] ?? 0) - (x.nacional ?? 0);
        const gapY = (y[notaField] ?? 0) - (y.nacional ?? 0);
        return gapX - gapY; // most negative first
      })[0] ?? null;
    if (worst) {
      redacaoRecs.push({
        scope: laggardScope,
        priority: 'Alta',
        action: `Oficinas de ${worst.label} (redação)`,
        impact: 'fecha gargalo de redação',
      });
    }
  }

  // Leader's strongest area (highest score among its areas)
  const leaderAreaScores = d.diagnosis.area_comparison.map(ar => ({
    area_name: ar.area_name,
    score: leaderScope === 'A' ? ar.school_1_score : ar.school_2_score,
  }));
  const leaderStrongestArea = [...leaderAreaScores].sort((x, y) => y.score - x.score)[0] ?? null;

  const monitoringRec: RecommendationRow = {
    scope: 'Ambas',
    priority: '—',
    action: 'Monitoramento diagnóstico mensal por área para reajustar trilhas',
    impact: 'estrutural',
  };

  if (!leaderStrongestArea) {
    // No area data — return only the recs that don't depend on area names
    return [...redacaoRecs, monitoringRec];
  }

  const benchmarkRec: RecommendationRow = {
    scope: 'Benchmark',
    priority: '—',
    action: `${laggardName} espelhar a metodologia de ${leaderStrongestArea.area_name} da ${leaderName}`,
    impact: 'estrutural',
  };

  // Leader preserve its strongest area
  const leaderPreserveRec: RecommendationRow = {
    scope: leaderScope,
    priority: 'Baixa',
    action: `Preservar desempenho em ${leaderStrongestArea.area_name}`,
    impact: `mantém liderança em ${leaderStrongestArea.area_name}`,
  };

  return [
    ...laggardAreaRecs,
    ...redacaoRecs,
    benchmarkRec,
    monitoringRec,
    leaderPreserveRec,
  ];
}
