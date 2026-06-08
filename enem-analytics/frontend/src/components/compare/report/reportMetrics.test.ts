import { describe, it, expect } from 'vitest';
import { areasWon, biggestGapArea, rankingGap, trendOverYears, statusClass, fmt, buildProjectionRows, buildRedacaoRows, buildSkillRows, buildRecommendations } from './reportMetrics';
import type { DiagnosisComparisonResult, TRIAreaProjection } from '@/lib/api';
import type { ReportData, ReportSchoolMeta } from './types';

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
    expect(r.area_name).toBe('Redação'); // |671-883|=212 é o maior gap
    expect(r.winner).toBe('B');
    expect(r.gap).toBe(212);
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

// --- buildProjectionRows ---

function makeProj(overrides: Partial<TRIAreaProjection> & { area: string; area_name: string }): TRIAreaProjection {
  return {
    codigo_inep: '12345',
    color: '#333',
    current_year: 2024,
    current_score: 600,
    historical_analysis: {
      total_years: 3,
      scores: [],
      trend: { direction: 'ascending', annual_change: 5.5, strength: 0.8, r_squared: 0.7 },
      statistics: { mean: 595, std: 10, min: 580, max: 610, avg_improvement: 5, max_improvement: 10 },
    },
    stretch_content: {
      total_items: 3,
      items: [
        { skill: 'H1', tri_score: 400, description: 'desc1', gap: 30 },
        { skill: 'H2', tri_score: 420, description: 'desc2', gap: 50 },
        { skill: 'H3', tri_score: 380, description: 'desc3', gap: 10 },
        { skill: 'H4', tri_score: 390, description: 'desc4', gap: 40 },
      ],
      tri_range: { min: 350, max: 650 },
    },
    projection: {
      target_year: 2026,
      scenarios: { trend_based: 615, conservative: 605, realistic: 618, optimistic: 630 },
      recommended: 618,
      confidence_interval: { low: 610, high: 625 },
      potential_gain: 18,
    },
    official_prediction: {
      target_year: 2026,
      current_score: 600,
      raw_score: 598,
      display_score: 620,
      confidence_interval: { low: 610, high: 630 } as any,
      raw_confidence_interval: { low: 588, high: 608 } as any,
      display_mode: 'delta',
      regime: 'regular',
      risk_level: 'normal',
      risk_reason: null,
      badge_text: null,
      historical_corridor: { low: 580, high: 640 } as any,
      raw_expected_change: 10,
      display_expected_change: 20,
      model_info: {},
    },
    insights: [],
    ...overrides,
  };
}

const projA1 = makeProj({ area: 'MT', area_name: 'Matemática', current_score: 700, projection: { target_year: 2026, scenarios: { trend_based: 715, conservative: 708, realistic: 720, optimistic: 735 }, recommended: 720, confidence_interval: { low: 712, high: 728 }, potential_gain: 20 } });
const projA2 = makeProj({ area: 'LC', area_name: 'Linguagens', current_score: 650 });
const projB1 = makeProj({ area: 'MT', area_name: 'Matemática', current_score: 680, official_prediction: { ...makeProj({ area: 'MT', area_name: 'Matemática' }).official_prediction, display_score: 695, display_expected_change: 15, risk_level: 'conservative' } });
const projB2 = makeProj({ area: 'CN', area_name: 'Ciências da Natureza', current_score: 610 });

describe('buildProjectionRows', () => {
  it('retorna [] se ambos arrays vazios', () => {
    expect(buildProjectionRows([], [])).toEqual([]);
  });

  it('produz union de áreas (A primeiro, depois exclusivas de B)', () => {
    const rows = buildProjectionRows([projA1, projA2], [projB1, projB2]);
    expect(rows).toHaveLength(3);
    expect(rows.map(r => r.area)).toEqual(['MT', 'LC', 'CN']);
  });

  it('mapeia célula A corretamente (current, recommended, scenarios, official_next, trend)', () => {
    const rows = buildProjectionRows([projA1], [projB1]);
    const cell = rows[0].a;
    expect(cell.current).toBe(700);
    expect(cell.recommended).toBe(720);
    expect(cell.potential_gain).toBe(20);
    expect(cell.scenarios).toEqual({ conservative: 708, realistic: 720, optimistic: 735 });
    expect(cell.official_next).toBe(620);
    expect(cell.official_change).toBe(20);
    expect(cell.risk_level).toBe('normal');
    expect(cell.trend_dir).toBe('ascending');
    expect(cell.trend_annual).toBe(5.5);
  });

  it('mapeia célula B corretamente', () => {
    const rows = buildProjectionRows([projA1], [projB1]);
    const cell = rows[0].b;
    expect(cell.current).toBe(680);
    expect(cell.official_next).toBe(695);
    expect(cell.official_change).toBe(15);
    expect(cell.risk_level).toBe('conservative');
  });

  it('target_year vem da projeção de A; fallback para B quando A ausente', () => {
    const rows = buildProjectionRows([projA1, projA2], [projB1, projB2]);
    expect(rows[0].target_year).toBe(2026);  // MT: A tem target_year 2026
    expect(rows[2].target_year).toBe(2026);  // CN: só B, fallback de B
  });

  it('a_focus ordenado por gap desc, máximo 3 itens', () => {
    const rows = buildProjectionRows([projA1], [projB1]);
    // stretch_content items: H1=30, H2=50, H3=10, H4=40 → desc: H2(50), H4(40), H1(30)
    expect(rows[0].a_focus).toEqual([
      { skill: 'H2', gap: 50 },
      { skill: 'H4', gap: 40 },
      { skill: 'H1', gap: 30 },
    ]);
  });

  it('célula all-null quando escola não tem aquela área', () => {
    const rows = buildProjectionRows([projA1, projA2], [projB1, projB2]);
    // LC: só A tem, B deve ser all-null
    const lcRow = rows.find(r => r.area === 'LC')!;
    expect(lcRow.b.current).toBeNull();
    expect(lcRow.b.recommended).toBeNull();
    expect(lcRow.b.scenarios).toBeNull();
    expect(lcRow.b.official_next).toBeNull();
    expect(lcRow.b.trend_dir).toBeNull();
    // CN: só B tem, A deve ser all-null
    const cnRow = rows.find(r => r.area === 'CN')!;
    expect(cnRow.a.current).toBeNull();
    expect(cnRow.a_focus).toEqual([]);
  });

  it('não mutaciona os arrays de entrada', () => {
    const a = [projA1, projA2];
    const b = [projB1, projB2];
    const lenA = a.length;
    const lenB = b.length;
    buildProjectionRows(a, b);
    expect(a).toHaveLength(lenA);
    expect(b).toHaveLength(lenB);
    expect(a[0]).toBe(projA1);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// buildRedacaoRows
// ─────────────────────────────────────────────────────────────────────────────

type StudyFocusReturn = {
  codigo_inep: string;
  focus_areas: (
    | {
        area: string; area_name: string; color: string;
        current_score: number; target_range: [number, number]; level: string;
        study_sequence: { concept: string; frequency: number; avg_difficulty: number;
          semantic_fields: string[]; lexical_fields: string[]; related_processes: string[];
          priority: 'high' | 'medium' | 'low'; estimated_impact: number; }[];
        total_concepts: number; estimated_total_impact: number;
      }
    | {
        area: 'REDACAO'; area_name: 'Redação'; color: string;
        kind: 'redacao_competencias'; n_redacoes: number;
        gargalo_principal: 'C1' | 'C2' | 'C3' | 'C4' | 'C5' | null;
        competencias: {
          competencia: 'C1' | 'C2' | 'C3' | 'C4' | 'C5';
          descricao: string;
          media_escola: number | null;
          media_nacional: number;
          gap: number | null;
          status: 'verde' | 'amarelo' | 'vermelho' | 'indisponivel';
        }[];
      }
  )[];
  total_estimated_improvement: number;
  study_plan: {
    phase_1: { name: string; description: string; concepts_count: number };
    phase_2: { name: string; description: string; concepts_count: number };
    phase_3: { name: string; description: string; concepts_count: number };
  };
};

function makeStudyFocus(gargalo: 'C1'|'C2'|'C3'|'C4'|'C5'|null, mediaEscola: number[]): StudyFocusReturn {
  return {
    codigo_inep: '11111',
    total_estimated_improvement: 10,
    study_plan: {
      phase_1: { name: 'p1', description: 'd', concepts_count: 1 },
      phase_2: { name: 'p2', description: 'd', concepts_count: 1 },
      phase_3: { name: 'p3', description: 'd', concepts_count: 1 },
    },
    focus_areas: [
      {
        area: 'REDACAO', area_name: 'Redação', color: '#000',
        kind: 'redacao_competencias', n_redacoes: 50,
        gargalo_principal: gargalo,
        competencias: (['C1','C2','C3','C4','C5'] as const).map((c, i) => ({
          competencia: c,
          descricao: `Descrição ${c}`,
          media_escola: mediaEscola[i] ?? null,
          media_nacional: 120,
          gap: mediaEscola[i] != null ? mediaEscola[i] - 120 : null,
          status: (mediaEscola[i] ?? 0) >= 120 ? 'verde' : 'vermelho',
        })),
      },
    ],
  };
}

const studyA = makeStudyFocus('C3', [130, 125, 100, 115, 118]);
const studyB = makeStudyFocus('C5', [128, 122, 115, 119, 95]);

describe('buildRedacaoRows', () => {
  it('retorna [] quando ambos são null', () => {
    expect(buildRedacaoRows(null, null)).toEqual([]);
  });

  it('produz 5 linhas (C1..C5) com dados corretos', () => {
    const rows = buildRedacaoRows(studyA, studyB);
    expect(rows).toHaveLength(5);
    expect(rows[0].comp).toBe('C1');
    expect(rows[0].a).toBe(130);
    expect(rows[0].b).toBe(128);
    expect(rows[0].nacional).toBe(120);
    expect(rows[0].label).toBe('Descrição C1');
  });

  it('reading menciona "gargalo" para o gargalo de A (C3)', () => {
    const rows = buildRedacaoRows(studyA, studyB);
    const c3 = rows.find(r => r.comp === 'C3')!;
    expect(c3.reading).toMatch(/gargalo/i);
    expect(c3.reading).toMatch(/escola a/i);
  });

  it('reading menciona "gargalo" para o gargalo de B (C5)', () => {
    const rows = buildRedacaoRows(studyA, studyB);
    const c5 = rows.find(r => r.comp === 'C5')!;
    expect(c5.reading).toMatch(/gargalo/i);
    expect(c5.reading).toMatch(/escola b/i);
  });

  it('quando ambas acima do nacional, reading indica isso', () => {
    const rows = buildRedacaoRows(studyA, studyB);
    // C1: A=130, B=128, nacional=120 → ambas acima
    const c1 = rows.find(r => r.comp === 'C1')!;
    expect(c1.reading).toMatch(/ambas/i);
  });

  it('produz 5 linhas quando apenas studyA disponível', () => {
    const rows = buildRedacaoRows(studyA, null);
    expect(rows).toHaveLength(5);
    rows.forEach(r => { expect(r.b).toBeNull(); });
  });

  it('national vem de B quando A é null', () => {
    const rows = buildRedacaoRows(null, studyB);
    expect(rows).toHaveLength(5);
    expect(rows[0].nacional).toBe(120);
  });

  it('não mutaciona os parâmetros de entrada', () => {
    const originalA = JSON.stringify(studyA);
    const originalB = JSON.stringify(studyB);
    buildRedacaoRows(studyA, studyB);
    expect(JSON.stringify(studyA)).toBe(originalA);
    expect(JSON.stringify(studyB)).toBe(originalB);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// buildSkillRows
// ─────────────────────────────────────────────────────────────────────────────

type SchoolSkillsReturn = {
  codigo_inep: string; ano: number; total_skills: number;
  worst_overall: { area: string; skill_num: number; performance: number;
    national_avg: number | null; diff: number | null; descricao: string;
    status: 'above' | 'below' | 'equal'; }[];
  by_area: Record<string, { skill_num: number; performance: number;
    national_avg: number | null; diff: number | null; descricao: string;
    status: 'above' | 'below' | 'equal'; }[]>;
};

const skillsData: SchoolSkillsReturn = {
  codigo_inep: '11111', ano: 2024, total_skills: 10,
  worst_overall: [
    { area: 'MT', skill_num: 1, performance: 200, national_avg: 400, diff: -200, descricao: 'Equações', status: 'below' },
    { area: 'CN', skill_num: 2, performance: 220, national_avg: 400, diff: -180, descricao: 'Termodinâmica', status: 'below' },
    { area: 'CH', skill_num: 3, performance: 600, national_avg: 400, diff: 200, descricao: 'História Medieval', status: 'above' },
  ],
  by_area: {
    LC: [
      { skill_num: 4, performance: 700, national_avg: 400, diff: 300, descricao: 'Interpretação Textual', status: 'above' },
      { skill_num: 5, performance: 500, national_avg: 400, diff: 100, descricao: 'Gramática', status: 'above' },
    ],
    MT: [
      { skill_num: 6, performance: 650, national_avg: 400, diff: 250, descricao: 'Geometria', status: 'above' },
      { skill_num: 1, performance: 200, national_avg: 400, diff: -200, descricao: 'Equações', status: 'below' },
    ],
  },
};

describe('buildSkillRows', () => {
  it('retorna [] quando input é null', () => {
    expect(buildSkillRows(null)).toEqual([]);
  });

  it('mapeia fracas (below) do worst_overall corretamente', () => {
    const rows = buildSkillRows(skillsData);
    const fracas = rows.filter(r => r.kind === 'fraca');
    expect(fracas).toHaveLength(2);
    expect(fracas[0].skill).toBe('Equações');
    expect(fracas[0].area).toBe('MT');
    expect(fracas[1].skill).toBe('Termodinâmica');
    expect(fracas[1].area).toBe('CN');
  });

  it('mapeia fortes (above) do by_area, no máximo 3, por maior performance', () => {
    const rows = buildSkillRows(skillsData);
    const fortes = rows.filter(r => r.kind === 'forte');
    // above em by_area: LC[0]=700, LC[1]=500, MT[0]=650 → top 3: 700, 650, 500
    expect(fortes).toHaveLength(3);
    expect(fortes[0].skill).toBe('Interpretação Textual');
    expect(fortes[1].skill).toBe('Geometria');
    expect(fortes[2].skill).toBe('Gramática');
  });

  it('dedup por descricao (Equações está em worst_overall e by_area)', () => {
    const rows = buildSkillRows(skillsData);
    const equacoes = rows.filter(r => r.skill === 'Equações');
    expect(equacoes).toHaveLength(1);
  });

  it('limita fracas a 3 mesmo se worst_overall tiver mais', () => {
    const big: SchoolSkillsReturn = {
      ...skillsData,
      worst_overall: Array.from({ length: 6 }, (_, i) => ({
        area: 'MT', skill_num: i, performance: 100 + i, national_avg: 400,
        diff: -300 + i, descricao: `Skill${i}`, status: 'below' as const,
      })),
    };
    const fracas = buildSkillRows(big).filter(r => r.kind === 'fraca');
    expect(fracas).toHaveLength(3);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// buildRecommendations
// ─────────────────────────────────────────────────────────────────────────────

function makeSchoolMeta(nome: string, nota: number): ReportSchoolMeta {
  return {
    codigo_inep: '11111', nome_escola: nome, uf: 'SP', cidade: 'São Paulo',
    tipo_escola: 'estadual', porte_label: 'Médio',
    nota_media: nota, ranking_brasil: 100, ranking_uf: 10, overall_health: 'good',
  };
}

function makeDiagnosis(): DiagnosisComparisonResult {
  return {
    school_1: { codigo_inep: '11111', info: { codigo_inep: '11111', nome_escola: 'A', porte: null, tipo_escola: null, localizacao: null, ano: 2024 }, overall_health: 'good' },
    school_2: { codigo_inep: '22222', info: { codigo_inep: '22222', nome_escola: 'B', porte: null, tipo_escola: null, localizacao: null, ano: 2024 }, overall_health: 'needs_attention' },
    area_comparison: [
      { area: 'MT', area_name: 'Matemática', school_1_score: 750, school_2_score: 600, difference: 150, school_1_status: 'good', school_2_status: 'needs_attention' },
      { area: 'LC', area_name: 'Linguagens', school_1_score: 700, school_2_score: 650, difference: 50, school_1_status: 'good', school_2_status: 'good' },
    ],
    winner_by_area: { MT: '11111', LC: '11111' },
  };
}

const reportDataFixture: ReportData = {
  generatedAt: new Date('2024-01-01'),
  baseYear: 2024,
  schoolA: makeSchoolMeta('Escola Alpha', 730),
  schoolB: makeSchoolMeta('Escola Beta', 625),
  diagnosis: makeDiagnosis(),
  history: [],
  projection: [
    {
      area: 'MT', area_name: 'Matemática', target_year: 2026,
      a: { current: 750, recommended: 760, potential_gain: 15, scenarios: null, official_next: null, official_change: null, risk_level: null, trend_dir: null, trend_annual: null },
      b: { current: 600, recommended: 640, potential_gain: 40, scenarios: null, official_next: null, official_change: null, risk_level: null, trend_dir: null, trend_annual: null },
      a_focus: [{ skill: 'H1', gap: 10 }],
      b_focus: [{ skill: 'H2', gap: 20 }],
    },
    {
      area: 'LC', area_name: 'Linguagens', target_year: 2026,
      a: { current: 700, recommended: 715, potential_gain: 10, scenarios: null, official_next: null, official_change: null, risk_level: null, trend_dir: null, trend_annual: null },
      b: { current: 650, recommended: 670, potential_gain: 20, scenarios: null, official_next: null, official_change: null, risk_level: null, trend_dir: null, trend_annual: null },
      a_focus: [],
      b_focus: [],
    },
  ],
  redacaoCompetencias: [
    { comp: 'C1', label: 'Domínio da norma', a: 130, b: 115, nacional: 120, reading: '' },
    { comp: 'C2', label: 'Tema e gênero', a: 125, b: 110, nacional: 120, reading: '' },
    { comp: 'C3', label: 'Argumentação', a: 122, b: 95, nacional: 120, reading: '' },
    { comp: 'C4', label: 'Coesão', a: 118, b: 100, nacional: 120, reading: '' },
    { comp: 'C5', label: 'Proposta de intervenção', a: 115, b: 88, nacional: 120, reading: '' },
  ],
};

describe('buildRecommendations', () => {
  it('inclui recomendação de área para o laggard (B) com impacto projetado', () => {
    const recs = buildRecommendations(reportDataFixture);
    // B perde em ambas as áreas; MT tem gap=150 (top-2 biggest) → Alta
    const lagAreaRec = recs.find(r =>
      r.scope === 'B' && r.action.toLowerCase().includes('matemática')
    );
    expect(lagAreaRec).toBeDefined();
    expect(lagAreaRec!.priority).toBe('Alta');
    // projection.b.potential_gain for MT=40 → impact menciona 40
    expect(lagAreaRec!.impact).toMatch(/40/);
  });

  it('inclui recomendação de redação para o gargalo do laggard', () => {
    const recs = buildRecommendations(reportDataFixture);
    // laggard B: menor nota em redação = C5 (88 vs nacional 120) → gargalo
    const redRec = recs.find(r =>
      r.scope === 'B' && r.action.toLowerCase().includes('redaç')
    );
    expect(redRec).toBeDefined();
    expect(redRec!.priority).toBe('Alta');
    expect(redRec!.impact).toMatch(/gargalo/i);
  });

  it('inclui recomendação "Ambas" de monitoramento', () => {
    const recs = buildRecommendations(reportDataFixture);
    const ambas = recs.find(r => r.scope === 'Ambas');
    expect(ambas).toBeDefined();
    expect(ambas!.action).toMatch(/monitoramento/i);
  });

  it('inclui recomendação de benchmark', () => {
    const recs = buildRecommendations(reportDataFixture);
    const bench = recs.find(r => r.scope === 'Benchmark');
    expect(bench).toBeDefined();
    // deve mencionar nomes reais vindos de d, não hardcoded
    expect(bench!.action).toContain('Escola Beta');   // laggard
    expect(bench!.action).toContain('Escola Alpha');  // leader
  });

  it('nomes vêm de d.schoolA/B (não hardcoded)', () => {
    const d2 = {
      ...reportDataFixture,
      schoolA: makeSchoolMeta('Colégio X', 730),
      schoolB: makeSchoolMeta('Instituto Y', 625),
    };
    const recs = buildRecommendations(d2);
    const allText = recs.map(r => r.action + r.impact).join(' ');
    expect(allText).toContain('Instituto Y');
    expect(allText).toContain('Colégio X');
  });

  it('inclui recomendação de preservação de área forte do líder', () => {
    const recs = buildRecommendations(reportDataFixture);
    const preserve = recs.find(r => r.scope === 'A' && r.priority === 'Baixa');
    expect(preserve).toBeDefined();
    expect(preserve!.action).toMatch(/preserv/i);
  });

  it('não produz linhas duplicadas', () => {
    const recs = buildRecommendations(reportDataFixture);
    const actions = recs.map(r => r.action);
    const unique = new Set(actions);
    expect(unique.size).toBe(actions.length);
  });

  it('não lança quando area_comparison é vazio e retorna array', () => {
    const emptyAreasDiag = makeDiagnosis();
    (emptyAreasDiag as any).area_comparison = [];
    const d: ReportData = {
      generatedAt: new Date('2024-01-01'),
      baseYear: 2024,
      schoolA: makeSchoolMeta('Escola Alpha', 730),
      schoolB: makeSchoolMeta('Escola Beta', 625),
      diagnosis: emptyAreasDiag,
      history: [],
    };
    expect(() => buildRecommendations(d)).not.toThrow();
    const recs = buildRecommendations(d);
    expect(Array.isArray(recs)).toBe(true);
    // The always-on monitoring rec must still be present
    const ambas = recs.find(r => r.scope === 'Ambas');
    expect(ambas).toBeDefined();
  });
});
