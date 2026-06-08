import { describe, it, expect } from 'vitest';
import { areasWon, biggestGapArea, rankingGap, trendOverYears, statusClass, fmt, buildProjectionRows } from './reportMetrics';
import type { DiagnosisComparisonResult, TRIAreaProjection } from '@/lib/api';

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
