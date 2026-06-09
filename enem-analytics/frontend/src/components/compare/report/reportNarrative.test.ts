import { describe, it, expect } from 'vitest';
import { executiveSummary, areaParagraph, projectionParagraph } from './reportNarrative';
import type { ProjectionRow } from './types';
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
    expect(s).toContain('96,0');       // média diag: A (695,5) × B (791,5) → vantagem 96,0
    expect(s).toMatch(/Redação/);      // maior gap mencionado
  });

  // Regressão: líder/vantagem vêm do diagnóstico, não de nota_media (que pode ser null)
  it('com nota_media null ainda nomeia o líder e a vantagem reais (sem vazar a nota inteira)', () => {
    const baseNull = {
      schoolA: { nome_escola: 'ALPHA', nota_media: null, ranking_brasil: null },
      schoolB: { nome_escola: 'BETA', nota_media: null, ranking_brasil: 5 },
      diagnosis: { area_comparison: [
        { area:'MT', area_name:'Matemática', school_1_score:600, school_2_score:700, difference:-100, school_1_status:'good', school_2_status:'excellent' },
        { area:'CH', area_name:'Ciências Humanas', school_1_score:620, school_2_score:680, difference:-60, school_1_status:'good', school_2_status:'good' },
      ] },
    } as unknown as ReportData;
    const s = executiveSummary(baseNull);
    // média diag: A=610, B=690 → BETA lidera por 80,0 pts (NÃO ~690)
    expect(s).toContain('BETA');
    expect(s).toContain('80,0');
    expect(s).not.toMatch(/lidera por 6\d\d/);  // não apresenta a nota inteira como "vantagem"
  });
});
describe('areaParagraph', () => {
  it('descreve a área e o vencedor com a diferença', () => {
    const p = areaParagraph(base.diagnosis.area_comparison[0], base);
    expect(p).toMatch(/212/);
    expect(p).toContain('CHRISTUS');   // vencedor da Redação (B)
  });
});

// --- projectionParagraph ---

function makeRow(area: string, areaName: string, gainA: number, gainB: number, year = 2026): ProjectionRow {
  return {
    area,
    area_name: areaName,
    target_year: year,
    a: { current: 600, recommended: 600 + gainA, potential_gain: gainA, scenarios: { conservative: 605, realistic: 610, optimistic: 615 }, official_next: 610, official_change: 10, risk_level: 'normal', trend_dir: 'ascending', trend_annual: 5 },
    b: { current: 580, recommended: 580 + gainB, potential_gain: gainB, scenarios: { conservative: 585, realistic: 590, optimistic: 595 }, official_next: 590, official_change: 10, risk_level: 'normal', trend_dir: 'stable', trend_annual: 0 },
    a_focus: [{ skill: 'H1', gap: 30 }],
    b_focus: [{ skill: 'H2', gap: 20 }],
  };
}

describe('projectionParagraph', () => {
  it('retorna string vazia para lista vazia', () => {
    expect(projectionParagraph([])).toBe('');
  });

  it('menciona o ano alvo correto', () => {
    const p = projectionParagraph([makeRow('MT', 'Matemática', 20, 15)]);
    expect(p).toContain('2026');
  });

  it('identifica a área com maior ganho potencial', () => {
    const rows = [
      makeRow('MT', 'Matemática', 20, 15),
      makeRow('CN', 'Ciências da Natureza', 35, 10),
    ];
    const p = projectionParagraph(rows);
    expect(p).toContain('Ciências da Natureza'); // CN tem gain 35 > MT 20
  });

  it('identifica a escola com maior upside agregado', () => {
    const rows = [
      makeRow('MT', 'Matemática', 20, 30),   // B domina
      makeRow('CH', 'Ciências Humanas', 10, 25),
    ];
    const p = projectionParagraph(rows);
    expect(p).toContain('Escola B');           // totalB=55 > totalA=30
  });

  it('identifica Escola A quando ela tem mais upside', () => {
    const rows = [
      makeRow('MT', 'Matemática', 40, 10),
      makeRow('LC', 'Linguagens', 20, 5),
    ];
    const p = projectionParagraph(rows);
    expect(p).toContain('Escola A');           // totalA=60 > totalB=15
  });
});
