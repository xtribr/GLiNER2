import { describe, it, expect } from 'vitest';
import { executiveSummary, areaParagraph } from './reportNarrative';
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
    expect(s).toContain('99,2');       // |663.2-762.4|
    expect(s).toMatch(/Redação/);      // maior gap mencionado
  });
});
describe('areaParagraph', () => {
  it('descreve a área e o vencedor com a diferença', () => {
    const p = areaParagraph(base.diagnosis.area_comparison[0], base);
    expect(p).toMatch(/212/);
    expect(p).toContain('CHRISTUS');   // vencedor da Redação (B)
  });
});
