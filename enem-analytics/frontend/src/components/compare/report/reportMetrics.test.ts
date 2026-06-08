import { describe, it, expect } from 'vitest';
import { areasWon, biggestGapArea, rankingGap, trendOverYears, statusClass, fmt } from './reportMetrics';
import type { DiagnosisComparisonResult } from '@/lib/api';

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
