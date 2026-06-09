import { describe, it, expect } from 'vitest';
import { masteryColor, pct, trendLabel, trendColor, insightClass, signed } from './triReportMetrics';

describe('masteryColor', () => {
  it('mapeia o nível 0–1 para as 5 faixas de cor', () => {
    expect(masteryColor(0.85)).toBe('#22c55e'); // ≥0.8
    expect(masteryColor(0.65)).toBe('#84cc16'); // ≥0.6
    expect(masteryColor(0.45)).toBe('#eab308'); // ≥0.4
    expect(masteryColor(0.25)).toBe('#f97316'); // ≥0.2
    expect(masteryColor(0.10)).toBe('#ef4444'); // <0.2
  });
});

describe('pct', () => {
  it('converte nível para porcentagem inteira', () => {
    expect(pct(0.836)).toBe(84);
    expect(pct(0.91)).toBe(91);
    expect(pct(null)).toBe(0);
  });
});

describe('trendLabel / trendColor', () => {
  it('rotula e colore a direção da tendência', () => {
    expect(trendLabel('ascending')).toMatch(/Crescente/);
    expect(trendLabel('descending')).toMatch(/Decrescente/);
    expect(trendLabel('stable')).toMatch(/Estável/);
    expect(trendLabel('insufficient_data')).toMatch(/insuf/);
    expect(trendColor('ascending')).toBe('#16a34a');
    expect(trendColor('descending')).toBe('#dc2626');
    expect(trendColor('stable')).toBe('#6b7280');
  });
});

describe('insightClass', () => {
  it('mapeia o tipo de insight para a classe CSS', () => {
    expect(insightClass('positive')).toBe('ins-pos');
    expect(insightClass('warning')).toBe('ins-warn');
    expect(insightClass('info')).toBe('ins-info');
    expect(insightClass('neutral')).toBe('ins-neutral');
  });
});

describe('signed', () => {
  it('formata delta com sinal e 1 casa pt-BR', () => {
    expect(signed(16.7)).toBe('+16,7');
    expect(signed(-5)).toBe('−5,0');
    expect(signed(0)).toBe('+0,0');
    expect(signed(null)).toBe('—');
  });
});
