'use client';
import type { DiagnosisComparisonResult } from '@/lib/api';
import type { ComparisonYearRow } from './types';

export function AreaBars({ diagnosis }: { diagnosis: DiagnosisComparisonResult }) {
  const areas = diagnosis.area_comparison;
  const max = Math.max(...areas.flatMap(a => [a.school_1_score, a.school_2_score]), 1);
  const h = (v: number) => (v / max) * 96;
  return (
    <svg viewBox="0 0 300 130" style={{ width: '100%', height: 120 }}>
      <line x1="28" y1="108" x2="298" y2="108" stroke="#e2e8f0" />
      {areas.map((a, i) => {
        const x = 38 + i * 54;
        return (
          <g key={a.area}>
            <rect x={x} y={108 - h(a.school_1_score)} width="15" height={h(a.school_1_score)} fill="#2563eb" />
            <rect x={x + 17} y={108 - h(a.school_2_score)} width="15" height={h(a.school_2_score)} fill="#16a34a" />
            <text x={x + 16} y="120" fontSize="8" fill="#64748b" textAnchor="middle">{a.area}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function EvolutionLine({ history }: { history: ComparisonYearRow[] }) {
  const ys = history.map(h => [h.a_media, h.b_media]).flat().filter((n): n is number => n != null);
  const min = Math.min(...ys, 0), max = Math.max(...ys, 1);
  const norm = (v: number) => 105 - ((v - min) / (max - min || 1)) * 90;
  const xs = (i: number) => 20 + (i * 276) / Math.max(history.length - 1, 1);
  const line = (key: 'a_media'|'b_media') => history.map((h, i) => h[key] != null ? `${xs(i)},${norm(h[key]!)}` : '').filter(Boolean).join(' ');
  return (
    <svg viewBox="0 0 300 120" style={{ width: '100%', height: 120 }}>
      <line x1="20" y1="105" x2="298" y2="105" stroke="#e2e8f0" />
      <polyline points={line('a_media')} fill="none" stroke="#2563eb" strokeWidth="2.2" />
      <polyline points={line('b_media')} fill="none" stroke="#16a34a" strokeWidth="2.2" />
      {history[0] && <text x="20" y="116" fontSize="7.5" fill="#94a3b8">{history[0].ano}</text>}
      {history.at(-1) && <text x="270" y="116" fontSize="7.5" fill="#94a3b8">{history.at(-1)!.ano}</text>}
    </svg>
  );
}

export function AreaRadar({ diagnosis }: { diagnosis: DiagnosisComparisonResult }) {
  const areas = diagnosis.area_comparison.slice(0, 5);
  const max = Math.max(...areas.flatMap(a => [a.school_1_score, a.school_2_score]), 1);
  const pt = (i: number, v: number, r = 95) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / areas.length;
    const rad = (v / max) * r;
    return `${150 + rad * Math.cos(ang)},${110 + rad * Math.sin(ang)}`;
  };
  const poly = (key: 'school_1_score'|'school_2_score') => areas.map((a, i) => pt(i, a[key])).join(' ');
  return (
    <svg viewBox="0 0 300 220" style={{ width: '100%', height: 200 }}>
      <polygon points={areas.map((_, i) => pt(i, max)).join(' ')} fill="none" stroke="#e2e8f0" />
      <polygon points={poly('school_1_score')} fill="rgba(37,99,235,.15)" stroke="#2563eb" strokeWidth="2" />
      <polygon points={poly('school_2_score')} fill="rgba(22,163,74,.13)" stroke="#16a34a" strokeWidth="2" />
      {areas.map((a, i) => { const [x, y] = pt(i, max * 1.12).split(','); return <text key={a.area} x={x} y={y} fontSize="8" fill="#64748b" textAnchor="middle">{a.area}</text>; })}
    </svg>
  );
}
