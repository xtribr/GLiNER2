import type { TRIAreaProjection } from '@/lib/api';
import { formatTriScore } from '@/lib/utils';

/** "Evolução Histórica e Projeção" — SVG puro (texto vetorial p/ impressão).
 *  Linha histórica na cor da área, com TODOS os anos rotulados e o valor de cada
 *  ano impresso acima do ponto; ponto de projeção (azul) com barra de intervalo de
 *  confiança e linha tracejada do score atual. Preenche a largura (height: auto). */
export function TriEvolutionChart({ p }: { p: TRIAreaProjection }) {
  const scores = p.historical_analysis.scores ?? [];
  if (scores.length === 0) return null;

  const projYear = p.projection.target_year;
  const projScore = p.projection.recommended;
  const ciLo = p.projection.confidence_interval.low;
  const ciHi = p.projection.confidence_interval.high;
  const current = p.current_score;

  const pts = [
    ...scores.map((s) => ({ ano: s.ano, score: s.score, proj: false })),
    { ano: projYear, score: projScore, proj: true },
  ];

  const W = 600, H = 230, padL = 48, padR = 24, padT = 30, padB = 28;
  const years = pts.map((d) => d.ano);
  const xMin = Math.min(...years), xMax = Math.max(...years);
  const allY = [...scores.map((s) => s.score), projScore, ciLo, ciHi, current];
  const yLo = Math.floor((Math.min(...allY) - 14) / 10) * 10;
  const yHi = Math.ceil((Math.max(...allY) + 14) / 10) * 10;

  const X = (ano: number) => padL + ((ano - xMin) / Math.max(1, xMax - xMin)) * (W - padL - padR);
  const Y = (v: number) => padT + (1 - (v - yLo) / Math.max(1, yHi - yLo)) * (H - padT - padB);

  const linePts = pts.map((d) => `${X(d.ano).toFixed(1)},${Y(d.score).toFixed(1)}`).join(' ');
  const yMid = Math.round((yLo + yHi) / 2);
  const projX = X(projYear);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {/* gridlines + rótulos do eixo Y */}
      {[yHi, yMid, yLo].map((v) => (
        <g key={v}>
          <line x1={padL} y1={Y(v)} x2={W - padR} y2={Y(v)} stroke="#eef2f6" strokeWidth="1" />
          <text x={padL - 6} y={Y(v) + 4} fontSize="11" fill="#94a3b8" textAnchor="end">{v.toLocaleString('pt-BR')}</text>
        </g>
      ))}
      {/* score atual (tracejado) */}
      <line x1={padL} y1={Y(current)} x2={W - padR} y2={Y(current)} stroke="#9ca3af" strokeWidth="1" strokeDasharray="5 5" />
      <text x={padL + 2} y={Y(current) - 4} fontSize="10.5" fill="#9ca3af" textAnchor="start">Atual</text>
      {/* intervalo de confiança da projeção (barra vertical) */}
      <line x1={projX} y1={Y(ciLo)} x2={projX} y2={Y(ciHi)} stroke={p.color} strokeWidth="9" strokeOpacity="0.22" strokeLinecap="round" />
      {/* linha histórica */}
      <polyline points={linePts} fill="none" stroke={p.color} strokeWidth="2.6" strokeLinejoin="round" />
      {/* pontos + valor do score (de TODOS os anos) acima do ponto */}
      {pts.map((d) => (
        <g key={d.ano}>
          <circle cx={X(d.ano)} cy={Y(d.score)} r={d.proj ? 5.5 : 3.6} fill={d.proj ? '#3b82f6' : p.color} stroke="#fff" strokeWidth="1.6" />
          <text x={X(d.ano)} y={Y(d.score) - 9} fontSize="10" fontWeight="700" fill={d.proj ? '#2563eb' : '#334155'} textAnchor="middle">{formatTriScore(d.score)}</text>
        </g>
      ))}
      {/* TODOS os anos no eixo X */}
      {pts.map((d) => (
        <text key={`x-${d.ano}`} x={X(d.ano)} y={H - 8} fontSize="11" fill={d.proj ? '#2563eb' : '#94a3b8'} fontWeight={d.proj ? 700 : 400} textAnchor="middle">{d.ano}</text>
      ))}
    </svg>
  );
}
