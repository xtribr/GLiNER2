'use client';
import { useEffect } from 'react';
import './TriReportDocument.css';
import { formatTriScore } from '@/lib/utils';
import type { TriReportData } from './triTypes';
import type { TRIAreaAnalysis, TRIAreaProjection } from '@/lib/api';
import { masteryColor, pct, trendLabel, trendColor, insightClass, signed } from './triReportMetrics';
import { TriEvolutionChart } from './TriReportChart';

interface Props { data: TriReportData; onReady?: () => void; }

const MAX_STRETCH = 12; // limita a lista por área para manter o PDF utilizável

function Gauge({ value }: { value: number }) {
  const p = pct(value);
  const color = masteryColor(value);
  const r = 26, c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 64 64" style={{ width: 54, height: 54 }}>
      <g transform="rotate(-90 32 32)">
        <circle cx="32" cy="32" r={r} fill="none" stroke="#e5e7eb" strokeWidth="7" />
        <circle cx="32" cy="32" r={r} fill="none" stroke={color} strokeWidth="7" strokeLinecap="round" strokeDasharray={`${(p / 100) * c} ${c}`} />
      </g>
      <text x="32" y="37" textAnchor="middle" fontSize="15" fontWeight="800" fill={color}>{p}%</text>
    </svg>
  );
}

function Scenario({ p }: { p: TRIAreaProjection }) {
  const off = p.official_prediction;
  const isRange = off.display_mode === 'range';
  const t = p.historical_analysis.trend;
  const items = p.stretch_content.items.slice(0, MAX_STRETCH);
  const hidden = p.stretch_content.items.length - items.length;
  return (
    <div className="scenario">
      <div className="kpis4">
        <div className="kpi"><div className="kv">{formatTriScore(p.current_score)}</div><div className="kl">Score Atual ({p.current_year})</div></div>
        <div className={`kpi ${isRange ? 'amber' : 'blue'}`}>
          <div className="kv">{formatTriScore(off.display_score)}</div>
          <div className="kl">Predição oficial {off.target_year}</div>
          <div className="ksub">{isRange ? `${formatTriScore(off.confidence_interval.low)} – ${formatTriScore(off.confidence_interval.high)}` : `${signed(off.display_expected_change)} pts`}</div>
        </div>
        <div className={`kpi ${p.projection.potential_gain >= 0 ? 'green' : 'red'}`}><div className="kv">{signed(p.projection.potential_gain)}</div><div className="kl">Cenário Pedagógico</div></div>
        <div className="kpi purple"><div className="kv">{p.stretch_content.total_items}</div><div className="kl">Conteúdos a Dominar</div></div>
      </div>

      {isRange && (
        <div className="estab">
          <div className="et">{off.badge_text || 'Projeção conservadora'}</div>
          <div className="er">{off.risk_reason || 'A predição oficial foi apresentada em faixa por exceder a volatilidade histórica observada.'}</div>
        </div>
      )}

      <div className="chartbox"><div className="cap">Evolução Histórica e Projeção</div><TriEvolutionChart p={p} /></div>

      <div className="grid2">
        <div className="box">
          <h4>Análise de Tendência</h4>
          <div className="kv2"><span>Direção</span><b style={{ color: trendColor(t.direction) }}>{trendLabel(t.direction)}</b></div>
          <div className="kv2"><span>Variação Anual</span><b style={{ color: t.annual_change >= 0 ? '#16a34a' : '#dc2626' }}>{signed(t.annual_change)} pts/ano</b></div>
          <div className="kv2"><span>Confiabilidade (R²)</span><b>{Math.round((t.r_squared ?? 0) * 100)}%</b></div>
          <div className="kv2"><span>Melhoria Média</span><b style={{ color: '#2563eb' }}>{signed(p.historical_analysis.statistics.avg_improvement)} pts</b></div>
        </div>
        <div className="box">
          <h4>Cenários Pedagógicos</h4>
          <div className="kv2"><span>Conservador</span><b style={{ color: '#475569' }}>{formatTriScore(p.projection.scenarios.conservative)}</b></div>
          <div className="kv2"><span style={{ color: '#2563eb' }}>Realista (Cenário)</span><b style={{ color: '#2563eb' }}>{formatTriScore(p.projection.scenarios.realistic)}</b></div>
          <div className="kv2"><span>Otimista</span><b style={{ color: '#16a34a' }}>{formatTriScore(p.projection.scenarios.optimistic)}</b></div>
          <div className="kv2 sep"><span>Intervalo de Confiança</span><b style={{ color: '#64748b' }}>{formatTriScore(p.projection.confidence_interval.low)} – {formatTriScore(p.projection.confidence_interval.high)}</b></div>
        </div>
      </div>

      {p.insights.length > 0 && (
        <div className="insights">
          {p.insights.map((ins, i) => (
            <div key={i} className={`insight ${insightClass(ins.type)}`}>
              <div className="ih">{ins.title}</div><div className="im">{ins.message}</div>
            </div>
          ))}
        </div>
      )}

      {items.length > 0 && (
        <div className="dominar">
          <div className="dh">Conteúdo a Dominar para Alcançar a Projeção</div>
          <div className="dsub">TRI entre {formatTriScore(p.stretch_content.tri_range.min)} e {formatTriScore(p.stretch_content.tri_range.max)} pontos{hidden > 0 ? ` · mostrando ${items.length} de ${p.stretch_content.items.length}` : ''}</div>
          <table>
            <thead><tr><th>Habilidade</th><th style={{ width: '14%' }}>TRI</th><th style={{ width: '12%' }}>Gap</th></tr></thead>
            <tbody>
              {items.map((it, i) => (
                <tr key={i}>
                  <td><span className="code">{it.skill}</span>{it.description ? ` — ${it.description}` : ''}</td>
                  <td>{formatTriScore(it.tri_score)}</td>
                  <td className="gap">+{formatTriScore(it.gap)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function AreaSection({ an, p }: { an: TRIAreaAnalysis; p: TRIAreaProjection | null }) {
  const isRange = an.display_mode === 'range';
  const changeColor = isRange ? '#b45309' : an.expected_change >= 0 ? '#16a34a' : '#dc2626';
  const masteryPct = pct(an.tri_mastery_level);
  const domLabel = an.area === 'RE' ? 'Domínio Competências' : 'Domínio TRI';
  return (
    <div className="areasec" style={{ borderTopColor: an.color }}>
      <div className="areahead"><span className="dot" style={{ background: an.color }} /><span className="atitle">{an.area_name}</span><span className="acode">({an.area})</span></div>

      <div className="sum3">
        <div><div className="lbl">Atual</div><div className="big">{formatTriScore(an.current_score)}</div></div>
        <div><div className="lbl">Previsto</div><div className="big" style={{ color: '#2563eb' }}>{formatTriScore(an.predicted_score)}</div></div>
        <div><div className="lbl">{isRange ? 'Faixa' : 'Mudança'}</div>
          <div className="big" style={{ color: changeColor }}>
            {isRange ? `${formatTriScore(an.confidence_interval.low)}–${formatTriScore(an.confidence_interval.high)}` : signed(an.expected_change)}
          </div>
        </div>
      </div>

      {isRange && (
        <div className="estab">
          <div className="et">{an.badge_text || 'Estabilidade histórica'}</div>
          {an.risk_reason && <div className="er">{an.risk_reason}</div>}
        </div>
      )}

      <div className="masteryrow">
        <div className="masterytop"><span>{domLabel}</span><span>{masteryPct}%</span></div>
        <div className="bar"><div className="barfill" style={{ width: `${masteryPct}%`, background: an.color }} /></div>
      </div>
      <div className="gp">
        <span>Gap Mediana: <b style={{ color: an.tri_gap_to_median >= 0 ? '#16a34a' : '#dc2626' }}>{signed(an.tri_gap_to_median)}</b></span>
        <span>Potencial: <b style={{ color: '#2563eb' }}>{pct(an.tri_potential)}%</b></span>
      </div>

      {p ? <Scenario p={p} /> : <p className="muted">Cenário detalhado indisponível para esta área.</p>}
    </div>
  );
}

export default function TriReportDocument({ data: d, onReady }: Props) {
  useEffect(() => { onReady?.(); }, [onReady]);
  const date = d.generatedAt.toLocaleDateString('pt-BR');
  return (
    <div className="xtri-tri">
      <div className="header">
        <img src="/logo-xtri.png" alt="X-TRI" />
        <div className="htxt">
          <h1>Relatório TRI</h1>
          <div className="meta">{d.schoolName}{d.baseYear ? ` · Base ENEM ${d.baseYear}` : ''} · Gerado em {date} · X-TRI Escolas</div>
        </div>
        <div className="gaugewrap"><Gauge value={d.overallMastery} /><span className="gaugelbl">Domínio Geral</span></div>
      </div>

      <p className="sub">Predição baseada em Teoria de Resposta ao Item (TRI) — predição oficial calibrada + cenários pedagógicos por área.</p>

      <div className="grid2">
        <div className="lcard" style={{ borderLeftColor: '#3b82f6' }}><div className="ct" style={{ color: '#2563eb' }}>Interpretação</div><p>{d.masteryInterpretation}</p></div>
        <div className="lcard" style={{ borderLeftColor: '#22c55e' }}><div className="ct" style={{ color: '#16a34a' }}>Recomendação</div><p>{d.recommendation}</p></div>
      </div>

      <div className="secbar">Análise por Área</div>
      {d.areas.map(({ analysis: an, projection: p }) => (<AreaSection key={an.area} an={an} p={p} />))}

      <div className="foot"><span>X-TRI Escolas · rankingenem.com</span><span>Análise TRI · {d.schoolName}</span></div>
    </div>
  );
}
