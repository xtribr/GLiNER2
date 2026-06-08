'use client';
import { useEffect } from 'react';
import './ReportDocument.css';
import type { ReportData } from './types';
import { areasWon, biggestGapArea, rankingGap, statusClass, statusLabel, fmt, winnerOfArea } from './reportMetrics';
import { executiveSummary, areaParagraph, projectionParagraph } from './reportNarrative';
import { AreaBars, EvolutionLine, AreaRadar } from './ReportCharts';

interface Props { data: ReportData; onReady?: () => void; }

export default function ReportDocument({ data: d, onReady }: Props) {
  useEffect(() => { onReady?.(); }, [onReady]);
  const won = areasWon(d.diagnosis);
  const big = biggestGapArea(d.diagnosis);
  const date = d.generatedAt.toLocaleDateString('pt-BR');

  return (
    <div className="xtri-report">
      <div className="header">
        <img src="/logo-xtri.png" alt="X-TRI" />
        <div>
          <h1>Relatório Comparativo de Escolas</h1>
          <div className="meta">Análise gerada em {date} | Base ENEM {d.baseYear} | X-TRI Escolas</div>
        </div>
      </div>

      <div className="sec">Resumo Executivo</div>
      <p className="an">{executiveSummary(d)}</p>

      <div className="sec">Escolas Comparadas</div>
      <table>
        <thead><tr><th style={{ width: '24%' }}>Campo</th><th>Escola A — {d.schoolA.nome_escola}</th><th>Escola B — {d.schoolB.nome_escola}</th></tr></thead>
        <tbody>
          <tr><td>INEP · UF · Cidade</td><td>{d.schoolA.codigo_inep} · {d.schoolA.uf} · {d.schoolA.cidade}</td><td>{d.schoolB.codigo_inep} · {d.schoolB.uf} · {d.schoolB.cidade}</td></tr>
          <tr><td>Tipo · Porte</td><td>{d.schoolA.tipo_escola} · {d.schoolA.porte_label}</td><td>{d.schoolB.tipo_escola} · {d.schoolB.porte_label}</td></tr>
          <tr><td>Média geral (TRI)</td><td className="a">{fmt(d.schoolA.nota_media)}</td><td className="b">{fmt(d.schoolB.nota_media)}</td></tr>
          <tr><td>Ranking Brasil · UF</td><td>#{d.schoolA.ranking_brasil} · #{d.schoolA.ranking_uf}</td><td>#{d.schoolB.ranking_brasil} · #{d.schoolB.ranking_uf}</td></tr>
        </tbody>
      </table>
      <div className="kpis">
        <div className="kpi"><div className="v" style={{ color: '#16a34a' }}>{fmt(Math.abs((d.schoolA.nota_media??0)-(d.schoolB.nota_media??0)))}</div><div className="l">Vantagem média</div></div>
        <div className="kpi"><div className="v">{won.a} × {won.b}</div><div className="l">Áreas A × B</div></div>
        <div className="kpi"><div className="v" style={{ color: '#ff6b5c' }}>{fmt(big.gap)}</div><div className="l">Maior gap ({big.area_name})</div></div>
        <div className="kpi"><div className="v">{rankingGap(d.schoolA.ranking_brasil, d.schoolB.ranking_brasil) ?? '—'}</div><div className="l">Gap de ranking</div></div>
      </div>

      <div className="sec">Comparação Detalhada — as 5 notas, uma a uma</div>
      {d.diagnosis.area_comparison.map((ar) => {
        const w = winnerOfArea(ar);
        const cls = w === 'A' ? 'win-a' : w === 'B' ? 'gap' : '';
        return (
          <div className={`areablock ${cls}`} key={ar.area}>
            <div className="areahead">
              <span className="t">{ar.area_name}
                <span className={`statusbadge ${statusClass(ar.school_1_status)}`}>A {statusLabel(ar.school_1_status)}</span>
                <span className={`statusbadge ${statusClass(ar.school_2_status)}`}>B {statusLabel(ar.school_2_status)}</span>
              </span>
              <span className="n"><span className="a">A {fmt(ar.school_1_score)}</span> · <span className="b">B {fmt(ar.school_2_score)}</span></span>
            </div>
            <p className="an">{areaParagraph(ar, d)}</p>
          </div>
        );
      })}

      <div className="sec">Visão Gráfica</div>
      <div className="grid2">
        <div><div className="cap">Notas por área — A (azul) × B (verde)</div><AreaBars diagnosis={d.diagnosis} /></div>
        <div><div className="cap">Radar das 5 áreas</div><AreaRadar diagnosis={d.diagnosis} /></div>
      </div>
      <div className="grid2">
        <div><div className="cap">Evolução da média</div><EvolutionLine history={d.history} /></div>
      </div>

      <div className="sec">Histórico Ano a Ano</div>
      <table>
        <thead><tr><th>Ano</th><th>Média A</th><th>Rank A</th><th>Média B</th><th>Rank B</th><th>Distância (B−A)</th></tr></thead>
        <tbody>
          {d.history.map((y) => (
            <tr key={y.ano}><td>{y.ano}</td><td className="a">{fmt(y.a_media)}</td><td>#{y.a_rank}</td><td className="b">{fmt(y.b_media)}</td><td>#{y.b_rank}</td>
              <td>{y.a_media != null && y.b_media != null ? fmt(y.b_media - y.a_media) : '—'}</td></tr>
          ))}
        </tbody>
      </table>

      {/* Seção de Projeção TRI — renderizada apenas quando disponível */}
      {d.projection && d.projection.length > 0 && (() => {
        const rows = d.projection;
        const targetYear = rows[0].target_year;
        const riskLabel = (r: string | null) =>
          r === 'conservative' ? 'Conservador' : r === 'outlier' ? 'Outlier' : r === 'normal' ? 'Normal' : '—';
        const trendLabel = (dir: string | null, annual: number | null) => {
          if (!dir || dir === 'insufficient_data') return '—';
          const sign = (annual ?? 0) > 0 ? '+' : '';
          return `${dir === 'ascending' ? '↑' : dir === 'descending' ? '↓' : '→'} ${sign}${fmt(annual)}/ano`;
        };
        return (
          <>
            <div className="sec">Projeção TRI — ENEM {targetYear} (cenários, predição oficial e conteúdos a focar)</div>
            <p className="an">{projectionParagraph(rows)}</p>

            <table>
              <thead>
                <tr>
                  <th style={{ width: '10%' }}>Área</th>
                  <th style={{ width: '5%' }}>Escola</th>
                  <th style={{ width: '8%' }}>Atual</th>
                  <th style={{ width: '8%' }}>Conserv.</th>
                  <th style={{ width: '8%' }}>Realista</th>
                  <th style={{ width: '8%' }}>Otimista</th>
                  <th style={{ width: '12%' }}>Recomend. (Δ)</th>
                  <th style={{ width: '14%' }}>Predição {targetYear} (Δ · Risco)</th>
                  <th style={{ width: '10%' }}>Tendência/ano</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  [
                    <tr key={`${row.area}-A`}>
                      <td rowSpan={2} style={{ fontWeight: 700, verticalAlign: 'middle' }}>{row.area_name}</td>
                      <td className="a">A</td>
                      <td className="a">{fmt(row.a.current)}</td>
                      <td>{fmt(row.a.scenarios?.conservative)}</td>
                      <td>{fmt(row.a.scenarios?.realistic)}</td>
                      <td>{fmt(row.a.scenarios?.optimistic)}</td>
                      <td className="a">{fmt(row.a.recommended)} ({row.a.potential_gain != null ? `+${fmt(row.a.potential_gain)}` : '—'})</td>
                      <td className="a">{fmt(row.a.official_next)} ({row.a.official_change != null ? (row.a.official_change >= 0 ? '+' : '') + fmt(row.a.official_change) : '—'}) · {riskLabel(row.a.risk_level)}</td>
                      <td>{trendLabel(row.a.trend_dir, row.a.trend_annual)}</td>
                    </tr>,
                    <tr key={`${row.area}-B`}>
                      <td className="b">B</td>
                      <td className="b">{fmt(row.b.current)}</td>
                      <td>{fmt(row.b.scenarios?.conservative)}</td>
                      <td>{fmt(row.b.scenarios?.realistic)}</td>
                      <td>{fmt(row.b.scenarios?.optimistic)}</td>
                      <td className="b">{fmt(row.b.recommended)} ({row.b.potential_gain != null ? `+${fmt(row.b.potential_gain)}` : '—'})</td>
                      <td className="b">{fmt(row.b.official_next)} ({row.b.official_change != null ? (row.b.official_change >= 0 ? '+' : '') + fmt(row.b.official_change) : '—'}) · {riskLabel(row.b.risk_level)}</td>
                      <td>{trendLabel(row.b.trend_dir, row.b.trend_annual)}</td>
                    </tr>,
                  ]
                ))}
              </tbody>
            </table>

            <div className="sec" style={{ marginTop: '6px' }}>Conteúdos a Focar por Área</div>
            {rows.map((row) => {
              const hasFocus = row.a_focus.length > 0 || row.b_focus.length > 0;
              if (!hasFocus) return null;
              return (
                <div className="areablock" key={`focus-${row.area}`}>
                  <div className="areahead">
                    <span className="t">{row.area_name}</span>
                    <span className="muted">stretch items (gap em pts TRI)</span>
                  </div>
                  <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
                    <div style={{ flex: 1 }}>
                      {row.a_focus.length > 0 && (
                        <>
                          <div className="cap" style={{ color: '#2563eb' }}>Escola A — conteúdos prioritários</div>
                          <table style={{ marginTop: '2px' }}>
                            <thead><tr><th>Habilidade</th><th>Gap (pts)</th></tr></thead>
                            <tbody>
                              {row.a_focus.map((item) => (
                                <tr key={item.skill}>
                                  <td className="a">{item.skill}</td>
                                  <td>+{fmt(item.gap)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </>
                      )}
                    </div>
                    <div style={{ flex: 1 }}>
                      {row.b_focus.length > 0 && (
                        <>
                          <div className="cap" style={{ color: '#16a34a' }}>Escola B — conteúdos prioritários</div>
                          <table style={{ marginTop: '2px' }}>
                            <thead><tr><th>Habilidade</th><th>Gap (pts)</th></tr></thead>
                            <tbody>
                              {row.b_focus.map((item) => (
                                <tr key={item.skill}>
                                  <td className="b">{item.skill}</td>
                                  <td>+{fmt(item.gap)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </>
        );
      })()}

      <div className="foot"><span>X-TRI Escolas · rankingenem.com</span><span>Base ENEM {d.baseYear}</span></div>
    </div>
  );
}
