'use client';
import { useEffect } from 'react';
import './ReportDocument.css';
import type { ReportData } from './types';
import { areasWon, biggestGapArea, rankingGap, statusClass, statusLabel, fmt, winnerOfArea } from './reportMetrics';
import { executiveSummary, areaParagraph } from './reportNarrative';
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

      {/* Seções 4–8 e 11 entram nas Fases 2–3 (render condicional) */}

      <div className="foot"><span>X-TRI Escolas · rankingenem.com</span><span>Base ENEM {d.baseYear}</span></div>
    </div>
  );
}
