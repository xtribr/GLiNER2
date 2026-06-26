'use client';

import Image from 'next/image';
import { useQuery } from '@tanstack/react-query';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api, type RedacaoAnalysis } from '@/lib/api';
import { formatNumber } from '@/lib/utils';
import RedacaoDivergenceMap from '@/components/redacao/RedacaoDivergenceMap';

// Identidade XTRI
const XTRI_BLUE = '#139ED3';
const XTRI_RED = '#FF4B2E';

const LEVELS = [0, 40, 80, 120, 160, 200];
// Níveis de nota: vermelho (baixo) → verde (alto)
const GRADE_COLORS: Record<number, string> = {
  0: '#FF4B2E', 40: '#FF8A70', 80: '#FFCB6B', 120: '#7FE0FF', 160: '#3ABFF8', 200: XTRI_BLUE,
};
// Divergência: azul XTRI (concordam) → vermelho XTRI (divergem muito)
const DIFF_COLORS: Record<number, string> = {
  0: XTRI_BLUE, 40: '#7FE0FF', 80: '#FFCB6B', 120: '#FF8A70', 160: '#FF6B5C', 200: XTRI_RED,
};

function StatCard({ value, label, hint, accent }: { value: string; label: string; hint?: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-2xl font-bold" style={{ color: accent ?? '#0f172a' }}>{value}</p>
      <p className="mt-1 text-sm font-medium text-slate-600">{label}</p>
      {hint ? <p className="mt-1 text-xs text-slate-400">{hint}</p> : null}
    </div>
  );
}

function ChartCard({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <section className="mt-10">
      <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
      {desc ? <p className="mb-4 mt-1 text-sm text-slate-500">{desc}</p> : <div className="mb-4" />}
      <div className="rounded-xl border border-slate-200 bg-white p-4">{children}</div>
    </section>
  );
}

function stackedData(d: RedacaoAnalysis, key: 'dist_niveis_pct' | 'dist_divergencia_pct') {
  return d.divergencia_avaliadores.por_competencia.map((c) => ({
    competencia: c.competencia,
    nome: c.nome,
    ...Object.fromEntries(LEVELS.map((l) => [String(l), c[key][String(l)] ?? 0])),
  }));
}

export default function RedacaoPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['redacao-analysis'],
    queryFn: api.getRedacaoAnalysis,
  });

  return (
    <main className="mx-auto max-w-5xl px-4 py-10">
      <header className="mb-8 flex items-start gap-4">
        <Image src="/logo-x.png" alt="XTRI" width={56} height={56} className="h-12 w-12 shrink-0 object-contain" priority />
        <div>
          <span
            className="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold"
            style={{ backgroundColor: 'rgba(19,158,211,0.10)', color: XTRI_BLUE }}
          >
            ENEM 2025 · Microdados oficiais INEP
          </span>
          <h1 className="mt-3 text-3xl font-bold tracking-tight text-slate-900">
            A dupla correção da redação, em números
          </h1>
          <p className="mt-2 max-w-2xl text-slate-600">
            Toda redação do ENEM passa por dois avaliadores independentes; se eles divergem demais,
            entra um terceiro. Veja como foi 2025 — em dados{' '}
            <strong style={{ color: XTRI_RED }}>nacionais, agregados e 100% anônimos</strong>.
          </p>
        </div>
      </header>

      {isLoading && <p className="text-slate-500">Carregando análise…</p>}
      {isError && <p style={{ color: XTRI_RED }}>Não foi possível carregar a análise da redação.</p>}

      {data && (
        <>
          <section className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            <StatCard value={formatNumber(data.nota.media)} label="Nota média" hint="0–1000" accent={XTRI_BLUE} />
            <StatCard value={formatNumber(data.redacoes_corrigidas)} label="Redações corrigidas" hint={`${formatNumber(data.total_participantes)} inscritos`} />
            <StatCard value={`${data.status.legenda.find((s) => s.codigo === '(ausente)')?.pct ?? 0}%`} label="Ausentes" hint="não fizeram" />
            <StatCard value={`${data.divergencia_avaliadores.pct_terceiro_avaliador}%`} label="3º avaliador" accent={XTRI_RED} hint="Av1 e Av2 divergiram" />
            <StatCard value={formatNumber(data.status.total_zeradas_anuladas)} label="Zeradas / anuladas" accent={XTRI_RED} hint="por algum problema" />
            <StatCard value={formatNumber(data.nota.nota_1000)} label="Notas 1000" accent={XTRI_BLUE} hint={`${formatNumber(data.nota.nota_zero)} zeraram`} />
          </section>

          <ChartCard
            title="Por que as redações zeram (motivos oficiais)"
            desc="Situação da redação segundo o dicionário do INEP, em % do total de participantes. A maioria (sem problemas) e os ausentes ficam de fora deste recorte para destacar os motivos de zero/anulação."
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.status.problemas} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" unit="%" fontSize={12} />
                <YAxis type="category" dataKey="label" width={210} fontSize={11} />
                <Tooltip formatter={(value, _name, item) => [`${value as number}% · ${formatNumber((item as unknown as { payload?: { count?: number } }).payload?.count ?? 0)} redações`, 'total']} />
                <Bar dataKey="pct" fill={XTRI_RED} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-3 text-xs text-slate-500">
              Visão geral: <strong>{data.status.legenda.find((s) => s.codigo === '1')?.pct}%</strong> sem
              problemas · <strong>{data.status.legenda.find((s) => s.codigo === '(ausente)')?.pct}%</strong>{' '}
              ausentes · <strong>{formatNumber(data.status.total_zeradas_anuladas)}</strong> redações zeradas/anuladas.
            </p>
          </ChartCard>

          <ChartCard title="Distribuição das notas" desc="Redações por faixa de 100 pontos (a faixa 0 inclui zeros e anuladas).">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.nota.histograma}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="faixa" fontSize={12} />
                <YAxis tickFormatter={(v) => formatNumber(v)} width={70} fontSize={12} />
                <Tooltip formatter={(value) => [formatNumber(value as number), 'redações']} labelFormatter={(v) => `Faixa ${v}–${Number(v) + 99}`} />
                <Bar dataKey="count" fill={XTRI_BLUE} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Notas dos avaliadores por competência"
            desc="Como os avaliadores distribuíram as notas (0 a 200, em níveis de 40) em cada competência. Cada barra soma 100%."
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stackedData(data, 'dist_niveis_pct')} layout="vertical" margin={{ left: 8 }} stackOffset="expand">
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={12} />
                <YAxis type="category" dataKey="competencia" width={44} fontSize={12} />
                <Tooltip formatter={(value, name) => [`${(value as number).toFixed(1)}%`, `${name} pts`]} />
                <Legend formatter={(v) => `${v} pts`} />
                {LEVELS.map((l) => (
                  <Bar key={l} dataKey={String(l)} stackId="a" fill={GRADE_COLORS[l]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Quanto os dois avaliadores divergem"
            desc={`Diferença entre Avaliador 1 e Avaliador 2 por competência (0 = deram a mesma nota). ${data.divergencia_avaliadores.pct_terceiro_avaliador}% das redações precisaram de um 3º avaliador. A ${data.divergencia_avaliadores.competencia_mais_divergente} é a mais subjetiva.`}
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stackedData(data, 'dist_divergencia_pct')} layout="vertical" margin={{ left: 8 }} stackOffset="expand">
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={12} />
                <YAxis type="category" dataKey="competencia" width={44} fontSize={12} />
                <Tooltip formatter={(value, name) => [`${(value as number).toFixed(1)}%`, name === '0' ? 'iguais' : `${name} pts de diferença`]} />
                <Legend formatter={(v) => (v === '0' ? 'iguais' : `${v} pts`)} />
                {LEVELS.map((l) => (
                  <Bar key={l} dataKey={String(l)} stackId="a" fill={DIFF_COLORS[l]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Mapa da divergência entre avaliadores por estado"
            desc={data.divergencia_por_uf.descricao}
          >
            <RedacaoDivergenceMap ufs={data.divergencia_por_uf.ufs} />
          </ChartCard>

          <ChartCard title="Nota média por competência" desc={`Onde os participantes mais perdem pontos. A ${data.divergencia_avaliadores.competencia_mais_dificil} é a de menor média.`}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={data.divergencia_avaliadores.por_competencia} layout="vertical" margin={{ left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" domain={[0, 200]} fontSize={12} />
                <YAxis type="category" dataKey="competencia" width={44} fontSize={12} />
                <Tooltip
                  formatter={(value) => [`${formatNumber(value as number)} / 200`, 'nota média']}
                  labelFormatter={(c) => data.divergencia_avaliadores.por_competencia.find((x) => x.competencia === c)?.nome ?? c}
                />
                <Bar dataKey="nota_media" radius={[0, 4, 4, 0]}>
                  {data.divergencia_avaliadores.por_competencia.map((c) => (
                    <Cell key={c.competencia} fill={c.competencia === data.divergencia_avaliadores.competencia_mais_dificil ? XTRI_RED : XTRI_BLUE} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
              {data.divergencia_avaliadores.por_competencia.map((c) => (
                <div key={c.competencia} className="rounded-lg border border-slate-200 bg-white p-3 text-sm">
                  <p className="font-semibold text-slate-800">{c.nome}</p>
                  <p className="mt-1 text-slate-600">média {formatNumber(c.nota_media)} · {c.concordancia_exata_pct}% avaliadores iguais</p>
                </div>
              ))}
            </div>
          </ChartCard>

          <footer className="mt-12 flex items-center gap-3 border-t border-slate-200 pt-6">
            <Image src="/logo-xtri.png" alt="X-TRI" width={28} height={28} className="h-7 w-7 object-contain" />
            <p className="text-xs text-slate-400">
              Fonte: {data.fonte}. {data.observacao} Os microdados do INEP são anônimos — não é possível
              (nem permitido) consultar o espelho individual de um participante por número de inscrição.
            </p>
          </footer>
        </>
      )}
    </main>
  );
}
