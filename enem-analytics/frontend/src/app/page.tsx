'use client';

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Image from 'next/image';
import Link from 'next/link';
import {
  ArrowRight,
  ChevronRight,
  Database,
  GitCompare,
  MapPin,
  School,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Trophy,
} from 'lucide-react';
import { api, type TopSchool } from '@/lib/api';
import { getYearRangeLabel } from '@/lib/enem-cycle';
import { formatTriScore } from '@/lib/utils';
import { StatCardSkeleton, TableRowSkeleton } from '@/components/ui/skeleton';
import BrazilChoropleth from '@/components/vitrine/BrazilChoropleth';

const UF_OPTIONS = [
  '', 'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS',
  'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO',
];

type AreaMetric = 'nota_media' | 'nota_mt' | 'nota_lc' | 'nota_cn' | 'nota_ch' | 'nota_redacao';
const AREA_TABS: Array<{ key: string; label: string; metric: AreaMetric }> = [
  { key: 'media', label: 'Média Geral', metric: 'nota_media' },
  { key: 'mt', label: 'Matemática', metric: 'nota_mt' },
  { key: 'lc', label: 'Linguagens', metric: 'nota_lc' },
  { key: 'cn', label: 'Ciências', metric: 'nota_cn' },
  { key: 'ch', label: 'Humanas', metric: 'nota_ch' },
  { key: 'red', label: 'Redação', metric: 'nota_redacao' },
];

const FEATURES = [
  { icon: TrendingUp, title: 'Evolução histórica', desc: 'Acompanhe a TRI da sua escola ao longo de toda a série do ENEM.' },
  { icon: GitCompare, title: 'Comparativo por área', desc: 'CN, CH, LC, MT e Redação contra a média nacional.' },
  { icon: Sparkles, title: 'Oráculo IA', desc: 'Diagnóstico e recomendações da IA dedicada da XTRI.' },
];

function variacaoPct(s: TopSchool): number | null {
  const h = s.history;
  if (!h || h.length < 2) return null;
  const sorted = [...h].sort((a, b) => a.ano - b.ano);
  const last = sorted[sorted.length - 1].nota_media;
  const prev = sorted[sorted.length - 2].nota_media;
  if (!prev) return null;
  return ((last - prev) / prev) * 100;
}

export default function Vitrine() {
  const [ano, setAno] = useState<number | undefined>(undefined);
  const [uf, setUf] = useState('');
  const [rede, setRede] = useState('');
  const [areaKey, setAreaKey] = useState('media');

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: api.getStats,
  });

  const { data: topData, isLoading: topLoading } = useQuery({
    queryKey: ['vitrineTop', ano, uf, rede],
    queryFn: () => api.getTopSchools(50, ano, uf || undefined, rede || undefined),
  });

  const metric = AREA_TABS.find((t) => t.key === areaKey)?.metric ?? 'nota_media';
  const schools = topData?.schools ?? [];

  const ranked = useMemo(() => {
    return [...schools]
      .filter((s) => s[metric] != null)
      .sort((a, b) => (b[metric] ?? 0) - (a[metric] ?? 0))
      .slice(0, 10);
  }, [schools, metric]);

  const mediaNacional = useMemo(() => {
    if (!stats?.avg_scores) return null;
    const v = Object.values(stats.avg_scores).filter((n): n is number => typeof n === 'number');
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }, [stats]);

  const maiorMedia = schools.length ? Math.max(...schools.map((s) => s.nota_media ?? 0)) : null;
  const yearRangeLabel = getYearRangeLabel(stats?.years);
  const years = (stats?.years ?? []).slice().sort((a, b) => b - a).slice(0, 6).reverse();
  const latestYear = years.length ? years[years.length - 1] : undefined;

  return (
    <div className="min-h-screen bg-[#f5fbff]">
      {/* NAV */}
      <header className="sticky top-0 z-30 border-b border-[#28B7ED]/20 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1120px] items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-2xl border border-[#28B7ED]/20 bg-white shadow-sm">
              <Image src="/logo-x.png" alt="Logo XTRI" width={34} height={34} className="h-8 w-8 object-contain" priority />
            </div>
            <div>
              <b className="block text-base font-bold leading-tight text-slate-950">Ranking ENEM XTRI</b>
              <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">X-TRI Escolas</span>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <Link href="/login" className="hidden rounded-xl border border-[#28B7ED]/20 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-[#28B7ED]/50 hover:text-[#28B7ED] sm:block">
              Entrar
            </Link>
            <Link href="/cadastro" className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-4 py-2 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105">
              Acessar análise gratuita
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </header>

      {/* HERO */}
      <section className="relative overflow-hidden bg-[#061927] text-white">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_12%_0%,rgba(40,183,237,0.55),transparent_36%),radial-gradient(circle_at_88%_22%,rgba(255,75,46,0.45),transparent_33%)]" />
        <div className="relative mx-auto grid max-w-[1120px] items-center gap-8 px-4 py-12 sm:px-6 lg:grid-cols-[1.4fr_0.9fr]">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold text-[#DFF7FF] backdrop-blur">
              <Database className="h-3.5 w-3.5" />
              {statsLoading ? 'Carregando base oficial' : `Base oficial INEP · ${yearRangeLabel}`}
            </span>
            <h1 className="mt-4 text-3xl font-black leading-[1.08] tracking-tight sm:text-4xl lg:text-5xl">
              Análise estratégica do desempenho da sua escola no{' '}
              <span className="bg-gradient-to-r from-[#28B7ED] to-[#7fe0ff] bg-clip-text text-transparent">ENEM</span>
            </h1>
            <p className="mt-4 max-w-xl text-sm leading-6 text-slate-300 sm:text-base">
              Transformamos os microdados oficiais do INEP em ranking, evolução por TRI e diagnóstico acionável — para gestores e educadores de todo o Brasil.
            </p>
            <div className="mt-6 flex flex-col gap-3 sm:flex-row">
              <Link href="/cadastro" className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-5 py-3 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/30 transition hover:brightness-105">
                Acessar análise gratuita
                <ArrowRight className="h-4 w-4" />
              </Link>
              <a href="#ranking" className="inline-flex items-center justify-center gap-2 rounded-2xl bg-white/95 px-5 py-3 text-sm font-bold text-slate-950 transition hover:bg-white">
                Ver ranking
                <ChevronRight className="h-4 w-4" />
              </a>
            </div>
          </div>

          <div className="flex flex-col gap-3">
            {FEATURES.map((f, i) => (
              <div key={f.title} className="flex items-start gap-3 rounded-2xl border border-white/12 bg-white/[0.07] p-4 backdrop-blur">
                <div className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl ${i === 1 ? 'bg-[#FF4B2E]/20 text-[#FF8A70]' : 'bg-[#28B7ED]/20 text-[#7fe0ff]'}`}>
                  <f.icon className="h-4.5 w-4.5" />
                </div>
                <div>
                  <p className="text-sm font-bold">{f.title}</p>
                  <p className="mt-0.5 text-xs leading-5 text-slate-400">{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FILTER BAR */}
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1120px] flex-col gap-3 px-4 py-4 sm:px-6">
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-1 text-xs font-bold text-slate-500">Ano:</span>
            {(years.length ? years : [latestYear]).filter(Boolean).map((y) => (
              <button
                key={y}
                onClick={() => setAno(y === latestYear && ano === undefined ? undefined : y)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                  (ano ?? latestYear) === y ? 'border-[#28B7ED] bg-[#28B7ED] text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-[#28B7ED]/50'
                }`}
              >
                {y}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <FilterSelect label="Estado" value={uf} onChange={setUf}
              options={UF_OPTIONS.map((u) => ({ value: u, label: u || 'Todos' }))} />
            <FilterSelect label="Rede" value={rede} onChange={setRede}
              options={[{ value: '', label: 'Todas' }, { value: 'Privada', label: 'Privada' }, { value: 'Pública', label: 'Pública' }]} />
            <div className="ml-auto flex flex-wrap gap-1">
              {AREA_TABS.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setAreaKey(t.key)}
                  className={`rounded-full px-3 py-1.5 text-xs font-bold transition ${
                    areaKey === t.key ? 'bg-[#E9F8FE] text-[#139ED3]' : 'text-slate-500 hover:text-[#139ED3]'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-[1120px] space-y-6 px-4 py-6 sm:px-6">
        {/* STATS */}
        <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {statsLoading ? (
            <><StatCardSkeleton /><StatCardSkeleton /><StatCardSkeleton /><StatCardSkeleton /></>
          ) : (
            <>
              <StatCard icon={TrendingUp} tone="cyan" value={formatTriScore(mediaNacional)} label="Média nacional TRI" sub={ano ? `ENEM ${ano}` : `ENEM ${latestYear ?? ''}`} />
              <StatCard icon={School} tone="cyan" value={stats?.total_schools?.toLocaleString('pt-BR') ?? '-'} label="Escolas no ranking" sub="Base consolidada" />
              <StatCard icon={Trophy} tone="orange" value={formatTriScore(maiorMedia)} label="Maior média TRI" sub="Líder do recorte" />
              <StatCard icon={MapPin} tone="orange" value={String(stats?.states?.length ?? 0)} label="UFs cobertas" sub="Brasil inteiro" />
            </>
          )}
        </section>

        {/* RANKING */}
        <section id="ranking" className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-col gap-3 border-b border-slate-100 p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
            <div>
              <h2 className="text-xl font-black tracking-tight text-slate-950">Ranking das Escolas</h2>
              <p className="mt-1 text-sm text-slate-500">
                {AREA_TABS.find((t) => t.key === areaKey)?.label} · {ano ?? latestYear ?? 'último ano'}
                {uf ? ` · ${uf}` : ''}{rede ? ` · ${rede}` : ''}
              </p>
            </div>
            <Link href="/cadastro" className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-4 py-2.5 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105">
              Ver minha escola
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50/80">
                <tr>
                  <th className="px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">#</th>
                  <th className="px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Escola</th>
                  <th className="px-4 py-3 text-center text-[10.5px] font-bold uppercase tracking-wider text-slate-400">UF</th>
                  <th className="px-4 py-3 text-center text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Rede</th>
                  <th className="px-4 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Média TRI</th>
                  <th className="px-4 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Variação</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {topLoading ? (
                  Array.from({ length: 8 }).map((_, i) => <TableRowSkeleton key={i} cols={6} />)
                ) : ranked.length ? (
                  ranked.map((s, idx) => (
                    <RankRow key={s.codigo_inep} pos={idx + 1} school={s} metric={metric} />
                  ))
                ) : (
                  <tr><td colSpan={6} className="p-8 text-center text-sm text-slate-400">Sem dados para este recorte.</td></tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="border-t border-slate-100 p-4 text-center">
            <Link href="/cadastro" className="text-sm font-bold text-[#139ED3] hover:underline">
              🔓 Cadastre-se grátis para ver o ranking completo e a análise da sua escola →
            </Link>
          </div>
        </section>

        {/* PROMO */}
        <Link href="/cadastro" className="block overflow-hidden rounded-3xl bg-gradient-to-r from-[#061927] via-[#139ED3] to-[#28B7ED] p-6 text-white shadow-lg transition hover:brightness-105 sm:p-7">
          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <h3 className="text-xl font-black">Não perca tempo.</h3>
              <p className="mt-1 max-w-xl text-sm text-[#dff1fb]">
                Transforme os dados da sua escola num plano de ação estratégico para o próximo ENEM. Acesso gratuito.
              </p>
            </div>
            <span className="inline-flex flex-shrink-0 items-center gap-2 rounded-2xl bg-white px-5 py-3 text-sm font-bold text-[#06283a]">
              Acessar análise
              <ArrowRight className="h-4 w-4" />
            </span>
          </div>
        </Link>

        {/* MAP — choropleth por UF */}
        <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <h2 className="text-lg font-black tracking-tight text-slate-950">Mapa de calor — média TRI por estado</h2>
          <p className="mt-1 text-sm text-slate-500">Média geral do ENEM por UF, no último ano disponível.</p>
          <div className="mt-4">
            <BrazilChoropleth />
          </div>
        </section>
      </main>

      <footer className="bg-[#061927] py-6 text-center text-xs text-slate-400">
        Dados: Inep/MEC — Microdados ENEM · <Link href="/cadastro" className="text-[#28B7ED] hover:underline">Cadastre-se</Link> · © 2026 XTRI Educação
      </footer>
    </div>
  );
}

function FilterSelect({ label, value, onChange, options }: {
  label: string; value: string; onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs">
      <span className="font-semibold text-slate-500">{label}:</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="bg-transparent font-bold text-slate-700 outline-none">
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}

function StatCard({ icon: Icon, tone, value, label, sub }: {
  icon: React.ElementType; tone: 'cyan' | 'orange'; value: string; label: string; sub: string;
}) {
  const ic = tone === 'cyan' ? 'bg-[#E9F8FE] text-[#139ED3]' : 'bg-[#FFF0EB] text-[#FF4B2E]';
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${ic}`}><Icon className="h-5 w-5" /></div>
      <p className="mt-3 text-2xl font-black tracking-tight text-slate-950">{value}</p>
      <p className="text-xs font-bold text-slate-700">{label}</p>
      <p className="text-[11px] text-slate-400">{sub}</p>
    </div>
  );
}

function RankRow({ pos, school, metric }: { pos: number; school: TopSchool; metric: AreaMetric }) {
  const v = variacaoPct(school);
  const medal = pos === 1 ? 'bg-[#FFF0EB] text-[#FF4B2E]' : pos === 2 ? 'bg-slate-100 text-slate-600' : pos === 3 ? 'bg-[#FFF0EB] text-[#E43E24]' : 'bg-[#E9F8FE] text-[#139ED3]';
  const initials = school.nome_escola.split(' ').filter((w) => w.length > 2).slice(0, 2).map((w) => w[0]).join('').toUpperCase() || school.nome_escola.slice(0, 2).toUpperCase();
  return (
    <tr className="transition hover:bg-[#E9F8FE]/50">
      <td className="px-4 py-3">
        <span className={`inline-flex h-8 min-w-8 items-center justify-center rounded-lg px-2 text-xs font-black ${medal}`}>{pos}º</span>
      </td>
      <td className="px-4 py-3">
        <Link href={`/schools/${school.codigo_inep}`} className="flex items-center gap-3 group">
          <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-[#28B7ED] to-[#139ED3] text-xs font-black text-white">{initials}</span>
          <span className="min-w-0">
            <span className="line-clamp-1 text-sm font-bold text-slate-950 group-hover:text-[#139ED3]">{school.nome_escola}</span>
            <span className="font-mono text-[11px] text-slate-400">{school.codigo_inep}</span>
          </span>
        </Link>
      </td>
      <td className="px-4 py-3 text-center">
        <span className="inline-flex h-7 min-w-7 items-center justify-center rounded-lg bg-slate-100 px-2 text-[11px] font-black text-slate-600">{school.uf || '--'}</span>
      </td>
      <td className="px-4 py-3 text-center">
        <span className={`rounded-full px-2.5 py-1 text-[11px] font-bold ${school.tipo_escola === 'Privada' ? 'bg-[#E9F8FE] text-[#139ED3]' : 'bg-[#FFF0EB] text-[#FF4B2E]'}`}>{school.tipo_escola || '--'}</span>
      </td>
      <td className="px-4 py-3 text-right">
        <span className="text-base font-black text-slate-950">{formatTriScore(school[metric])}</span>
      </td>
      <td className="px-4 py-3 text-right">
        {v == null ? (
          <span className="text-xs text-slate-300">—</span>
        ) : (
          <span className={`inline-flex items-center gap-1 text-xs font-bold ${v >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
            {v >= 0 ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
            {v >= 0 ? '+' : ''}{v.toFixed(1)}%
          </span>
        )}
      </td>
    </tr>
  );
}
