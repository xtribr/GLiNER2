'use client';

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Image from 'next/image';
import Link from 'next/link';
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Database,
  GitCompare,
  MapPin,
  School,
  Sparkles,
  TrendingUp,
  Trophy,
} from 'lucide-react';
import { api } from '@/lib/api';
import { getYearRangeLabel } from '@/lib/enem-cycle';
import { formatNumber, formatRanking, formatTriScore } from '@/lib/utils';
import { StatCardSkeleton, TableRowSkeleton } from '@/components/ui/skeleton';
import { RangeSlider } from '@/components/ui/RangeSlider';
import BrazilChoropleth from '@/components/vitrine/BrazilChoropleth';

const UF_OPTIONS = [
  '', 'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS',
  'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO',
];

const FEATURES = [
  { icon: TrendingUp, title: 'Evolução histórica', desc: 'Acompanhe a TRI da sua escola ao longo de toda a série do ENEM.' },
  { icon: GitCompare, title: 'Comparativo por área', desc: 'CN, CH, LC, MT e Redação contra a média nacional.' },
  { icon: Sparkles, title: 'Oráculo IA', desc: 'Diagnóstico e recomendações da IA dedicada da XTRI.' },
];

const fmtNota = (v: number | null | undefined) =>
  v == null ? '-' : v.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// Média só das 4 áreas objetivas (LC, CH, CN, MT) — sem redação
const mediaObjetivas = (s: { media_cn: number | null; media_ch: number | null; media_lc: number | null; media_mt: number | null }) => {
  const areas = [s.media_lc, s.media_ch, s.media_cn, s.media_mt];
  return areas.every((v) => v != null) ? (areas as number[]).reduce((a, b) => a + b, 0) / 4 : null;
};

export default function Vitrine() {
  const [ano, setAno] = useState<number | undefined>(undefined);
  const [uf, setUf] = useState('');
  const [municipio, setMunicipio] = useState('');
  const [rede, setRede] = useState<'Privada' | 'Pública' | ''>('');
  const [localizacao, setLocalizacao] = useState<'Urbana' | 'Rural' | ''>('');
  const [alunos, setAlunos] = useState<[number, number] | null>(null);
  const [media, setMedia] = useState<[number, number] | null>(null);
  const [page, setPage] = useState(1);
  const limit = 50;

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: api.getStats,
  });

  // Limites dos sliders para o recorte categórico atual
  const { data: bounds } = useQuery({
    queryKey: ['filter-bounds', ano, uf, municipio, rede, localizacao],
    queryFn: () => api.getFilterBounds({
      ano,
      uf: uf || undefined,
      municipio: municipio || undefined,
      tipo_escola: rede || undefined,
      localizacao: localizacao || undefined,
    }),
  });

  // Ao mudar o recorte categórico, os sliders voltam aos limites do recorte
  useEffect(() => {
    if (bounds) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset dos sliders ao mudar os limites do recorte (sync com a query externa de bounds)
      setAlunos([bounds.min_alunos, bounds.max_alunos]);
      setMedia([Math.floor(bounds.min_media), Math.ceil(bounds.max_media)]);
      setPage(1);
    }
  }, [bounds]);

  const { data: schools, isLoading: schoolsLoading } = useQuery({
    queryKey: ['vitrine-list', ano, uf, municipio, rede, localizacao, alunos, media, page],
    queryFn: () => api.listSchools({
      page,
      limit,
      ano,
      uf: uf || undefined,
      municipio: municipio || undefined,
      tipo_escola: rede || undefined,
      localizacao: localizacao || undefined,
      min_participantes: alunos?.[0],
      max_participantes: alunos?.[1],
      min_media: media?.[0],
      max_media: media?.[1],
      order_by: 'ranking',
      order: 'asc',
    }),
  });

  const { data: municipiosData, isLoading: municipiosLoading } = useQuery({
    queryKey: ['municipios', ano, uf],
    queryFn: () => api.getMunicipios(ano, uf),
    enabled: Boolean(uf),
  });

  const rows = schools ?? [];
  const municipios = useMemo(() => municipiosData?.municipios ?? [], [municipiosData]);
  const hasMore = rows.length === limit;

  useEffect(() => {
    if (municipio && municipios.length > 0 && !municipios.includes(municipio)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- limpa o município quando ele sai da lista do novo recorte
      setMunicipio('');
      setPage(1);
    }
  }, [municipio, municipios]);

  const mediaNacional = useMemo(() => {
    if (!stats?.avg_scores) return null;
    const v = Object.values(stats.avg_scores).filter((n): n is number => typeof n === 'number');
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }, [stats]);

  const yearRangeLabel = getYearRangeLabel(stats?.years);
  const years = (stats?.years ?? []).slice().sort((a, b) => b - a).slice(0, 6).reverse();
  const latestYear = years.length ? years[years.length - 1] : undefined;

  // Maior média TRI do recorte atual (1ª colocada da página 1, sem filtro de página).
  const topMedia = page === 1 && rows.length ? rows[0].ultima_nota : null;

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
            <Link href="/cadastro" className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-3 py-2 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105 sm:px-4">
              <span className="sm:hidden">Acessar grátis</span>
              <span className="hidden sm:inline">Acessar análise gratuita</span>
              <ArrowRight className="h-4 w-4 flex-shrink-0" />
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
        <div className="mx-auto flex max-w-[1120px] flex-col gap-4 px-4 py-4 sm:px-6">
          {/* Ano */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-1 text-xs font-bold text-slate-500">Ano:</span>
            {(years.length ? years : [latestYear]).filter(Boolean).map((y) => (
              <button
                key={y}
                onClick={() => {
                  setAno(y === latestYear && ano === undefined ? undefined : y);
                  setMunicipio('');
                  setPage(1);
                }}
                className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                  (ano ?? latestYear) === y ? 'border-[#28B7ED] bg-[#28B7ED] text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-[#28B7ED]/50'
                }`}
              >
                {y}
              </button>
            ))}
          </div>

          {/* Categóricos */}
          <div className="flex flex-wrap items-center gap-2">
            <FilterSelect
              label="Estado"
              value={uf}
              onChange={(value) => {
                setUf(value);
                setMunicipio('');
                setPage(1);
              }}
              options={UF_OPTIONS.map((u) => ({ value: u, label: u || 'Todos' }))}
            />
            <FilterSelect
              label="Município"
              value={municipio}
              onChange={(value) => {
                setMunicipio(value);
                setPage(1);
              }}
              disabled={!uf || municipiosLoading}
              options={[
                { value: '', label: !uf ? 'Escolha um estado' : municipiosLoading ? 'Carregando...' : 'Todos' },
                ...municipios.map((city) => ({ value: city, label: city })),
              ]}
            />
            <FilterSelect
              label="Rede"
              value={rede}
              onChange={(value) => {
                setRede(value as 'Privada' | 'Pública' | '');
                setPage(1);
              }}
              options={[{ value: '', label: 'Todas' }, { value: 'Privada', label: 'Privada' }, { value: 'Pública', label: 'Pública' }]}
            />
            <FilterSelect
              label="Localização"
              value={localizacao}
              onChange={(value) => {
                setLocalizacao(value as 'Urbana' | 'Rural' | '');
                setPage(1);
              }}
              options={[{ value: '', label: 'Todas' }, { value: 'Urbana', label: 'Urbana' }, { value: 'Rural', label: 'Rural' }]}
            />
          </div>

          {/* Range sliders */}
          {bounds && alunos && media && bounds.max_alunos > 0 && (
            <div className="grid gap-x-10 gap-y-5 border-t border-slate-100 pt-4 sm:grid-cols-2">
              <div>
                <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-slate-500">Alunos (participantes)</label>
                <RangeSlider
                  min={bounds.min_alunos}
                  max={bounds.max_alunos}
                  value={alunos}
                  onChange={(v) => { setAlunos(v); setPage(1); }}
                />
              </div>
              <div>
                <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-slate-500">Média geral</label>
                <RangeSlider
                  min={Math.floor(bounds.min_media)}
                  max={Math.ceil(bounds.max_media)}
                  value={media}
                  onChange={(v) => { setMedia(v); setPage(1); }}
                />
              </div>
            </div>
          )}
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
              <StatCard icon={Trophy} tone="orange" value={topMedia != null ? formatTriScore(topMedia) : '-'} label="Maior média TRI" sub="Líder do recorte" />
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
                Por posição geral · {ano ?? latestYear ?? 'último ano'}
                {uf ? ` · ${uf}` : ''}{municipio ? ` · ${municipio}` : ''}{rede ? ` · ${rede}` : ''}{localizacao ? ` · ${localizacao}` : ''}
              </p>
            </div>
            <Link href="/cadastro" className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-4 py-2.5 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105">
              Ver minha escola
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>

          {/* Tabela */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50/80">
                <tr>
                  <th className="px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Posição Geral</th>
                  <th className="px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Posição</th>
                  <th className="px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Escola</th>
                  <th className="px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Cidade</th>
                  <th className="px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Dependência</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">Alunos</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">LC</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">CH</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">CN</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">MT</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-400">RD</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-500">Média Obj.</th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-bold uppercase tracking-wider text-slate-700">Média Geral</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {schoolsLoading ? (
                  Array.from({ length: 10 }).map((_, i) => <TableRowSkeleton key={i} cols={13} />)
                ) : rows.length ? (
                  rows.map((school, i) => (
                    <tr key={school.codigo_inep} className="transition hover:bg-[#E9F8FE]/50">
                      <td className="whitespace-nowrap px-4 py-4 font-medium text-slate-500">
                        {formatRanking(school.ultimo_ranking)}
                      </td>
                      <td className="whitespace-nowrap px-4 py-4 font-bold text-[#1B2A6B]">
                        {(page - 1) * limit + i + 1}
                      </td>
                      <td className="px-4 py-4">
                        <Link href={`/schools/${school.codigo_inep}`} className="font-bold text-slate-950 hover:text-[#139ED3]">
                          {school.nome_escola}
                        </Link>
                        <p className="font-mono text-[11px] text-slate-400">{school.codigo_inep}</p>
                      </td>
                      <td className="px-4 py-4 text-sm">
                        <span className="block max-w-[150px] truncate text-slate-900" title={school.municipio ?? ''}>{school.municipio || '-'}</span>
                        {school.uf && <span className="block text-xs text-slate-400">{school.uf}</span>}
                      </td>
                      <td className="whitespace-nowrap px-4 py-4">
                        {school.tipo_escola && (
                          <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-bold ${
                            school.tipo_escola === 'Privada' ? 'bg-[#E9F8FE] text-[#139ED3]' : 'bg-[#FFF0EB] text-[#FF4B2E]'
                          }`}>
                            {school.tipo_escola}
                          </span>
                        )}
                        {school.localizacao && <span className="mt-1 block text-xs text-slate-400">{school.localizacao}</span>}
                      </td>
                      <td className="whitespace-nowrap px-3 py-4 text-right text-slate-600">
                        {school.qt_matriculas ? formatNumber(school.qt_matriculas) : '-'}
                      </td>
                      <td className="whitespace-nowrap px-3 py-4 text-right tabular-nums text-slate-600">{fmtNota(school.media_lc)}</td>
                      <td className="whitespace-nowrap px-3 py-4 text-right tabular-nums text-slate-600">{fmtNota(school.media_ch)}</td>
                      <td className="whitespace-nowrap px-3 py-4 text-right tabular-nums text-slate-600">{fmtNota(school.media_cn)}</td>
                      <td className="whitespace-nowrap px-3 py-4 text-right tabular-nums text-slate-600">{fmtNota(school.media_mt)}</td>
                      <td className="whitespace-nowrap px-3 py-4 text-right tabular-nums text-slate-600">{fmtNota(school.media_redacao)}</td>
                      <td className="whitespace-nowrap px-3 py-4 text-right font-semibold tabular-nums text-slate-700">{fmtNota(mediaObjetivas(school))}</td>
                      <td className="whitespace-nowrap px-3 py-4 text-right font-bold tabular-nums text-slate-950">{fmtNota(school.ultima_nota)}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={12} className="p-8 text-center text-sm text-slate-400">Sem dados para este recorte.</td></tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between border-t border-slate-100 px-6 py-4">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <ChevronLeft className="h-4 w-4" />
              Anterior
            </button>
            <span className="text-sm text-slate-600">Página {page}</span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={!hasMore}
              className="flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Próxima
              <ChevronRight className="h-4 w-4" />
            </button>
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
        Dados: Inep/MEC — Microdados ENEM · <Link href="/cadastro" className="text-[#28B7ED] hover:underline">Cadastre-se</Link> ·{' '}
        <Link href="/termos-de-uso" className="text-[#28B7ED] hover:underline">Termos de Uso</Link> ·{' '}
        <Link href="/politica-de-privacidade" className="text-[#28B7ED] hover:underline">Política de Privacidade</Link> ·{' '}
        <Link href="/politica-de-compliance-ia" className="text-[#28B7ED] hover:underline">Compliance IA</Link> · © 2026 XTRI Educação
      </footer>
    </div>
  );
}

function FilterSelect({ label, value, onChange, options, disabled = false }: {
  label: string; value: string; onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
  disabled?: boolean;
}) {
  return (
    <label className={`flex max-w-full items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs ${disabled ? 'opacity-60' : ''}`}>
      <span className="font-semibold text-slate-500">{label}:</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="min-w-0 max-w-[13rem] bg-transparent font-bold text-slate-700 outline-none disabled:cursor-not-allowed"
      >
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
