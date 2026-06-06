'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, ArrowRight, Lock, Trophy } from 'lucide-react';
import { api } from '@/lib/api';
import { formatTriScore } from '@/lib/utils';

const AREAS = [
  { key: 'nota_cn', label: 'Ciências da Natureza', short: 'CN' },
  { key: 'nota_ch', label: 'Ciências Humanas', short: 'CH' },
  { key: 'nota_lc', label: 'Linguagens', short: 'LC' },
  { key: 'nota_mt', label: 'Matemática', short: 'MT' },
  { key: 'nota_redacao', label: 'Redação', short: 'RED' },
] as const;

interface PublicSchoolSummaryProps {
  codigoInep: string;
}

// Resumo PÚBLICO de uma escola (visitante deslogado). Mostra apenas o snapshot
// do último ano (posição, média TRI, notas por área) e empurra o cadastro para
// a análise profunda. O endpoint /summary não expõe histórico nem PII.
export default function PublicSchoolSummary({ codigoInep }: PublicSchoolSummaryProps) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['schoolSummary', codigoInep],
    queryFn: () => api.getSchoolSummary(codigoInep),
  });

  return (
    <div className="min-h-screen bg-[#f5fbff]">
      {/* Header público */}
      <header className="sticky top-0 z-20 border-b border-[#28B7ED]/20 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Link href="/" className="flex min-w-0 items-center gap-3">
            <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-[#28B7ED]/20 bg-white shadow-sm">
              <Image src="/logo-x.png" alt="Logo XTRI" width={34} height={34} className="h-8 w-8 object-contain" />
            </div>
            <span className="truncate text-base font-bold text-slate-950">Ranking ENEM XTRI</span>
          </Link>
          <div className="flex flex-shrink-0 items-center gap-2 sm:gap-3">
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

      <main className="mx-auto max-w-5xl px-4 py-6 sm:px-6 sm:py-8">
        <Link href="/schools" className="inline-flex items-center gap-2 text-sm font-semibold text-[#139ED3] hover:underline">
          <ArrowLeft className="h-4 w-4" />
          Voltar ao ranking
        </Link>

        {isLoading ? (
          <div className="mt-5 h-48 animate-pulse rounded-3xl bg-white shadow-sm" />
        ) : isError || !data ? (
          <div className="mt-5 rounded-3xl border border-slate-200 bg-white p-10 text-center shadow-sm">
            <h1 className="text-xl font-black text-slate-950">Escola não encontrada</h1>
            <p className="mt-2 text-sm text-slate-500">Não há dados públicos para o código {codigoInep}.</p>
          </div>
        ) : (
          <>
            {/* Hero da escola */}
            <section className="mt-5 overflow-hidden rounded-3xl border border-[#28B7ED]/25 bg-[#061927] text-white shadow-xl">
              <div className="relative p-6 sm:p-8">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_12%_0%,rgba(40,183,237,0.5),transparent_38%),radial-gradient(circle_at_88%_20%,rgba(255,75,46,0.4),transparent_34%)]" />
                <div className="relative">
                  <div className="flex flex-wrap items-center gap-2 text-xs font-semibold">
                    {data.uf && <span className="rounded-lg bg-white/10 px-2.5 py-1">{data.uf}</span>}
                    {data.tipo_escola && <span className="rounded-full bg-white/10 px-2.5 py-1">{data.tipo_escola}</span>}
                    {data.ultimo_ano && <span className="rounded-full bg-white/10 px-2.5 py-1">ENEM {data.ultimo_ano}</span>}
                  </div>
                  <h1 className="mt-3 text-2xl font-black tracking-tight sm:text-3xl">{data.nome_escola}</h1>
                  <p className="mt-1 font-mono text-xs text-slate-400">INEP {data.codigo_inep}</p>

                  <div className="mt-6 flex flex-wrap gap-6">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#DFF7FF]/80">Posição nacional</p>
                      <p className="mt-1 flex items-center gap-2 text-3xl font-black">
                        <Trophy className="h-6 w-6 text-[#FFCB6B]" />
                        {data.ranking_brasil ? `#${data.ranking_brasil}` : '--'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#DFF7FF]/80">Média TRI</p>
                      <p className="mt-1 text-3xl font-black">{formatTriScore(data.nota_media)}</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#DFF7FF]/80">Anos no ENEM</p>
                      <p className="mt-1 text-3xl font-black">{data.anos_participacao}</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Notas por área */}
            <section className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-5">
              {AREAS.map((area) => (
                <div key={area.key} className="rounded-2xl border border-slate-200 bg-white p-4 text-center shadow-sm">
                  <p className="text-[11px] font-black uppercase tracking-[0.14em] text-[#139ED3]">{area.short}</p>
                  <p className="mt-2 text-xl font-black text-slate-950">{formatTriScore(data[area.key])}</p>
                  <p className="mt-1 truncate text-[11px] text-slate-400" title={area.label}>{area.label}</p>
                </div>
              ))}
            </section>

            {/* CTA gated: análise profunda */}
            <section className="mt-5 overflow-hidden rounded-3xl border border-[#FF4B2E]/25 bg-gradient-to-br from-white to-[#FFF0EB]/50 p-6 shadow-sm sm:p-8">
              <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-start gap-3">
                  <div className="rounded-2xl bg-[#FFF0EB] p-3 text-[#FF4B2E]">
                    <Lock className="h-6 w-6" />
                  </div>
                  <div>
                    <h2 className="text-lg font-black text-slate-950">Análise completa da escola</h2>
                    <p className="mt-1 max-w-xl text-sm leading-6 text-slate-500">
                      Evolução histórica por TRI, diagnóstico por área, projeções e recomendações do Oráculo IA — gratuito para a sua escola após o cadastro.
                    </p>
                  </div>
                </div>
                <Link
                  href="/cadastro"
                  className="inline-flex flex-shrink-0 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-5 py-3 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105"
                >
                  Acessar análise gratuita
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
