'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { formatNumber, formatRanking } from '@/lib/utils';
import Link from 'next/link';
import { Search, ChevronLeft, ChevronRight, Bell, School as SchoolIcon, SearchX } from 'lucide-react';
import { TableRowSkeleton } from '@/components/ui/skeleton';
import { EmptyState } from '@/components/ui/empty-state';
import { RangeSlider } from '@/components/ui/RangeSlider';

const UF_OPTIONS = [
  '', 'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS',
  'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO'
];

const fmtNota = (v: number | null | undefined) =>
  v == null ? '-' : v.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// Média só das 4 áreas objetivas (LC, CH, CN, MT) — sem redação
const mediaObjetivas = (s: { media_cn: number | null; media_ch: number | null; media_lc: number | null; media_mt: number | null }) => {
  const areas = [s.media_lc, s.media_ch, s.media_cn, s.media_mt];
  return areas.every((v) => v != null) ? (areas as number[]).reduce((a, b) => a + b, 0) / 4 : null;
};

export default function SchoolsPage() {
  const [search, setSearch] = useState('');
  const [uf, setUf] = useState('');
  const [municipio, setMunicipio] = useState('');
  const [tipoEscola, setTipoEscola] = useState<'Privada' | 'Pública' | ''>('');
  const [localizacao, setLocalizacao] = useState<'Urbana' | 'Rural' | ''>('');
  const [alunos, setAlunos] = useState<[number, number] | null>(null);
  const [media, setMedia] = useState<[number, number] | null>(null);
  const [page, setPage] = useState(1);
  const limit = 50;

  // Limites dos sliders para o recorte categórico atual
  const { data: bounds } = useQuery({
    queryKey: ['filter-bounds', uf, municipio, tipoEscola, localizacao],
    queryFn: () => api.getFilterBounds({
      uf: uf || undefined,
      municipio: municipio || undefined,
      tipo_escola: tipoEscola || undefined,
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

  const { data: schools, isLoading } = useQuery({
    queryKey: ['schools', search, uf, municipio, tipoEscola, localizacao, alunos, media, page],
    queryFn: () => api.listSchools({
      page,
      limit,
      search: search || undefined,
      uf: uf || undefined,
      municipio: municipio || undefined,
      tipo_escola: tipoEscola || undefined,
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
    queryKey: ['municipios', uf],
    queryFn: () => api.getMunicipios(undefined, uf),
    enabled: Boolean(uf),
  });

  const hasMore = schools?.length === limit;
  const municipios = municipiosData?.municipios ?? [];

  useEffect(() => {
    if (municipio && municipios.length > 0 && !municipios.includes(municipio)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- limpa o município quando ele sai da lista do novo recorte
      setMunicipio('');
      setPage(1);
    }
  }, [municipio, municipios]);

  return (
    <div className="min-h-screen">
      {/* Page Header */}
      <div className="bg-white border-b border-slate-200 sticky top-0 z-20">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 bg-[#E9F8FE] rounded-xl flex items-center justify-center">
                <SchoolIcon className="h-5 w-5 text-[#28B7ED]" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Escolas</h1>
                <p className="text-sm text-slate-500">Busque e filtre escolas por nome, estado, município ou faixa de notas</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
                <Bell className="h-5 w-5 text-slate-600" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
      {/* Filters */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 space-y-5">
        <div className="flex flex-col gap-4 md:flex-row md:flex-wrap">
          <div className="min-w-0 flex-1 md:min-w-[260px]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Buscar por nome ou código INEP..."
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setPage(1);
                }}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-slate-950 placeholder:text-slate-400 caret-[#28B7ED] focus:ring-2 focus:ring-[#28B7ED] focus:border-[#28B7ED] outline-none"
              />
            </div>
          </div>
          <div className="w-full md:w-48">
            <select
              value={uf}
              onChange={(e) => {
                setUf(e.target.value);
                setMunicipio('');
                setPage(1);
              }}
              className={`w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#28B7ED] focus:border-[#28B7ED] outline-none bg-white ${uf ? 'text-slate-950' : 'text-slate-500'}`}
            >
              <option value="" className="text-slate-500">Todos os Estados</option>
              {UF_OPTIONS.filter(Boolean).map((state) => (
                <option key={state} value={state} className="text-slate-950">{state}</option>
              ))}
            </select>
          </div>
          <div className="w-full md:w-56">
            <select
              value={municipio}
              onChange={(e) => {
                setMunicipio(e.target.value);
                setPage(1);
              }}
              disabled={!uf || municipiosLoading}
              className={`w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#28B7ED] focus:border-[#28B7ED] outline-none bg-white disabled:cursor-not-allowed disabled:opacity-60 ${municipio ? 'text-slate-950' : 'text-slate-500'}`}
            >
              <option value="" className="text-slate-500">
                {!uf ? 'Escolha um Estado' : municipiosLoading ? 'Carregando municípios' : 'Todos os Municípios'}
              </option>
              {municipios.map((city) => (
                <option key={city} value={city} className="text-slate-950">{city}</option>
              ))}
            </select>
          </div>
          <div className="w-full md:w-40">
            <select
              value={tipoEscola}
              onChange={(e) => {
                setTipoEscola(e.target.value as 'Privada' | 'Pública' | '');
                setPage(1);
              }}
              className={`w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#28B7ED] focus:border-[#28B7ED] outline-none bg-white ${tipoEscola ? 'text-slate-950' : 'text-slate-500'}`}
            >
              <option value="" className="text-slate-500">Todas as Redes</option>
              <option value="Privada" className="text-slate-950">Privada</option>
              <option value="Pública" className="text-slate-950">Pública</option>
            </select>
          </div>
          <div className="w-full md:w-36">
            <select
              value={localizacao}
              onChange={(e) => {
                setLocalizacao(e.target.value as 'Urbana' | 'Rural' | '');
                setPage(1);
              }}
              className={`w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#28B7ED] focus:border-[#28B7ED] outline-none bg-white ${localizacao ? 'text-slate-950' : 'text-slate-500'}`}
            >
              <option value="" className="text-slate-500">Todas Localizações</option>
              <option value="Urbana" className="text-slate-950">Urbana</option>
              <option value="Rural" className="text-slate-950">Rural</option>
            </select>
          </div>
        </div>

        {/* Range sliders */}
        {bounds && alunos && media && bounds.max_alunos > 0 && (
          <div className="grid gap-x-10 gap-y-5 border-t border-gray-100 pt-4 sm:grid-cols-2">
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
              <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-slate-500">Média geral (TRI)</label>
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

      {/* Results */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        {!isLoading && schools?.length === 0 ? (
          <EmptyState
            icon={SearchX}
            title="Nenhuma escola encontrada"
            description="Ajuste os filtros ou limpe a busca para ver todas as escolas."
            action={{
              label: "Limpar filtros",
              onClick: () => {
                setSearch('');
                setUf('');
                setMunicipio('');
                setTipoEscola('');
                setLocalizacao('');
                setPage(1);
              },
            }}
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Posição Geral</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Posição</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Escola</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cidade</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Dependência</th>
                    <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Alunos</th>
                    <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">LC</th>
                    <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">CH</th>
                    <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">CN</th>
                    <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">MT</th>
                    <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">RD</th>
                    <th className="px-3 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Média Obj.</th>
                    <th className="px-3 py-3 text-right text-xs font-bold text-gray-700 uppercase tracking-wider">Média Geral</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {isLoading
                    ? Array.from({ length: 10 }).map((_, i) => (
                        <TableRowSkeleton key={i} cols={13} />
                      ))
                    : schools?.map((school, i) => (
                    <tr key={school.codigo_inep} className="hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-4 whitespace-nowrap text-gray-500 font-medium">
                        {formatRanking(school.ultimo_ranking)}
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap font-bold text-[#1B2A6B]">
                        {(page - 1) * limit + i + 1}
                      </td>
                      <td className="px-4 py-4">
                        <Link
                          href={`/schools/${school.codigo_inep}`}
                          className="text-blue-600 hover:text-blue-800 font-medium"
                        >
                          {school.nome_escola}
                        </Link>
                        <p className="text-gray-400 text-xs">{school.codigo_inep}</p>
                      </td>
                      <td className="px-4 py-4 text-sm">
                        <span className="block max-w-[150px] truncate text-gray-900" title={school.municipio ?? ''}>{school.municipio || '-'}</span>
                        {school.uf && <span className="block text-xs text-gray-400">{school.uf}</span>}
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap">
                        {school.tipo_escola && (
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            school.tipo_escola === 'Privada'
                              ? 'bg-purple-100 text-purple-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            {school.tipo_escola}
                          </span>
                        )}
                        {school.localizacao && <span className="block mt-1 text-xs text-gray-400">{school.localizacao}</span>}
                      </td>
                      <td className="px-3 py-4 whitespace-nowrap text-right text-gray-600">
                        {school.qt_matriculas ? formatNumber(school.qt_matriculas) : '-'}
                      </td>
                      <td className="px-3 py-4 whitespace-nowrap text-right text-gray-600 tabular-nums">{fmtNota(school.media_lc)}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-right text-gray-600 tabular-nums">{fmtNota(school.media_ch)}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-right text-gray-600 tabular-nums">{fmtNota(school.media_cn)}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-right text-gray-600 tabular-nums">{fmtNota(school.media_mt)}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-right text-gray-600 tabular-nums">{fmtNota(school.media_redacao)}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-right font-semibold text-gray-700 tabular-nums">{fmtNota(mediaObjetivas(school))}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-right font-bold text-gray-900 tabular-nums">{fmtNota(school.ultima_nota)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="flex items-center gap-1 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
                Anterior
              </button>
              <span className="text-sm text-gray-600">Página {page}</span>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={!hasMore}
                className="flex items-center gap-1 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Próxima
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </>
        )}
      </div>

      <footer className="pt-2 text-center text-xs text-slate-500">
        <Link href="/termos-de-uso" className="font-semibold text-[#139ED3] hover:underline">Termos de Uso</Link> ·{' '}
        <Link href="/politica-de-privacidade" className="font-semibold text-[#139ED3] hover:underline">Política de Privacidade</Link> ·{' '}
        <Link href="/politica-de-compliance-ia" className="font-semibold text-[#139ED3] hover:underline">Compliance IA</Link>
      </footer>
      </div>
    </div>
  );
}
