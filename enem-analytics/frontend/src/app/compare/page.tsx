'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { formatNumber, formatRanking } from '@/lib/utils';
import { Search, GitCompare } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function ComparePage() {
  const [search1, setSearch1] = useState('');
  const [search2, setSearch2] = useState('');
  const [school1, setSchool1] = useState<string | null>(null);
  const [school2, setSchool2] = useState<string | null>(null);

  const { data: results1 } = useQuery({
    queryKey: ['search', search1],
    queryFn: () => api.searchSchools(search1, 5),
    enabled: search1.length >= 2,
  });

  const { data: results2 } = useQuery({
    queryKey: ['search', search2],
    queryFn: () => api.searchSchools(search2, 5),
    enabled: search2.length >= 2,
  });

  const { data: comparison, isLoading: comparing } = useQuery({
    queryKey: ['compare', school1, school2],
    queryFn: () => api.compareSchools(school1!, school2!),
    enabled: !!school1 && !!school2,
  });

  const chartData = comparison?.comparison.map((c) => ({
    ano: c.ano,
    [comparison.escola1.nome_escola.slice(0, 20)]: c.escola1.nota_media,
    [comparison.escola2.nome_escola.slice(0, 20)]: c.escola2.nota_media,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Comparar Escolas</h1>
        <p className="text-gray-600 mt-1">
          Compare o desempenho de duas escolas lado a lado
        </p>
      </div>

      {/* School Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* School 1 */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Escola 1</h3>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Buscar escola..."
              value={search1}
              onChange={(e) => setSearch1(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            />
          </div>
          {results1 && search1.length >= 2 && !school1 && (
            <div className="mt-2 border border-gray-200 rounded-lg overflow-hidden">
              {results1.map((s) => (
                <button
                  key={s.codigo_inep}
                  onClick={() => {
                    setSchool1(s.codigo_inep);
                    setSearch1(s.nome_escola);
                  }}
                  className="w-full px-4 py-2 text-left hover:bg-gray-50 border-b last:border-b-0"
                >
                  <p className="font-medium text-gray-900">{s.nome_escola}</p>
                  <p className="text-sm text-gray-500">{s.uf} - {s.codigo_inep}</p>
                </button>
              ))}
            </div>
          )}
          {school1 && (
            <div className="mt-2 p-3 bg-blue-50 rounded-lg">
              <p className="font-medium text-blue-900">{search1}</p>
              <button
                onClick={() => {
                  setSchool1(null);
                  setSearch1('');
                }}
                className="text-sm text-blue-600 hover:text-blue-800 mt-1"
              >
                Alterar
              </button>
            </div>
          )}
        </div>

        {/* School 2 */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Escola 2</h3>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Buscar escola..."
              value={search2}
              onChange={(e) => setSearch2(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            />
          </div>
          {results2 && search2.length >= 2 && !school2 && (
            <div className="mt-2 border border-gray-200 rounded-lg overflow-hidden">
              {results2.map((s) => (
                <button
                  key={s.codigo_inep}
                  onClick={() => {
                    setSchool2(s.codigo_inep);
                    setSearch2(s.nome_escola);
                  }}
                  className="w-full px-4 py-2 text-left hover:bg-gray-50 border-b last:border-b-0"
                >
                  <p className="font-medium text-gray-900">{s.nome_escola}</p>
                  <p className="text-sm text-gray-500">{s.uf} - {s.codigo_inep}</p>
                </button>
              ))}
            </div>
          )}
          {school2 && (
            <div className="mt-2 p-3 bg-green-50 rounded-lg">
              <p className="font-medium text-green-900">{search2}</p>
              <button
                onClick={() => {
                  setSchool2(null);
                  setSearch2('');
                }}
                className="text-sm text-green-600 hover:text-green-800 mt-1"
              >
                Alterar
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Comparison Results */}
      {comparing && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      )}

      {comparison && (
        <div className="space-y-6">
          {/* Chart */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Evolução Comparativa</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ano" />
                <YAxis domain={[400, 900]} />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey={comparison.escola1.nome_escola.slice(0, 20)}
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={{ r: 5 }}
                />
                <Line
                  type="monotone"
                  dataKey={comparison.escola2.nome_escola.slice(0, 20)}
                  stroke="#22c55e"
                  strokeWidth={3}
                  dot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Table */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Comparação por Ano</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Ano</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-blue-600 uppercase" colSpan={2}>
                      {comparison.escola1.nome_escola.slice(0, 30)}
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-green-600 uppercase" colSpan={2}>
                      {comparison.escola2.nome_escola.slice(0, 30)}
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Diferença</th>
                  </tr>
                  <tr>
                    <th></th>
                    <th className="px-3 py-2 text-right text-xs text-gray-400">Ranking</th>
                    <th className="px-3 py-2 text-right text-xs text-gray-400">Média</th>
                    <th className="px-3 py-2 text-right text-xs text-gray-400">Ranking</th>
                    <th className="px-3 py-2 text-right text-xs text-gray-400">Média</th>
                    <th className="px-3 py-2 text-right text-xs text-gray-400">Média</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {comparison.comparison.map((c) => {
                    const diff = (c.escola1.nota_media || 0) - (c.escola2.nota_media || 0);
                    return (
                      <tr key={c.ano} className="hover:bg-gray-50">
                        <td className="px-6 py-4 font-medium text-gray-900">{c.ano}</td>
                        <td className="px-3 py-4 text-right text-blue-600">{formatRanking(c.escola1.ranking)}</td>
                        <td className="px-3 py-4 text-right font-semibold text-blue-600">{formatNumber(c.escola1.nota_media)}</td>
                        <td className="px-3 py-4 text-right text-green-600">{formatRanking(c.escola2.ranking)}</td>
                        <td className="px-3 py-4 text-right font-semibold text-green-600">{formatNumber(c.escola2.nota_media)}</td>
                        <td className={`px-6 py-4 text-right font-semibold ${diff > 0 ? 'text-blue-600' : diff < 0 ? 'text-green-600' : 'text-gray-500'}`}>
                          {diff > 0 ? '+' : ''}{formatNumber(diff)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {!school1 && !school2 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <GitCompare className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900">Selecione duas escolas para comparar</h3>
          <p className="text-gray-500 mt-2">
            Busque e selecione as escolas nos campos acima para ver a comparação
          </p>
        </div>
      )}
    </div>
  );
}
