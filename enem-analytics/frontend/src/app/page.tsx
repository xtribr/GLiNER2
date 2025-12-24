'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { formatNumber } from '@/lib/utils';
import Link from 'next/link';
import { Trophy, School, Calendar, MapPin } from 'lucide-react';

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: api.getStats,
  });

  const { data: topSchools, isLoading: topLoading } = useQuery({
    queryKey: ['topSchools'],
    queryFn: () => api.getTopSchools(10),
  });

  if (statsLoading || topLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard ENEM</h1>
        <p className="text-gray-600 mt-1">
          Análise de desempenho de escolas no ENEM ({stats?.years[0]} - {stats?.years[stats?.years.length - 1]})
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={School}
          label="Total de Escolas"
          value={stats?.total_schools.toLocaleString('pt-BR') || '-'}
          color="blue"
        />
        <StatCard
          icon={Calendar}
          label="Anos de Dados"
          value={`${stats?.years.length || 0} anos`}
          color="green"
        />
        <StatCard
          icon={Trophy}
          label="Total de Registros"
          value={stats?.total_records.toLocaleString('pt-BR') || '-'}
          color="purple"
        />
        <StatCard
          icon={MapPin}
          label="Estados"
          value={`${stats?.states.length || 0} UFs`}
          color="orange"
        />
      </div>

      {/* Average Scores */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Médias Nacionais</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: 'Ciências da Natureza', value: stats?.avg_scores.nota_cn, color: 'bg-green-100 text-green-800' },
            { label: 'Ciências Humanas', value: stats?.avg_scores.nota_ch, color: 'bg-blue-100 text-blue-800' },
            { label: 'Linguagens', value: stats?.avg_scores.nota_lc, color: 'bg-purple-100 text-purple-800' },
            { label: 'Matemática', value: stats?.avg_scores.nota_mt, color: 'bg-orange-100 text-orange-800' },
            { label: 'Redação', value: stats?.avg_scores.nota_redacao, color: 'bg-red-100 text-red-800' },
          ].map((item) => (
            <div key={item.label} className={`rounded-lg p-4 ${item.color}`}>
              <p className="text-sm font-medium opacity-80">{item.label}</p>
              <p className="text-2xl font-bold mt-1">{formatNumber(item.value)}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Top Schools */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">Top 10 Escolas - {topSchools?.ano}</h2>
            <Link
              href="/schools"
              className="text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              Ver todas →
            </Link>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Ranking
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Escola
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  UF
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Média
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  CN
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  CH
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  LC
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  MT
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  RED
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {topSchools?.schools.map((school, index) => (
                <tr
                  key={school.codigo_inep}
                  className="hover:bg-gray-50 transition-colors"
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${
                      index === 0 ? 'bg-yellow-100 text-yellow-800' :
                      index === 1 ? 'bg-gray-200 text-gray-700' :
                      index === 2 ? 'bg-orange-100 text-orange-800' :
                      'bg-gray-100 text-gray-600'
                    }`}>
                      {school.ranking}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <Link
                      href={`/schools/${school.codigo_inep}`}
                      className="text-blue-600 hover:text-blue-800 font-medium"
                    >
                      {school.nome_escola}
                    </Link>
                    <p className="text-gray-500 text-sm">{school.codigo_inep}</p>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {school.uf}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right font-semibold text-gray-900">
                    {formatNumber(school.nota_media)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">
                    {formatNumber(school.nota_cn)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">
                    {formatNumber(school.nota_ch)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">
                    {formatNumber(school.nota_lc)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">
                    {formatNumber(school.nota_mt)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">
                    {formatNumber(school.nota_redacao)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  color: 'blue' | 'green' | 'purple' | 'orange';
}) {
  const colors = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    purple: 'bg-purple-50 text-purple-600',
    orange: 'bg-orange-50 text-orange-600',
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-lg ${colors[color]}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  );
}
