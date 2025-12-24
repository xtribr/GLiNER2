'use client';

import { useQuery } from '@tanstack/react-query';
import { useParams } from 'next/navigation';
import { api } from '@/lib/api';
import { formatNumber, formatRanking, getTrendColor, getTrendIcon } from '@/lib/utils';
import Link from 'next/link';
import { ArrowLeft, Trophy, TrendingUp, Calendar, MapPin } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

export default function SchoolDetailPage() {
  const params = useParams();
  const codigo_inep = params.codigo_inep as string;

  const { data: school, isLoading: schoolLoading } = useQuery({
    queryKey: ['school', codigo_inep],
    queryFn: () => api.getSchool(codigo_inep),
  });

  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ['schoolHistory', codigo_inep],
    queryFn: () => api.getSchoolHistory(codigo_inep),
  });

  if (schoolLoading || historyLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!school) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Escola não encontrada</p>
        <Link href="/schools" className="text-blue-600 hover:text-blue-800 mt-4 inline-block">
          ← Voltar para lista de escolas
        </Link>
      </div>
    );
  }

  const latestScore = school.historico[school.historico.length - 1];

  // Prepare chart data
  const chartData = school.historico.map((h) => ({
    ano: h.ano,
    'Média': h.nota_media,
    'CN': h.nota_cn,
    'CH': h.nota_ch,
    'LC': h.nota_lc,
    'MT': h.nota_mt,
    'Redação': h.nota_redacao,
  }));

  // Radar data for latest year
  const radarData = latestScore ? [
    { subject: 'CN', value: latestScore.nota_cn, fullMark: 1000 },
    { subject: 'CH', value: latestScore.nota_ch, fullMark: 1000 },
    { subject: 'LC', value: latestScore.nota_lc, fullMark: 1000 },
    { subject: 'MT', value: latestScore.nota_mt, fullMark: 1000 },
    { subject: 'RED', value: latestScore.nota_redacao, fullMark: 1000 },
  ] : [];

  return (
    <div className="space-y-6">
      {/* Back button */}
      <Link
        href="/schools"
        className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900"
      >
        <ArrowLeft className="h-4 w-4" />
        Voltar para lista
      </Link>

      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{school.nome_escola}</h1>
            <div className="flex items-center gap-4 mt-2 text-gray-600">
              <span className="flex items-center gap-1">
                <MapPin className="h-4 w-4" />
                {school.uf}
              </span>
              <span>INEP: {school.codigo_inep}</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {school.tendencia && (
              <div className={`flex items-center gap-1 ${getTrendColor(school.tendencia)}`}>
                <TrendingUp className="h-5 w-5" />
                <span className="font-medium capitalize">{school.tendencia}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          icon={Trophy}
          label="Ranking Atual"
          value={formatRanking(latestScore?.ranking_brasil)}
          color="yellow"
        />
        <StatCard
          icon={TrendingUp}
          label="Nota Média"
          value={formatNumber(latestScore?.nota_media)}
          color="blue"
        />
        <StatCard
          icon={Calendar}
          label="Anos no ENEM"
          value={`${school.historico.length} anos`}
          color="green"
        />
        <StatCard
          icon={Trophy}
          label="Melhor Ranking"
          value={`${formatRanking(school.melhor_ranking)} (${school.melhor_ano})`}
          color="purple"
        />
      </div>

      {/* Latest Scores */}
      {latestScore && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Notas {latestScore.ano}
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              { label: 'Ciências da Natureza', value: latestScore.nota_cn, color: 'bg-green-100 text-green-800' },
              { label: 'Ciências Humanas', value: latestScore.nota_ch, color: 'bg-blue-100 text-blue-800' },
              { label: 'Linguagens', value: latestScore.nota_lc, color: 'bg-purple-100 text-purple-800' },
              { label: 'Matemática', value: latestScore.nota_mt, color: 'bg-orange-100 text-orange-800' },
              { label: 'Redação', value: latestScore.nota_redacao, color: 'bg-red-100 text-red-800' },
            ].map((item) => (
              <div key={item.label} className={`rounded-lg p-4 ${item.color}`}>
                <p className="text-sm font-medium opacity-80">{item.label}</p>
                <p className="text-2xl font-bold mt-1">{formatNumber(item.value)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Evolution Chart */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Evolução das Notas</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ano" />
              <YAxis domain={[400, 1000]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Média" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="MT" stroke="#f97316" strokeWidth={2} />
              <Line type="monotone" dataKey="Redação" stroke="#ef4444" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Perfil de Desempenho</h2>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis angle={30} domain={[0, 1000]} />
              <Radar
                name="Notas"
                dataKey="value"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.5}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Histórico Completo</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Ano</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Ranking</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Variação</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Média</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">CN</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">CH</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">LC</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">MT</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">RED</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {history?.history.map((h) => (
                <tr key={h.ano} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">{h.ano}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">{formatRanking(h.ranking_brasil)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    {h.ranking_change !== null && (
                      <span className={h.ranking_change > 0 ? 'text-green-600' : h.ranking_change < 0 ? 'text-red-600' : 'text-gray-500'}>
                        {h.ranking_change > 0 ? `+${h.ranking_change}` : h.ranking_change === 0 ? '-' : h.ranking_change}
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right font-semibold">{formatNumber(h.nota_media)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_cn)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_ch)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_lc)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_mt)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_redacao)}</td>
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
  color: 'blue' | 'green' | 'purple' | 'yellow';
}) {
  const colors = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    purple: 'bg-purple-50 text-purple-600',
    yellow: 'bg-yellow-50 text-yellow-600',
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-lg ${colors[color]}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  );
}
