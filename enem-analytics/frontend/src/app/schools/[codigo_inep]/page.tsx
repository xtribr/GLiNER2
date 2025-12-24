'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams } from 'next/navigation';
import { api } from '@/lib/api';
import { formatNumber, formatRanking } from '@/lib/utils';
import Link from 'next/link';
import { ArrowLeft, TrendingUp, TrendingDown, Award, BookOpen, Calculator, PenTool, Grid3X3, AlertTriangle, CheckCircle, Lightbulb, Brain } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

export default function SchoolDetailPage() {
  const params = useParams();
  const codigo_inep = params.codigo_inep as string;

  const { data: school, isLoading: schoolLoading } = useQuery({
    queryKey: ['school', codigo_inep],
    queryFn: () => api.getSchool(codigo_inep),
  });

  const { data: history } = useQuery({
    queryKey: ['schoolHistory', codigo_inep],
    queryFn: () => api.getSchoolHistory(codigo_inep),
  });

  const { data: schoolSkills, isLoading: skillsLoading } = useQuery({
    queryKey: ['schoolSkills', codigo_inep],
    queryFn: () => api.getSchoolSkills(codigo_inep, 10),
  });

  // Selected areas for chart - state for interactive selection (must be before early returns)
  const [selectedAreas, setSelectedAreas] = useState<string[]>(['Média']);
  const [selectedSkillArea, setSelectedSkillArea] = useState<string | null>(null);

  if (schoolLoading) {
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
  const previousScore = school.historico.length > 1 ? school.historico[school.historico.length - 2] : null;

  // Calculate changes
  const getChange = (current: number | null | undefined, previous: number | null | undefined) => {
    if (!current || !previous) return null;
    return ((current - previous) / previous * 100).toFixed(1);
  };

  const mediaChange = getChange(latestScore?.nota_media, previousScore?.nota_media);
  const rankingChange = previousScore?.ranking_brasil && latestScore?.ranking_brasil
    ? previousScore.ranking_brasil - latestScore.ranking_brasil
    : null;

  // Area configuration with colors
  const areaConfig: Record<string, { color: string; dataKey: string }> = {
    'Média': { color: '#3b82f6', dataKey: 'Média' },
    'Redação': { color: '#3b82f6', dataKey: 'Redação' },
    'Matemática': { color: '#f97316', dataKey: 'Matemática' },
    'Linguagens': { color: '#ec4899', dataKey: 'Linguagens' },
    'Humanas': { color: '#8b5cf6', dataKey: 'Humanas' },
    'Natureza': { color: '#22c55e', dataKey: 'Natureza' },
  };

  // Toggle area selection
  const toggleArea = (areaName: string) => {
    setSelectedAreas((prev) => {
      if (prev.includes(areaName)) {
        // Don't remove if it's the last one
        if (prev.length === 1) return prev;
        return prev.filter((a) => a !== areaName);
      }
      return [...prev, areaName];
    });
  };

  // Line chart data with all areas
  const lineChartData = school.historico.map((h) => ({
    ano: h.ano,
    Média: h.nota_media,
    Redação: h.nota_redacao,
    Matemática: h.nota_mt,
    Linguagens: h.nota_lc,
    Humanas: h.nota_ch,
    Natureza: h.nota_cn,
  }));

  // Pie chart data
  const pieData = latestScore ? [
    { name: 'CN', value: latestScore.nota_cn, color: '#22c55e' },
    { name: 'CH', value: latestScore.nota_ch, color: '#8b5cf6' },
    { name: 'LC', value: latestScore.nota_lc, color: '#ec4899' },
    { name: 'MT', value: latestScore.nota_mt, color: '#f97316' },
    { name: 'RED', value: latestScore.nota_redacao, color: '#3b82f6' },
  ].filter(d => d.value) : [];

  const totalScore = pieData.reduce((acc, d) => acc + (d.value || 0), 0);

  // Bar data sorted
  const barData = latestScore ? [
    { name: 'Redação', nota: latestScore.nota_redacao || 0, color: '#3b82f6', max: 1000 },
    { name: 'Matemática', nota: latestScore.nota_mt || 0, color: '#f97316', max: 1000 },
    { name: 'Linguagens', nota: latestScore.nota_lc || 0, color: '#ec4899', max: 1000 },
    { name: 'Humanas', nota: latestScore.nota_ch || 0, color: '#8b5cf6', max: 1000 },
    { name: 'Natureza', nota: latestScore.nota_cn || 0, color: '#22c55e', max: 1000 },
  ].sort((a, b) => b.nota - a.nota) : [];

  // Custom label for line chart
  const CustomLabel = ({ x, y, value }: { x: number; y: number; value: number }) => (
    <text x={x} y={y - 10} fill="#6b7280" fontSize={10} textAnchor="middle">
      {value?.toFixed(0)}
    </text>
  );

  return (
    <div className="space-y-6 pb-8">
      {/* Back button */}
      <Link
        href="/schools"
        className="inline-flex items-center gap-2 text-gray-500 hover:text-gray-900 text-sm"
      >
        <ArrowLeft className="h-4 w-4" />
        Voltar
      </Link>

      {/* Header with gradient background */}
      <div className="relative overflow-hidden rounded-3xl p-6 text-white" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)' }}>
        <div className="relative z-10">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-bold">{school.nome_escola}</h1>
              <div className="flex items-center gap-3 mt-2 text-white/80 text-sm">
                <span>{school.uf}</span>
                <span>•</span>
                <span>INEP: {school.codigo_inep}</span>
                {school.tipo_escola && (
                  <>
                    <span>•</span>
                    <span className="px-2 py-0.5 rounded-full bg-white/20 text-xs">
                      {school.tipo_escola}
                    </span>
                  </>
                )}
              </div>
            </div>
            <div className="text-right">
              <p className="text-white/70 text-sm">Ano {latestScore?.ano}</p>
            </div>
          </div>
        </div>
        <div className="absolute -right-10 -top-10 w-40 h-40 bg-white/10 rounded-full blur-2xl"></div>
        <div className="absolute -left-10 -bottom-10 w-32 h-32 bg-white/10 rounded-full blur-xl"></div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-12 gap-5">

        {/* Left Sidebar - KPI Cards */}
        <div className="col-span-12 md:col-span-2 space-y-4">
          {/* Média */}
          <div className="relative overflow-hidden rounded-2xl p-4 text-white" style={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
            <Grid3X3 className="absolute top-3 right-3 h-5 w-5 opacity-50" />
            <p className="text-xs opacity-80 uppercase tracking-wide">Média</p>
            <p className="text-3xl font-bold mt-1">{formatNumber(latestScore?.nota_media)}</p>
            {mediaChange && (
              <div className={`flex items-center gap-1 mt-2 text-xs ${parseFloat(mediaChange) >= 0 ? 'text-green-200' : 'text-red-200'}`}>
                {parseFloat(mediaChange) >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                <span>{parseFloat(mediaChange) >= 0 ? '+' : ''}{mediaChange}%</span>
              </div>
            )}
          </div>

          {/* Ranking */}
          <div className="relative overflow-hidden rounded-2xl p-4 text-white" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
            <Award className="absolute top-3 right-3 h-5 w-5 opacity-50" />
            <p className="text-xs opacity-80 uppercase tracking-wide">Ranking</p>
            <p className="text-3xl font-bold mt-1">#{latestScore?.ranking_brasil}</p>
            {rankingChange !== null && (
              <div className={`flex items-center gap-1 mt-2 text-xs ${rankingChange >= 0 ? 'text-green-200' : 'text-red-200'}`}>
                {rankingChange >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                <span>{rankingChange >= 0 ? '+' : ''}{rankingChange} pos</span>
              </div>
            )}
          </div>

          {/* Redação */}
          <div className="relative overflow-hidden rounded-2xl p-4 text-white" style={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
            <PenTool className="absolute top-3 right-3 h-5 w-5 opacity-50" />
            <p className="text-xs opacity-80 uppercase tracking-wide">Redação</p>
            <p className="text-3xl font-bold mt-1">{formatNumber(latestScore?.nota_redacao)}</p>
          </div>

          {/* Matemática */}
          <div className="relative overflow-hidden rounded-2xl p-4 text-white" style={{ background: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)' }}>
            <Calculator className="absolute top-3 right-3 h-5 w-5 opacity-50" />
            <p className="text-xs opacity-80 uppercase tracking-wide">Matemática</p>
            <p className="text-3xl font-bold mt-1">{formatNumber(latestScore?.nota_mt)}</p>
          </div>

          {/* Habilidades */}
          {latestScore?.desempenho_habilidades && (
            <div className="relative overflow-hidden rounded-2xl p-4 text-white" style={{ background: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)' }}>
              <BookOpen className="absolute top-3 right-3 h-5 w-5 opacity-50" />
              <p className="text-xs opacity-80 uppercase tracking-wide">Habilidades</p>
              <p className="text-3xl font-bold mt-1">{(latestScore.desempenho_habilidades * 100).toFixed(0)}%</p>
            </div>
          )}
        </div>

        {/* Main Content Area */}
        <div className="col-span-12 md:col-span-10 space-y-5">

          {/* Trend Chart */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-gray-900">Evolução das Notas</h2>
              <div className="flex items-center gap-4 text-xs flex-wrap">
                {selectedAreas.map((area) => (
                  <div key={area} className="flex items-center gap-1.5">
                    <div
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: areaConfig[area]?.color || '#3b82f6' }}
                    ></div>
                    <span className="text-gray-500">{area}</span>
                  </div>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={lineChartData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
                <XAxis
                  dataKey="ano"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                />
                <YAxis
                  domain={[400, 1000]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'white',
                    border: 'none',
                    borderRadius: '12px',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                    fontSize: '12px'
                  }}
                  formatter={(value: number) => [value?.toFixed(1), '']}
                />
                {selectedAreas.map((area, index) => (
                  <Line
                    key={area}
                    type="monotone"
                    dataKey={areaConfig[area]?.dataKey || area}
                    stroke={areaConfig[area]?.color || '#3b82f6'}
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: areaConfig[area]?.color || '#3b82f6', strokeWidth: 2, stroke: 'white' }}
                    activeDot={{ r: 6 }}
                    label={index === 0 ? <CustomLabel x={0} y={0} value={0} /> : undefined}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Bottom Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

            {/* Detail Grid */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
              <h3 className="text-base font-semibold text-gray-900 mb-4">Detalhes {latestScore?.ano}</h3>
              <p className="text-xs text-gray-500 mb-3">Clique para visualizar no gráfico</p>
              <div className="overflow-hidden rounded-xl border border-gray-100">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Área</th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">Nota</th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">%</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {barData.map((item) => {
                      const isSelected = selectedAreas.includes(item.name);
                      return (
                        <tr
                          key={item.name}
                          onClick={() => toggleArea(item.name)}
                          className={`cursor-pointer transition-all ${
                            isSelected
                              ? 'bg-blue-50 ring-1 ring-inset ring-blue-200'
                              : 'hover:bg-gray-50'
                          }`}
                        >
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <div
                                className={`w-2.5 h-2.5 rounded-full transition-transform ${isSelected ? 'scale-125' : ''}`}
                                style={{ backgroundColor: item.color }}
                              ></div>
                              <span className={`${isSelected ? 'font-medium text-gray-900' : 'text-gray-700'}`}>
                                {item.name}
                              </span>
                            </div>
                          </td>
                          <td className={`px-3 py-2 text-right font-medium ${isSelected ? 'text-blue-600' : 'text-gray-900'}`}>
                            {formatNumber(item.nota)}
                          </td>
                          <td className="px-3 py-2 text-right text-gray-500">{((item.nota / item.max) * 100).toFixed(0)}%</td>
                        </tr>
                      );
                    })}
                    {/* Média row */}
                    <tr
                      onClick={() => toggleArea('Média')}
                      className={`cursor-pointer transition-all ${
                        selectedAreas.includes('Média')
                          ? 'bg-blue-50 ring-1 ring-inset ring-blue-200'
                          : 'hover:bg-gray-50'
                      }`}
                    >
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-2.5 h-2.5 rounded-full transition-transform ${selectedAreas.includes('Média') ? 'scale-125' : ''}`}
                            style={{ backgroundColor: '#3b82f6' }}
                          ></div>
                          <span className={`${selectedAreas.includes('Média') ? 'font-medium text-gray-900' : 'text-gray-700'}`}>
                            Média
                          </span>
                        </div>
                      </td>
                      <td className={`px-3 py-2 text-right font-medium ${selectedAreas.includes('Média') ? 'text-blue-600' : 'text-gray-900'}`}>
                        {formatNumber(latestScore?.nota_media)}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-500">
                        {latestScore?.nota_media ? ((latestScore.nota_media / 1000) * 100).toFixed(0) : 0}%
                      </td>
                    </tr>
                  </tbody>
                  <tfoot>
                    <tr className="bg-gray-50 font-medium">
                      <td className="px-3 py-2 text-gray-700">Total</td>
                      <td className="px-3 py-2 text-right text-gray-900">{formatNumber(totalScore)}</td>
                      <td className="px-3 py-2 text-right text-gray-500">{((totalScore / 5000) * 100).toFixed(0)}%</td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>

            {/* Pie Chart */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
              <h3 className="text-base font-semibold text-gray-900 mb-4">Distribuição por Área</h3>
              <div className="relative">
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={75}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(value: number, name: string) => [formatNumber(value), name]}
                      contentStyle={{
                        backgroundColor: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                        fontSize: '12px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                {/* Center text */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-gray-900">{((totalScore / 5000) * 100).toFixed(0)}%</p>
                    <p className="text-xs text-gray-500">{formatNumber(totalScore)}</p>
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 mt-2">
                {pieData.map((item) => (
                  <div key={item.name} className="flex items-center gap-1.5 text-xs">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }}></div>
                    <span className="text-gray-600">{item.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Progress Bars */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
              <h3 className="text-base font-semibold text-gray-900 mb-4">Notas por Área</h3>
              <div className="space-y-3">
                {barData.map((item) => (
                  <div key={item.name}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-600">{item.name}</span>
                      <span className="text-xs font-medium text-gray-900">{formatNumber(item.nota)}</span>
                    </div>
                    <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${(item.nota / item.max) * 100}%`,
                          backgroundColor: item.color
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
          <h2 className="text-base font-semibold text-gray-900">Histórico Completo</h2>
          <span className="text-xs text-gray-500">{school.historico.length} anos</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 uppercase">Ano</th>
                <th className="px-5 py-3 text-center text-xs font-medium text-gray-500 uppercase">Ranking</th>
                <th className="px-5 py-3 text-right text-xs font-medium text-gray-500 uppercase">Média</th>
                <th className="px-5 py-3 text-right text-xs font-medium text-gray-500 uppercase">CN</th>
                <th className="px-5 py-3 text-right text-xs font-medium text-gray-500 uppercase">CH</th>
                <th className="px-5 py-3 text-right text-xs font-medium text-gray-500 uppercase">LC</th>
                <th className="px-5 py-3 text-right text-xs font-medium text-gray-500 uppercase">MT</th>
                <th className="px-5 py-3 text-right text-xs font-medium text-gray-500 uppercase">RED</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {history?.history.map((h, idx) => {
                const isLatest = idx === (history.history.length - 1);
                return (
                  <tr key={h.ano} className={`hover:bg-gray-50 transition-colors ${isLatest ? 'bg-blue-50/50' : ''}`}>
                    <td className="px-5 py-3 whitespace-nowrap font-medium text-gray-900">{h.ano}</td>
                    <td className="px-5 py-3 whitespace-nowrap text-center">
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-medium ${
                        isLatest ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-600'
                      }`}>
                        #{h.ranking_brasil}
                      </span>
                    </td>
                    <td className={`px-5 py-3 whitespace-nowrap text-right font-semibold ${isLatest ? 'text-blue-600' : 'text-gray-900'}`}>
                      {formatNumber(h.nota_media)}
                    </td>
                    <td className="px-5 py-3 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_cn)}</td>
                    <td className="px-5 py-3 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_ch)}</td>
                    <td className="px-5 py-3 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_lc)}</td>
                    <td className="px-5 py-3 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_mt)}</td>
                    <td className="px-5 py-3 whitespace-nowrap text-right text-gray-600">{formatNumber(h.nota_redacao)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Skills Analysis Section */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-red-50">
                <AlertTriangle className="h-5 w-5 text-red-600" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-gray-900">Habilidades - Pontos de Atenção</h2>
                <p className="text-xs text-gray-500">Comparação com a média nacional (ENEM 2024)</p>
              </div>
            </div>
          </div>
        </div>

        {skillsLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : schoolSkills ? (
          <div className="p-5">
            {/* Area Filter Tabs */}
            <div className="flex flex-wrap gap-2 mb-5">
              <button
                onClick={() => setSelectedSkillArea(null)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  selectedSkillArea === null
                    ? 'bg-gray-900 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                Todas ({schoolSkills.worst_overall.length})
              </button>
              {(['CN', 'CH', 'LC', 'MT'] as const).map((area) => {
                const areaColors: Record<string, string> = {
                  CN: 'bg-green-100 text-green-700',
                  CH: 'bg-purple-100 text-purple-700',
                  LC: 'bg-pink-100 text-pink-700',
                  MT: 'bg-orange-100 text-orange-700',
                };
                const areaNames: Record<string, string> = {
                  CN: 'Natureza',
                  CH: 'Humanas',
                  LC: 'Linguagens',
                  MT: 'Matemática',
                };
                return (
                  <button
                    key={area}
                    onClick={() => setSelectedSkillArea(area)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                      selectedSkillArea === area
                        ? areaColors[area]
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {areaNames[area]}
                  </button>
                );
              })}
            </div>

            {/* Skills List */}
            <div className="space-y-3">
              {(selectedSkillArea
                ? schoolSkills.by_area[selectedSkillArea] || []
                : schoolSkills.worst_overall
              ).map((skill, index) => {
                const areaColors: Record<string, { bg: string; text: string; border: string }> = {
                  CN: { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200' },
                  CH: { bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200' },
                  LC: { bg: 'bg-pink-50', text: 'text-pink-700', border: 'border-pink-200' },
                  MT: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
                };
                const area = 'area' in skill ? skill.area : selectedSkillArea || 'CN';
                const colors = areaColors[area] || areaColors.CN;

                return (
                  <div
                    key={`${area}-${skill.skill_num}`}
                    className={`p-4 rounded-xl border ${colors.border} ${colors.bg} transition-all hover:shadow-sm`}
                  >
                    <div className="flex items-start gap-4">
                      {/* Rank */}
                      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                        index < 3 ? 'bg-red-100 text-red-700' : 'bg-gray-200 text-gray-600'
                      }`}>
                        {index + 1}
                      </div>

                      {/* Skill Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          {'area' in skill && (
                            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${colors.bg} ${colors.text}`}>
                              {skill.area}
                            </span>
                          )}
                          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${colors.bg} ${colors.text}`}>
                            H{skill.skill_num.toString().padStart(2, '0')}
                          </span>
                        </div>
                        <p className="text-sm text-gray-700">{skill.descricao}</p>
                      </div>

                      {/* Performance Comparison */}
                      <div className="flex-shrink-0 text-right">
                        <div className="flex items-center gap-3">
                          {/* School Performance */}
                          <div>
                            <div className={`text-lg font-bold ${
                              skill.performance < 30 ? 'text-red-600' :
                              skill.performance < 50 ? 'text-orange-600' :
                              'text-green-600'
                            }`}>
                              {skill.performance.toFixed(1)}%
                            </div>
                            <div className="text-xs text-gray-500">escola</div>
                          </div>

                          {/* Comparison Arrow */}
                          {skill.diff !== null && (
                            <div className={`flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium ${
                              skill.status === 'above' ? 'bg-green-100 text-green-700' :
                              skill.status === 'below' ? 'bg-red-100 text-red-700' :
                              'bg-gray-100 text-gray-600'
                            }`}>
                              {skill.status === 'above' ? (
                                <TrendingUp className="h-3 w-3" />
                              ) : skill.status === 'below' ? (
                                <TrendingDown className="h-3 w-3" />
                              ) : null}
                              {skill.diff > 0 ? '+' : ''}{skill.diff.toFixed(1)}
                            </div>
                          )}

                          {/* National Average */}
                          {skill.national_avg !== null && (
                            <div>
                              <div className="text-lg font-bold text-gray-400">
                                {skill.national_avg.toFixed(1)}%
                              </div>
                              <div className="text-xs text-gray-400">média BR</div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="mt-3 ml-12">
                      <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
                        {/* National average marker */}
                        {skill.national_avg !== null && (
                          <div
                            className="absolute top-0 bottom-0 w-0.5 bg-gray-500 z-10"
                            style={{ left: `${skill.national_avg}%` }}
                          />
                        )}
                        {/* School performance bar */}
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${skill.performance}%`,
                            backgroundColor: skill.performance < 30 ? '#dc2626' :
                                           skill.performance < 50 ? '#ea580c' :
                                           '#16a34a'
                          }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Legend */}
            <div className="mt-5 pt-4 border-t border-gray-100">
              <div className="flex flex-wrap gap-6 text-xs text-gray-500">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <span>Crítico (&lt;30%)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                  <span>Atenção (30-50%)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span>Bom (&gt;50%)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-0.5 h-3 bg-gray-500"></div>
                  <span>Média Nacional</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-5 text-center text-gray-500">
            Dados de habilidades não disponíveis para esta escola
          </div>
        )}
      </div>
    </div>
  );
}
