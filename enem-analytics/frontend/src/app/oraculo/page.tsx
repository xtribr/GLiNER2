'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_BASE } from '@/lib/api';
import Link from 'next/link';
import {
  Sparkles,
  TrendingUp,
  BookOpen,
  Brain,
  Calculator,
  Globe,
  ChevronDown,
  ChevronUp,
  Info,
  Filter
} from 'lucide-react';

interface Prediction {
  rank: number;
  area: string;
  tema: string;
  conceitos: string[];
  habilidades: string[];
  probabilidade: number;
  tipo: string;
  justificativa: string;
}

interface OracleResponse {
  total: number;
  ano_predicao: number;
  gerado_em: string;
  predicoes: Prediction[];
}

const AREA_ICONS: Record<string, React.ReactNode> = {
  'Linguagens': <BookOpen className="h-5 w-5" />,
  'Ciências Humanas': <Globe className="h-5 w-5" />,
  'Ciências da Natureza': <Brain className="h-5 w-5" />,
  'Matemática': <Calculator className="h-5 w-5" />
};

const AREA_COLORS: Record<string, string> = {
  'Linguagens': 'bg-purple-100 text-purple-800 border-purple-200',
  'Ciências Humanas': 'bg-amber-100 text-amber-800 border-amber-200',
  'Ciências da Natureza': 'bg-green-100 text-green-800 border-green-200',
  'Matemática': 'bg-blue-100 text-blue-800 border-blue-200'
};

const TIPO_COLORS: Record<string, string> = {
  'Tendência 2025': 'bg-red-100 text-red-700',
  'Recorrente': 'bg-gray-100 text-gray-700'
};

export default function OraculoPage() {
  const [selectedArea, setSelectedArea] = useState<string>('');
  const [selectedTipo, setSelectedTipo] = useState<string>('');
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  const { data, isLoading, error } = useQuery<OracleResponse>({
    queryKey: ['oracle-predictions', selectedArea, selectedTipo],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (selectedArea) params.set('area', selectedArea);
      if (selectedTipo) params.set('tipo', selectedTipo);

      const response = await fetch(`${API_BASE}/api/oracle/predictions?${params}`);
      if (!response.ok) throw new Error('Falha ao carregar predições');
      return response.json();
    }
  });

  const getProbabilityColor = (prob: number) => {
    if (prob >= 0.6) return 'text-green-600 bg-green-50';
    if (prob >= 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-gray-600 bg-gray-50';
  };

  const getProbabilityBar = (prob: number) => {
    const width = Math.round(prob * 100);
    let color = 'bg-gray-400';
    if (prob >= 0.6) color = 'bg-green-500';
    else if (prob >= 0.4) color = 'bg-yellow-500';
    return (
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div className={`${color} h-2 rounded-full transition-all`} style={{ width: `${width}%` }} />
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 sticky top-0 z-20">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Oráculo ENEM 2026</h1>
                <p className="text-sm text-slate-500">Predições baseadas em análise de 3.099 questões (2009-2025)</p>
              </div>
            </div>
            <Link
              href="/"
              className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
            >
              ← Voltar
            </Link>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6 max-w-7xl mx-auto">
        {/* Info Card */}
        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-6 text-white shadow-xl">
          <div className="flex items-start gap-4">
            <Info className="h-6 w-6 flex-shrink-0 mt-1" />
            <div>
              <h2 className="font-semibold text-lg mb-2">Como funciona o Oráculo?</h2>
              <p className="text-indigo-100 text-sm">
                O Oráculo analisa padrões históricos de 16 anos de provas do ENEM, identifica tendências
                recorrentes e combina com eventos atuais para prever os temas mais prováveis do ENEM 2026.
                As predições são rankeadas por probabilidade estimada.
              </p>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Filter className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-700">Filtros</span>
          </div>
          <div className="flex flex-wrap gap-4">
            <select
              value={selectedArea}
              onChange={(e) => setSelectedArea(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none bg-white text-sm"
            >
              <option value="">Todas as Áreas</option>
              <option value="Linguagens">Linguagens</option>
              <option value="Ciências Humanas">Ciências Humanas</option>
              <option value="Ciências da Natureza">Ciências da Natureza</option>
              <option value="Matemática">Matemática</option>
            </select>

            <select
              value={selectedTipo}
              onChange={(e) => setSelectedTipo(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none bg-white text-sm"
            >
              <option value="">Todos os Tipos</option>
              <option value="Tendência">Tendência 2025</option>
              <option value="Recorrente">Recorrente</option>
            </select>

            {data && (
              <div className="ml-auto flex items-center gap-2 text-sm text-gray-500">
                <TrendingUp className="h-4 w-4" />
                <span>{data.total} predições</span>
              </div>
            )}
          </div>
        </div>

        {/* Predictions Table */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
            </div>
          ) : error ? (
            <div className="text-center py-20 text-red-500">
              Erro ao carregar predições. Tente novamente.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider w-16">
                      Rank
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider w-40">
                      Área
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                      Tema
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider w-32">
                      Tipo
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider w-40">
                      Probabilidade
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-slate-600 uppercase tracking-wider w-20">
                      Detalhes
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {data?.predicoes.map((pred) => (
                    <>
                      <tr
                        key={pred.rank}
                        className="hover:bg-slate-50 transition-colors cursor-pointer"
                        onClick={() => setExpandedRow(expandedRow === pred.rank ? null : pred.rank)}
                      >
                        <td className="px-4 py-4">
                          <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full font-bold text-sm ${
                            pred.rank <= 3 ? 'bg-yellow-100 text-yellow-700' :
                            pred.rank <= 10 ? 'bg-indigo-100 text-indigo-700' :
                            'bg-gray-100 text-gray-600'
                          }`}>
                            {pred.rank}
                          </span>
                        </td>
                        <td className="px-4 py-4">
                          <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${AREA_COLORS[pred.area] || 'bg-gray-100 text-gray-800'}`}>
                            {AREA_ICONS[pred.area]}
                            {pred.area.split(' ')[0]}
                          </span>
                        </td>
                        <td className="px-4 py-4">
                          <span className="font-medium text-slate-900">{pred.tema}</span>
                        </td>
                        <td className="px-4 py-4">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${TIPO_COLORS[pred.tipo] || 'bg-gray-100 text-gray-700'}`}>
                            {pred.tipo}
                          </span>
                        </td>
                        <td className="px-4 py-4">
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <span className={`text-sm font-semibold px-2 py-0.5 rounded ${getProbabilityColor(pred.probabilidade)}`}>
                                {(pred.probabilidade * 100).toFixed(0)}%
                              </span>
                            </div>
                            {getProbabilityBar(pred.probabilidade)}
                          </div>
                        </td>
                        <td className="px-4 py-4 text-center">
                          <button className="p-1 hover:bg-slate-100 rounded-lg transition-colors">
                            {expandedRow === pred.rank ? (
                              <ChevronUp className="h-5 w-5 text-slate-400" />
                            ) : (
                              <ChevronDown className="h-5 w-5 text-slate-400" />
                            )}
                          </button>
                        </td>
                      </tr>
                      {expandedRow === pred.rank && (
                        <tr className="bg-slate-50">
                          <td colSpan={6} className="px-4 py-4">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                              <div>
                                <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">Conceitos Relacionados</h4>
                                <div className="flex flex-wrap gap-1">
                                  {pred.conceitos.map((c, i) => (
                                    <span key={i} className="px-2 py-1 bg-white border border-slate-200 rounded text-xs text-slate-700">
                                      {c}
                                    </span>
                                  ))}
                                </div>
                              </div>
                              <div>
                                <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">Habilidades Prováveis</h4>
                                <div className="flex flex-wrap gap-1">
                                  {pred.habilidades.map((h, i) => (
                                    <span key={i} className="px-2 py-1 bg-indigo-50 border border-indigo-200 rounded text-xs text-indigo-700 font-medium">
                                      {h}
                                    </span>
                                  ))}
                                </div>
                              </div>
                              <div>
                                <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">Justificativa</h4>
                                <p className="text-xs text-slate-600">{pred.justificativa}</p>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="text-center text-xs text-slate-400 py-4">
          <p>Predições geradas em {data?.gerado_em} | Baseado em análise de padrões históricos do ENEM</p>
          <p className="mt-1">As probabilidades são estimativas e não garantem o conteúdo da prova oficial.</p>
        </div>
      </div>
    </div>
  );
}
