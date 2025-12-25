'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, TRIAreaAnalysis } from '@/lib/api';

interface TRIAnalysisProps {
  codigoInep: string;
}

function MasteryGauge({ value, label }: { value: number; label: string }) {
  const percentage = Math.round(value * 100);

  const getColor = (val: number) => {
    if (val >= 0.8) return '#22c55e';
    if (val >= 0.6) return '#84cc16';
    if (val >= 0.4) return '#eab308';
    if (val >= 0.2) return '#f97316';
    return '#ef4444';
  };

  const color = getColor(value);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          <circle
            cx="50" cy="50" r="40"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="12"
          />
          <circle
            cx="50" cy="50" r="40"
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${percentage * 2.51} 251`}
            className="transition-all duration-1000"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-bold" style={{ color }}>
            {percentage}%
          </span>
        </div>
      </div>
      <span className="text-xs text-gray-500 mt-1">{label}</span>
    </div>
  );
}

function AreaCard({ area, isExpanded, onToggle }: {
  area: TRIAreaAnalysis;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const changeColor = area.expected_change >= 0 ? 'text-green-600' : 'text-red-600';
  const masteryPercent = Math.round(area.tri_mastery_level * 100);

  return (
    <div
      className="bg-white rounded-xl border border-gray-200 overflow-hidden cursor-pointer transition-all hover:border-gray-300 hover:shadow-sm"
      onClick={onToggle}
    >
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: area.color }}
            />
            <span className="font-semibold text-gray-900">{area.area_name}</span>
            <span className="text-xs text-gray-400">({area.area})</span>
          </div>
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>

        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-gray-500 text-xs">Atual</div>
            <div className="text-xl font-bold text-gray-900">{area.current_score.toFixed(0)}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">Previsto</div>
            <div className="text-xl font-bold text-blue-600">{area.predicted_score.toFixed(0)}</div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">Mudança</div>
            <div className={`text-xl font-bold ${changeColor}`}>
              {area.expected_change >= 0 ? '+' : ''}{area.expected_change.toFixed(0)}
            </div>
          </div>
        </div>

        <div className="mt-4">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Domínio TRI</span>
            <span>{masteryPercent}%</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2">
            <div
              className="h-2 rounded-full transition-all duration-500"
              style={{
                width: `${masteryPercent}%`,
                backgroundColor: area.color
              }}
            />
          </div>
        </div>

        <div className="mt-3 flex gap-4 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-gray-500">Gap Mediana:</span>
            <span className={area.tri_gap_to_median >= 0 ? 'text-green-600' : 'text-red-600'}>
              {area.tri_gap_to_median >= 0 ? '+' : ''}{area.tri_gap_to_median.toFixed(0)}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-gray-500">Potencial:</span>
            <span className="text-blue-600">{(area.tri_potential * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-gray-100 p-4 bg-gray-50">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Accessible Content */}
            <div>
              <h4 className="text-sm font-semibold text-green-600 mb-2 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Conteúdo Dominado
              </h4>
              {area.accessible_content_sample.length > 0 ? (
                <ul className="space-y-2">
                  {area.accessible_content_sample.map((content, idx) => (
                    <li key={idx} className="text-xs bg-white p-2 rounded border border-gray-100">
                      <div className="flex justify-between items-start mb-1">
                        <span className="font-mono text-green-600 font-medium">{content.skill}</span>
                        <span className="text-gray-400">TRI: {content.tri_score.toFixed(0)}</span>
                      </div>
                      <p className="text-gray-600 line-clamp-2">{content.description}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-gray-400 italic">Sem dados disponíveis</p>
              )}
            </div>

            {/* Stretch Content */}
            <div>
              <h4 className="text-sm font-semibold text-amber-600 mb-2 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
                Próximos Desafios
              </h4>
              {area.stretch_content_sample.length > 0 ? (
                <ul className="space-y-2">
                  {area.stretch_content_sample.map((content, idx) => (
                    <li key={idx} className="text-xs bg-white p-2 rounded border border-gray-100">
                      <div className="flex justify-between items-start mb-1">
                        <span className="font-mono text-amber-600 font-medium">{content.skill}</span>
                        <span className="text-gray-400">
                          TRI: {content.tri_score.toFixed(0)}
                          {content.gap && <span className="text-red-500 ml-1">(+{content.gap.toFixed(0)})</span>}
                        </span>
                      </div>
                      <p className="text-gray-600 line-clamp-2">{content.description}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-gray-400 italic">Sem dados disponíveis</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function TRIAnalysis({ codigoInep }: TRIAnalysisProps) {
  const [expandedArea, setExpandedArea] = useState<string | null>(null);

  const { data, error, isLoading } = useQuery({
    queryKey: ['tri-analysis', codigoInep],
    queryFn: () => api.getTRIAnalysis(codigoInep),
  });

  if (isLoading) {
    return (
      <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 rounded-2xl shadow-sm border border-blue-100 p-6 animate-pulse">
        <div className="h-6 bg-blue-100 rounded w-1/3 mb-4" />
        <div className="h-32 bg-blue-50 rounded mb-4" />
        <div className="grid grid-cols-2 gap-4">
          <div className="h-48 bg-blue-50 rounded" />
          <div className="h-48 bg-blue-50 rounded" />
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-red-200 p-6">
        <p className="text-red-600">Erro ao carregar análise TRI</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 rounded-2xl shadow-sm border border-blue-100 overflow-hidden">
      <div className="px-5 py-4 border-b border-blue-100/50 bg-white/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h2 className="text-base font-semibold text-gray-900">Análise TRI</h2>
              <p className="text-xs text-gray-500">Predição baseada em Teoria de Resposta ao Item</p>
            </div>
          </div>
          <MasteryGauge value={data.overall_tri_mastery} label="Domínio Geral" />
        </div>
      </div>

      <div className="p-5 space-y-5">
        {/* Insights */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white rounded-xl p-4 border-l-4 border-blue-500 shadow-sm">
            <h3 className="text-sm font-semibold text-blue-600 mb-2">Interpretação</h3>
            <p className="text-sm text-gray-600">{data.insights.mastery_interpretation}</p>
          </div>
          <div className="bg-white rounded-xl p-4 border-l-4 border-green-500 shadow-sm">
            <h3 className="text-sm font-semibold text-green-600 mb-2">Recomendação</h3>
            <p className="text-sm text-gray-600">{data.insights.recommendation}</p>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4">
          {data.area_analysis.map((area) => (
            <div
              key={area.area}
              className="bg-white rounded-xl p-3 text-center cursor-pointer hover:shadow-md transition-all border border-gray-100"
              onClick={() => setExpandedArea(expandedArea === area.area ? null : area.area)}
            >
              <div
                className="w-10 h-10 rounded-full mx-auto mb-2 flex items-center justify-center text-white font-bold text-sm"
                style={{ backgroundColor: area.color }}
              >
                {area.area}
              </div>
              <div className="text-lg font-bold text-gray-900">{area.predicted_score.toFixed(0)}</div>
              <div className={`text-xs font-medium ${area.expected_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {area.expected_change >= 0 ? '+' : ''}{area.expected_change.toFixed(0)} pts
              </div>
            </div>
          ))}
        </div>

        {/* Area Details */}
        <div className="space-y-4">
          <h3 className="text-base font-semibold text-gray-900">Análise por Área</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {data.area_analysis.map((area) => (
              <AreaCard
                key={area.area}
                area={area}
                isExpanded={expandedArea === area.area}
                onToggle={() => setExpandedArea(expandedArea === area.area ? null : area.area)}
              />
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="text-xs text-gray-400 text-center pt-4 border-t border-gray-100">
          Análise baseada em {data.area_analysis.reduce((acc, a) =>
            acc + a.accessible_content_sample.length + a.stretch_content_sample.length, 0
          )} itens de conteúdo TRI
        </div>
      </div>
    </div>
  );
}
