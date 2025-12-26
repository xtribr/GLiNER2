'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import {
  Brain,
  Sparkles,
  TrendingUp,
  BookOpen,
  Target,
  Zap,
  ChevronRight,
  Info,
  Network,
  Lightbulb,
  ArrowUpRight,
} from 'lucide-react';

interface BrainXInsightsProps {
  codigoInep: string;
}

export function BrainXInsights({ codigoInep }: BrainXInsightsProps) {
  const [selectedArea, setSelectedArea] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'concepts' | 'study' | 'network'>('concepts');

  const { data: conceptAnalysis, isLoading: conceptsLoading } = useQuery({
    queryKey: ['glinerConcepts', codigoInep],
    queryFn: () => api.getGlinerConceptAnalysis(codigoInep, 15),
  });

  const { data: studyFocus, isLoading: studyLoading } = useQuery({
    queryKey: ['glinerStudyFocus', codigoInep],
    queryFn: () => api.getGlinerStudyFocus(codigoInep),
  });

  const isLoading = conceptsLoading || studyLoading;

  if (isLoading) {
    return (
      <div className="bg-gradient-to-br from-purple-50 via-indigo-50 to-blue-50 rounded-2xl p-6 border border-purple-100">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 animate-pulse">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="h-5 w-40 bg-purple-200 rounded animate-pulse" />
            <div className="h-3 w-60 bg-purple-100 rounded mt-1 animate-pulse" />
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-40 bg-white/50 rounded-xl animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  const areaData = selectedArea
    ? conceptAnalysis?.area_analyses.find((a) => a.area === selectedArea)
    : null;

  return (
    <div className="bg-gradient-to-br from-purple-50 via-indigo-50 to-blue-50 rounded-2xl shadow-sm border border-purple-100 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-purple-100/50 bg-white/60 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600">
              <Brain className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-gray-900">Inteligência BrainX</h2>
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded-full font-medium flex items-center gap-1">
                  <Sparkles className="h-3 w-3" /> NER
                </span>
              </div>
              <p className="text-xs text-gray-500">
                {conceptAnalysis?.summary.total_unique_concepts} conceitos |{' '}
                {conceptAnalysis?.summary.total_semantic_fields} campos semânticos |{' '}
                {conceptAnalysis?.summary.total_lexical_fields} campos lexicais
              </p>
            </div>
          </div>

          {/* Tab Buttons */}
          <div className="flex items-center gap-1 bg-white/80 rounded-lg p-1">
            <button
              onClick={() => setActiveTab('concepts')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeTab === 'concepts'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Conceitos
            </button>
            <button
              onClick={() => setActiveTab('study')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeTab === 'study'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Foco de Estudo
            </button>
            <button
              onClick={() => setActiveTab('network')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeTab === 'network'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Rede
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-5">
        {activeTab === 'concepts' && (
          <ConceptsTab
            conceptAnalysis={conceptAnalysis}
            selectedArea={selectedArea}
            setSelectedArea={setSelectedArea}
            areaData={areaData}
          />
        )}

        {activeTab === 'study' && studyFocus && (
          <StudyFocusTab studyFocus={studyFocus} />
        )}

        {activeTab === 'network' && (
          <NetworkTab codigoInep={codigoInep} conceptAnalysis={conceptAnalysis} />
        )}
      </div>
    </div>
  );
}

// Concepts Tab Component
function ConceptsTab({
  conceptAnalysis,
  selectedArea,
  setSelectedArea,
  areaData,
}: {
  conceptAnalysis: Awaited<ReturnType<typeof api.getGlinerConceptAnalysis>> | undefined;
  selectedArea: string | null;
  setSelectedArea: (area: string | null) => void;
  areaData: Awaited<ReturnType<typeof api.getGlinerConceptAnalysis>>['area_analyses'][0] | null | undefined;
}) {
  if (!conceptAnalysis) return null;

  return (
    <div className="space-y-5">
      {/* Area Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {conceptAnalysis.area_analyses.map((area) => {
          const isSelected = selectedArea === area.area;
          return (
            <button
              key={area.area}
              onClick={() => setSelectedArea(isSelected ? null : area.area)}
              className={`text-left p-4 rounded-xl border transition-all ${
                isSelected
                  ? 'bg-white shadow-md border-purple-300 ring-2 ring-purple-200'
                  : 'bg-white/70 border-gray-100 hover:bg-white hover:shadow-sm'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div
                  className="h-8 w-8 rounded-lg flex items-center justify-center text-white text-xs font-bold"
                  style={{ backgroundColor: area.color }}
                >
                  {area.area}
                </div>
                <span className="text-lg font-bold" style={{ color: area.color }}>
                  {area.predicted_score}
                </span>
              </div>
              <h4 className="text-sm font-medium text-gray-900 mb-1">{area.area_name}</h4>
              <div className="flex items-center gap-3 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <Lightbulb className="h-3 w-3" />
                  {area.unique_concepts}
                </span>
                <span className="flex items-center gap-1">
                  <BookOpen className="h-3 w-3" />
                  {area.unique_semantic_fields}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Area Details */}
      {areaData && (
        <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div
              className="h-10 w-10 rounded-xl flex items-center justify-center text-white font-bold"
              style={{ backgroundColor: areaData.color }}
            >
              {areaData.area}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">{areaData.area_name}</h3>
              <p className="text-xs text-gray-500">
                {areaData.total_content_items} itens de conteúdo analisados
              </p>
            </div>
          </div>

          {/* Top Concepts */}
          <div className="mb-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-amber-500" />
              Conceitos Prioritários
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
              {areaData.top_concepts.slice(0, 9).map((concept, idx) => (
                <div
                  key={concept.concept}
                  className={`p-3 rounded-lg border transition-all ${
                    concept.importance === 'high'
                      ? 'bg-amber-50 border-amber-200'
                      : concept.importance === 'medium'
                      ? 'bg-blue-50 border-blue-200'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                >
                  <div className="flex items-start justify-between mb-1">
                    <span className="text-xs font-medium text-gray-900 line-clamp-1">
                      {concept.concept}
                    </span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                        concept.importance === 'high'
                          ? 'bg-amber-100 text-amber-700'
                          : concept.importance === 'medium'
                          ? 'bg-blue-100 text-blue-700'
                          : 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {concept.frequency.toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(concept.confidence * 100, 100)}%`,
                          backgroundColor: areaData.color,
                        }}
                      />
                    </div>
                    <span className="text-[10px] text-gray-500">
                      {(concept.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {(concept.semantic_fields?.length > 0 || concept.lexical_fields?.length > 0) && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {concept.semantic_fields?.slice(0, 2).map((field: string) => (
                        <span
                          key={field}
                          className="text-[10px] px-1.5 py-0.5 bg-purple-100 text-purple-600 rounded"
                        >
                          {field}
                        </span>
                      ))}
                      {concept.lexical_fields?.slice(0, 1).map((field: string) => (
                        <span
                          key={field}
                          className="text-[10px] px-1.5 py-0.5 bg-green-100 text-green-600 rounded"
                        >
                          {field}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Semantic Fields & Historical Contexts */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                <Brain className="h-4 w-4 text-purple-500" />
                Campos Semânticos
              </h4>
              <div className="space-y-1">
                {areaData.semantic_fields?.slice(0, 5).map((field: { field: string; count: number }) => (
                  <div key={field.field} className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">{field.field}</span>
                    <span className="font-medium text-gray-900">{field.count}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                <Target className="h-4 w-4 text-green-500" />
                Contextos Históricos
              </h4>
              <div className="space-y-1">
                {areaData.historical_contexts?.slice(0, 5).map((ctx: { context: string; count: number }) => (
                  <div key={ctx.context} className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">{ctx.context}</span>
                    <span className="font-medium text-gray-900">{ctx.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Priority Concepts Summary */}
      {!selectedArea && (
        <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Zap className="h-4 w-4 text-amber-500" />
            Conceitos Mais Relevantes (Todas as Áreas)
          </h3>
          <div className="flex flex-wrap gap-2">
            {conceptAnalysis.priority_concepts.slice(0, 20).map((concept, idx) => (
              <span
                key={concept.concept}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all hover:scale-105"
                style={{
                  backgroundColor:
                    conceptAnalysis.area_analyses.find((a) => a.area === concept.area)?.color +
                    '20',
                  color: conceptAnalysis.area_analyses.find((a) => a.area === concept.area)?.color,
                }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{
                    backgroundColor: conceptAnalysis.area_analyses.find(
                      (a) => a.area === concept.area
                    )?.color,
                  }}
                />
                {concept.concept}
                <span className="opacity-60">{concept.frequency.toFixed(0)}%</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Study Focus Tab Component
function StudyFocusTab({
  studyFocus,
}: {
  studyFocus: Awaited<ReturnType<typeof api.getGlinerStudyFocus>>;
}) {
  return (
    <div className="space-y-5">
      {/* Overall Improvement Banner */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl p-4 text-white">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm opacity-90">Potencial de Melhoria Estimado</p>
            <p className="text-3xl font-bold">+{studyFocus.total_estimated_improvement} pts</p>
          </div>
          <div className="text-right">
            <p className="text-sm opacity-90">Áreas em Foco</p>
            <p className="text-2xl font-bold">{studyFocus.focus_areas.length}</p>
          </div>
        </div>
      </div>

      {/* Study Plan Phases */}
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(studyFocus.study_plan).map(([key, phase], idx) => (
          <div
            key={key}
            className={`p-4 rounded-xl border ${
              idx === 0
                ? 'bg-blue-50 border-blue-200'
                : idx === 1
                ? 'bg-purple-50 border-purple-200'
                : 'bg-amber-50 border-amber-200'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <div
                className={`h-6 w-6 rounded-full flex items-center justify-center text-white text-xs font-bold ${
                  idx === 0 ? 'bg-blue-500' : idx === 1 ? 'bg-purple-500' : 'bg-amber-500'
                }`}
              >
                {idx + 1}
              </div>
              <span className="text-sm font-semibold text-gray-900">{phase.name}</span>
            </div>
            <p className="text-xs text-gray-600 mb-2">{phase.description}</p>
            <p className="text-lg font-bold text-gray-900">{phase.concepts_count} conceitos</p>
          </div>
        ))}
      </div>

      {/* Focus Areas Detail */}
      <div className="space-y-4">
        {studyFocus.focus_areas.map((area) => (
          <div key={area.area} className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div
                  className="h-10 w-10 rounded-xl flex items-center justify-center text-white font-bold"
                  style={{ backgroundColor: area.color }}
                >
                  {area.area}
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{area.area_name}</h3>
                  <p className="text-xs text-gray-500">
                    Nível: <span className="font-medium capitalize">{area.level}</span> | Score:{' '}
                    {area.current_score}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-500">Impacto Estimado</p>
                <p className="text-xl font-bold text-green-600">
                  +{area.estimated_total_impact} pts
                </p>
              </div>
            </div>

            {/* Study Sequence */}
            <div className="space-y-2">
              <p className="text-xs font-medium text-gray-500 mb-2">
                Sequência de Estudo ({area.study_sequence.length} conceitos)
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {area.study_sequence.slice(0, 9).map((concept, idx) => (
                  <div
                    key={concept.concept}
                    className={`p-3 rounded-lg border ${
                      concept.priority === 'high'
                        ? 'bg-gradient-to-r from-amber-50 to-orange-50 border-amber-200'
                        : concept.priority === 'medium'
                        ? 'bg-blue-50 border-blue-200'
                        : 'bg-gray-50 border-gray-200'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white ${
                            concept.priority === 'high'
                              ? 'bg-amber-500'
                              : concept.priority === 'medium'
                              ? 'bg-blue-500'
                              : 'bg-gray-400'
                          }`}
                        >
                          {idx + 1}
                        </span>
                        <span className="text-xs font-medium text-gray-900 line-clamp-1">
                          {concept.concept}
                        </span>
                      </div>
                      <span className="text-[10px] text-green-600 font-medium">
                        +{concept.estimated_impact}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-[10px] text-gray-500 ml-7">
                      <span>TRI: {concept.avg_difficulty}</span>
                      {concept.semantic_fields?.[0] && (
                        <span className="px-1.5 py-0.5 bg-purple-100 text-purple-600 rounded">
                          {concept.semantic_fields[0]}
                        </span>
                      )}
                      {concept.lexical_fields?.[0] && (
                        <span className="px-1.5 py-0.5 bg-green-100 text-green-600 rounded">
                          {concept.lexical_fields[0]}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Network Tab Component
function NetworkTab({
  codigoInep,
  conceptAnalysis,
}: {
  codigoInep: string;
  conceptAnalysis: Awaited<ReturnType<typeof api.getGlinerConceptAnalysis>> | undefined;
}) {
  const [networkArea, setNetworkArea] = useState<string | undefined>(undefined);

  const { data: graphData, isLoading } = useQuery({
    queryKey: ['glinerGraph', codigoInep, networkArea],
    queryFn: () => api.getGlinerKnowledgeGraph(codigoInep, networkArea),
  });

  return (
    <div className="space-y-4">
      {/* Area Filter */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500">Filtrar por área:</span>
        <div className="flex gap-1">
          <button
            onClick={() => setNetworkArea(undefined)}
            className={`px-3 py-1 text-xs rounded-lg transition-all ${
              !networkArea
                ? 'bg-purple-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Todas
          </button>
          {conceptAnalysis?.area_analyses.map((area) => (
            <button
              key={area.area}
              onClick={() => setNetworkArea(area.area)}
              className={`px-3 py-1 text-xs rounded-lg transition-all ${
                networkArea === area.area
                  ? 'text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
              style={
                networkArea === area.area ? { backgroundColor: area.color } : undefined
              }
            >
              {area.area}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="h-80 bg-white rounded-xl flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600" />
        </div>
      ) : graphData ? (
        <div className="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Network className="h-5 w-5 text-purple-600" />
              <h3 className="font-semibold text-gray-900">Rede de Conhecimento</h3>
            </div>
            <div className="flex items-center gap-4 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-blue-500" />
                {graphData.stats.concept_nodes} conceitos
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-purple-500" />
                {graphData.stats.semantic_nodes} c. semânticos
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                {graphData.stats.lexical_nodes} c. lexicais
              </span>
              <span>{graphData.stats.total_edges} conexões</span>
            </div>
          </div>

          {/* Visual Network Representation */}
          <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl p-4">
            {/* Nodes Grid */}
            <div className="flex flex-wrap gap-2 justify-center">
              {graphData.nodes.slice(0, 40).map((node) => (
                <div
                  key={node.id}
                  className="group relative"
                >
                  <div
                    className="px-3 py-1.5 rounded-full text-white text-xs font-medium shadow-sm cursor-pointer transition-all hover:scale-105 hover:shadow-md"
                    style={{
                      backgroundColor: node.color,
                    }}
                  >
                    {node.label.length > 15 ? node.label.slice(0, 15) + '...' : node.label}
                  </div>
                  {/* Tooltip */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-20 shadow-lg">
                    <p className="font-semibold">{node.label}</p>
                    <p className="text-gray-300">
                      {node.type === 'conceito_cientifico' ? 'Conceito' :
                       node.type === 'campo_semantico' ? 'Campo Semântico' : 'Campo Lexical'}
                    </p>
                    <p className="text-gray-300">Frequência: {node.count}x</p>
                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
                  </div>
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-6 mt-4 pt-3 border-t border-slate-200">
              <span className="flex items-center gap-1.5 text-xs text-gray-600">
                <div className="w-3 h-3 rounded-full bg-blue-500" /> Conceito
              </span>
              <span className="flex items-center gap-1.5 text-xs text-gray-600">
                <div className="w-3 h-3 rounded-full bg-purple-500" /> Semântico
              </span>
              <span className="flex items-center gap-1.5 text-xs text-gray-600">
                <div className="w-3 h-3 rounded-full bg-green-500" /> Lexical
              </span>
            </div>
          </div>

          {/* Top Connections */}
          <div className="mt-4 pt-4 border-t border-gray-100">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Principais Conexões</h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
              {graphData.edges.slice(0, 8).map((edge, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs text-gray-600 p-2 bg-gray-50 rounded-lg">
                  <span className="font-medium truncate">
                    {edge.source.replace('concept_', '').replace('semantic_', '').replace('lexical_', '')}
                  </span>
                  <ArrowUpRight className="h-3 w-3 text-gray-400 flex-shrink-0" />
                  <span className="truncate">
                    {edge.target.replace('concept_', '').replace('semantic_', '').replace('lexical_', '')}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
