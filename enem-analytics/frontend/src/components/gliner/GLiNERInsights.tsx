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

// Network Tab Component with Enhanced Neural Network Visualization
function NetworkTab({
  codigoInep,
  conceptAnalysis,
}: {
  codigoInep: string;
  conceptAnalysis: Awaited<ReturnType<typeof api.getGlinerConceptAnalysis>> | undefined;
}) {
  const [networkArea, setNetworkArea] = useState<string | undefined>(undefined);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'all' | 'semantic' | 'lexical' | 'concepts'>('all');

  const { data: graphData, isLoading } = useQuery({
    queryKey: ['glinerGraph', codigoInep, networkArea],
    queryFn: () => api.getGlinerKnowledgeGraph(codigoInep, networkArea),
  });

  // Enhanced node positioning with spiral/cluster layout
  type GraphNode = { id: string; label: string; type: string; color: string; size: number; count: number };
  const calculatePositions = (nodes: GraphNode[]) => {
    if (!nodes) return {};

    // Separate nodes by type first
    const allSemanticNodes = nodes.filter(n => n.type === 'campo_semantico');
    const allLexicalNodes = nodes.filter(n => n.type === 'campo_lexical');
    const allConceptNodes = nodes.filter(n => n.type === 'conceito_cientifico');

    const positions: { [key: string]: { x: number; y: number; node: GraphNode; ring: number; emphasis: boolean } } = {};

    // Center point adjusted to keep nodes in view (shifted down to prevent top overflow)
    const centerX = 50;
    const centerY = 55;

    // Determine layout based on view mode
    if (viewMode === 'semantic') {
      // Semantic-focused: Large grid layout for semantic fields
      const count = Math.min(allSemanticNodes.length, 16);
      allSemanticNodes.slice(0, count).forEach((node, i) => {
        const cols = 4;
        const row = Math.floor(i / cols);
        const col = i % cols;
        const spacing = 20;
        const startX = centerX - ((cols - 1) * spacing) / 2;
        const startY = 25;
        positions[node.id] = {
          x: startX + col * spacing,
          y: startY + row * spacing,
          node,
          ring: 1,
          emphasis: true
        };
      });
    } else if (viewMode === 'lexical') {
      // Lexical-focused: Hexagonal layout
      const count = Math.min(allLexicalNodes.length, 19);
      allLexicalNodes.slice(0, count).forEach((node, i) => {
        if (i === 0) {
          positions[node.id] = { x: centerX, y: centerY, node, ring: 0, emphasis: true };
        } else if (i <= 6) {
          const angle = ((i - 1) / 6) * Math.PI * 2 - Math.PI / 2;
          positions[node.id] = { x: centerX + Math.cos(angle) * 18, y: centerY + Math.sin(angle) * 18, node, ring: 1, emphasis: true };
        } else {
          const angle = ((i - 7) / 12) * Math.PI * 2 - Math.PI / 6;
          positions[node.id] = { x: centerX + Math.cos(angle) * 35, y: centerY + Math.sin(angle) * 35, node, ring: 2, emphasis: true };
        }
      });
    } else if (viewMode === 'concepts') {
      // Concepts-focused: Spiral layout for many concepts
      const count = Math.min(allConceptNodes.length, 36);
      allConceptNodes.slice(0, count).forEach((node, i) => {
        const spiralRadius = 10 + (i * 1.1);
        const angle = (i * 0.5) - Math.PI / 2;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * Math.min(spiralRadius, 38),
          y: centerY + Math.sin(angle) * Math.min(spiralRadius, 38),
          node,
          ring: Math.floor(i / 12) + 1,
          emphasis: true
        };
      });
    } else {
      // All mode: Hierarchical rings with semantic fields as the core

      // Core ring: Semantic fields (inner, larger, more prominent)
      const semanticCount = Math.min(allSemanticNodes.length, 8);
      allSemanticNodes.slice(0, semanticCount).forEach((node, i) => {
        const angle = (i / semanticCount) * Math.PI * 2 - Math.PI / 2;
        const radius = 12;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          node,
          ring: 1,
          emphasis: true
        };
      });

      // Middle ring: Lexical fields
      const lexicalCount = Math.min(allLexicalNodes.length, 14);
      allLexicalNodes.slice(0, lexicalCount).forEach((node, i) => {
        const angle = (i / lexicalCount) * Math.PI * 2 - Math.PI / 6;
        const radius = 24;
        const jitter = (i % 2) * 1;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * (radius + jitter),
          y: centerY + Math.sin(angle) * (radius + jitter),
          node,
          ring: 2,
          emphasis: false
        };
      });

      // Outer ring: Concepts (smaller, more numerous)
      const conceptCount = Math.min(allConceptNodes.length, 20);
      allConceptNodes.slice(0, conceptCount).forEach((node, i) => {
        const angle = (i / conceptCount) * Math.PI * 2;
        const radius = 35;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          node,
          ring: 3,
          emphasis: false
        };
      });
    }

    return positions;
  };

  const nodePositions = graphData ? calculatePositions(graphData.nodes) : {};

  // Get connected nodes for highlighting
  const getConnectedNodes = (nodeId: string) => {
    if (!graphData) return new Set<string>();
    const connected = new Set<string>();
    graphData.edges.forEach(edge => {
      if (edge.source === nodeId) connected.add(edge.target);
      if (edge.target === nodeId) connected.add(edge.source);
    });
    return connected;
  };

  const connectedNodes = hoveredNode ? getConnectedNodes(hoveredNode) : new Set<string>();
  const selectedConnections = selectedNode ? getConnectedNodes(selectedNode) : new Set<string>();

  return (
    <div className="space-y-4">
      {/* Filters Row */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        {/* Area Filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Área:</span>
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

        {/* View Mode Filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Visualizar:</span>
          <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5">
            <button
              onClick={() => setViewMode('all')}
              className={`px-3 py-1 text-xs rounded-md transition-all ${
                viewMode === 'all'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Todos
            </button>
            <button
              onClick={() => setViewMode('semantic')}
              className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1 ${
                viewMode === 'semantic'
                  ? 'bg-purple-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="w-2 h-2 rounded-full bg-purple-500" />
              Semânticos
            </button>
            <button
              onClick={() => setViewMode('lexical')}
              className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1 ${
                viewMode === 'lexical'
                  ? 'bg-emerald-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              Lexicais
            </button>
            <button
              onClick={() => setViewMode('concepts')}
              className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1 ${
                viewMode === 'concepts'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              Conceitos
            </button>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="h-[500px] bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl flex items-center justify-center">
          <div className="text-center">
            <div className="relative w-16 h-16 mx-auto mb-4">
              <div className="absolute inset-0 rounded-full border-4 border-purple-500/30 animate-ping" />
              <div className="absolute inset-2 rounded-full border-4 border-purple-400/50 animate-pulse" />
              <div className="absolute inset-4 rounded-full bg-purple-500 animate-pulse" />
            </div>
            <p className="text-purple-300 text-sm">Carregando rede neural...</p>
          </div>
        </div>
      ) : graphData ? (
        <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900 rounded-2xl overflow-hidden shadow-2xl">
          {/* Header Stats */}
          <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-purple-500 rounded-xl blur-lg opacity-50" />
                <div className="relative p-2 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl">
                  <Network className="h-5 w-5 text-white" />
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-white">Rede Neural de Conhecimento</h3>
                <p className="text-xs text-slate-400">Visualização interativa de conexões conceituais</p>
              </div>
            </div>
            <div className="flex items-center gap-6 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500 shadow-lg shadow-blue-500/50" />
                <span className="text-slate-300">{graphData.stats.concept_nodes} conceitos</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-purple-500 shadow-lg shadow-purple-500/50" />
                <span className="text-slate-300">{graphData.stats.semantic_nodes} semânticos</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/50" />
                <span className="text-slate-300">{graphData.stats.lexical_nodes} lexicais</span>
              </div>
              <div className="px-3 py-1 bg-white/10 rounded-full">
                <span className="text-slate-300">{graphData.stats.total_edges} conexões</span>
              </div>
            </div>
          </div>

          {/* Neural Network Visualization */}
          <div className="relative h-[450px] overflow-hidden">
            {/* Background Grid Effect */}
            <div className="absolute inset-0 opacity-10">
              <div className="absolute inset-0" style={{
                backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.3) 1px, transparent 0)`,
                backgroundSize: '40px 40px'
              }} />
            </div>

            {/* Glow Effects */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-purple-500/20 rounded-full blur-3xl" />
            <div className="absolute top-1/4 left-1/4 w-48 h-48 bg-blue-500/10 rounded-full blur-3xl" />
            <div className="absolute bottom-1/4 right-1/4 w-48 h-48 bg-emerald-500/10 rounded-full blur-3xl" />

            {/* SVG for connections */}
            <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 1 }}>
              <defs>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="rgba(139, 92, 246, 0.6)" />
                  <stop offset="100%" stopColor="rgba(59, 130, 246, 0.6)" />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>

              {/* Draw edges */}
              {graphData.edges.slice(0, 50).map((edge, idx) => {
                const sourcePos = nodePositions[edge.source];
                const targetPos = nodePositions[edge.target];
                if (!sourcePos || !targetPos) return null;

                const isHighlighted = hoveredNode === edge.source || hoveredNode === edge.target ||
                                     selectedNode === edge.source || selectedNode === edge.target;

                return (
                  <line
                    key={idx}
                    x1={`${sourcePos.x}%`}
                    y1={`${sourcePos.y}%`}
                    x2={`${targetPos.x}%`}
                    y2={`${targetPos.y}%`}
                    stroke={isHighlighted ? 'rgba(168, 85, 247, 0.8)' : 'rgba(148, 163, 184, 0.15)'}
                    strokeWidth={isHighlighted ? 2 : 1}
                    filter={isHighlighted ? 'url(#glow)' : undefined}
                    className="transition-all duration-300"
                  />
                );
              })}

              {/* Animated pulse rings on hovered node */}
              {hoveredNode && nodePositions[hoveredNode] && (
                <>
                  <circle
                    cx={`${nodePositions[hoveredNode].x}%`}
                    cy={`${nodePositions[hoveredNode].y}%`}
                    r="30"
                    fill="none"
                    stroke="rgba(168, 85, 247, 0.4)"
                    strokeWidth="2"
                    className="animate-ping"
                  />
                  <circle
                    cx={`${nodePositions[hoveredNode].x}%`}
                    cy={`${nodePositions[hoveredNode].y}%`}
                    r="20"
                    fill="none"
                    stroke="rgba(168, 85, 247, 0.6)"
                    strokeWidth="1"
                    className="animate-pulse"
                  />
                </>
              )}
            </svg>

            {/* Nodes */}
            {Object.entries(nodePositions).map(([id, { x, y, node, emphasis }]) => {
              const isHovered = hoveredNode === id;
              const isConnected = connectedNodes.has(id) || selectedConnections.has(id);
              const isSelected = selectedNode === id;
              const isDimmed = (hoveredNode || selectedNode) && !isHovered && !isConnected && !isSelected;

              // Semantic fields are more prominent in focused views
              const isSemantic = node.type === 'campo_semantico';
              const isLexical = node.type === 'campo_lexical';

              // Dynamic sizing based on view mode and type
              let baseSize = 40;
              if (viewMode === 'semantic' && isSemantic) {
                baseSize = 72;
              } else if (viewMode === 'lexical' && isLexical) {
                baseSize = 64;
              } else if (viewMode === 'concepts') {
                baseSize = 44;
              } else if (isSemantic) {
                baseSize = 60;
              } else if (isLexical) {
                baseSize = 50;
              }

              const size = isHovered || isSelected ? baseSize + 10 : baseSize;
              const labelMaxLength = baseSize > 55 ? 14 : 10;

              // Enhanced glow for semantic fields
              const hasGlow = emphasis || isHovered || isSelected || isConnected;
              const glowIntensity = isSemantic ? 1.2 : isLexical ? 1 : 0.8;

              return (
                <div
                  key={id}
                  className="absolute transform -translate-x-1/2 -translate-y-1/2 transition-all duration-300 cursor-pointer group"
                  style={{
                    left: `${x}%`,
                    top: `${y}%`,
                    zIndex: isHovered || isSelected ? 30 : isConnected ? 25 : isSemantic ? 20 : isLexical ? 15 : 10,
                    opacity: isDimmed ? 0.25 : 1,
                  }}
                  onMouseEnter={() => setHoveredNode(id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  onClick={() => setSelectedNode(selectedNode === id ? null : id)}
                >
                  {/* Ambient glow for semantic/lexical fields */}
                  {isSemantic && (
                    <div
                      className="absolute rounded-full animate-pulse"
                      style={{
                        width: `${size * 1.8}px`,
                        height: `${size * 1.8}px`,
                        left: `${-size * 0.4}px`,
                        top: `${-size * 0.4}px`,
                        background: `radial-gradient(circle, ${node.color}40 0%, transparent 70%)`,
                      }}
                    />
                  )}

                  {/* Glow effect */}
                  {hasGlow && (
                    <div
                      className="absolute inset-0 rounded-full transition-all"
                      style={{
                        backgroundColor: node.color,
                        opacity: (isHovered || isSelected ? 0.7 : 0.4) * glowIntensity,
                        transform: `scale(${1.4 + glowIntensity * 0.2})`,
                        filter: 'blur(12px)',
                      }}
                    />
                  )}

                  {/* Node */}
                  <div
                    className="relative rounded-full flex items-center justify-center text-white font-medium shadow-lg transition-all duration-300"
                    style={{
                      width: `${size}px`,
                      height: `${size}px`,
                      background: isSemantic
                        ? `linear-gradient(135deg, ${node.color}, ${node.color}dd)`
                        : node.color,
                      boxShadow: isHovered || isSelected
                        ? `0 0 40px ${node.color}90, 0 0 80px ${node.color}50, inset 0 0 20px rgba(255,255,255,0.1)`
                        : isSemantic
                        ? `0 8px 32px ${node.color}60, inset 0 0 15px rgba(255,255,255,0.1)`
                        : `0 4px 20px ${node.color}40`,
                      border: isSelected
                        ? '3px solid white'
                        : isSemantic
                        ? '2px solid rgba(255,255,255,0.4)'
                        : '2px solid rgba(255,255,255,0.2)',
                    }}
                  >
                    <span
                      className={`text-center leading-tight px-1 drop-shadow-md ${
                        baseSize > 55 ? 'text-[11px]' : 'text-[9px]'
                      }`}
                    >
                      {node.label.length > labelMaxLength ? node.label.slice(0, labelMaxLength) + '...' : node.label}
                    </span>
                  </div>

                  {/* Type indicator badge for semantic fields */}
                  {isSemantic && viewMode === 'all' && (
                    <div
                      className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-purple-400 border-2 border-slate-900 flex items-center justify-center"
                      title="Campo Semântico"
                    >
                      <Brain className="w-2 h-2 text-white" />
                    </div>
                  )}

                  {/* Tooltip - appears below if node is in top half */}
                  {y < 50 ? (
                    <div className={`absolute left-1/2 -translate-x-1/2 transition-all duration-200 pointer-events-none ${
                      isHovered ? 'opacity-100 top-full mt-2' : 'opacity-0 top-full mt-0'
                    }`} style={{ zIndex: 50 }}>
                      <div className="bg-slate-900/95 backdrop-blur-sm border border-white/20 rounded-xl px-4 py-3 shadow-2xl min-w-[200px]">
                        <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-slate-900/95 border-l border-t border-white/20 transform rotate-45" />
                        <p className="font-semibold text-white text-sm mb-1">{node.label}</p>
                        <div className="flex items-center gap-2 mb-2">
                          <span
                            className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                            style={{ backgroundColor: `${node.color}30`, color: node.color }}
                          >
                            {node.type === 'conceito_cientifico' ? 'Conceito' :
                             node.type === 'campo_semantico' ? 'Semântico' : 'Lexical'}
                          </span>
                          <span className="text-[10px] text-slate-400">{node.count}x</span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className={`absolute left-1/2 -translate-x-1/2 transition-all duration-200 pointer-events-none ${
                      isHovered ? 'opacity-100 bottom-full mb-2' : 'opacity-0 bottom-full mb-0'
                    }`} style={{ zIndex: 50 }}>
                      <div className="bg-slate-900/95 backdrop-blur-sm border border-white/20 rounded-xl px-4 py-3 shadow-2xl min-w-[200px]">
                        <p className="font-semibold text-white text-sm mb-1">{node.label}</p>
                        <div className="flex items-center gap-2 mb-2">
                          <span
                            className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                            style={{ backgroundColor: `${node.color}30`, color: node.color }}
                          >
                            {node.type === 'conceito_cientifico' ? 'Conceito' :
                             node.type === 'campo_semantico' ? 'Semântico' : 'Lexical'}
                          </span>
                          <span className="text-[10px] text-slate-400">{node.count}x</span>
                        </div>
                        <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-slate-900/95 border-r border-b border-white/20 transform rotate-45" />
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Center decoration */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
              <div className="relative">
                <div className="absolute inset-0 w-20 h-20 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 blur-xl" />
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border border-white/10 flex items-center justify-center">
                  <Brain className="w-8 h-8 text-purple-400" />
                </div>
              </div>
            </div>
          </div>

          {/* Legend & Selected Info */}
          <div className="px-6 py-4 border-t border-white/10 bg-black/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div className={`flex items-center gap-2 transition-opacity ${viewMode === 'concepts' || viewMode === 'all' ? 'opacity-100' : 'opacity-40'}`}>
                  <div className="w-4 h-4 rounded-full bg-blue-500 shadow-lg shadow-blue-500/30" />
                  <span className="text-xs text-slate-400">Conceito Científico</span>
                  <span className="text-xs text-slate-500">({graphData.stats.concept_nodes})</span>
                </div>
                <div className={`flex items-center gap-2 transition-opacity ${viewMode === 'semantic' || viewMode === 'all' ? 'opacity-100' : 'opacity-40'}`}>
                  <div className="w-5 h-5 rounded-full bg-purple-500 shadow-lg shadow-purple-500/30 animate-pulse" />
                  <span className="text-xs text-slate-300 font-medium">Campo Semântico</span>
                  <span className="text-xs text-purple-400 font-semibold">({graphData.stats.semantic_nodes})</span>
                </div>
                <div className={`flex items-center gap-2 transition-opacity ${viewMode === 'lexical' || viewMode === 'all' ? 'opacity-100' : 'opacity-40'}`}>
                  <div className="w-4 h-4 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/30" />
                  <span className="text-xs text-slate-400">Campo Lexical</span>
                  <span className="text-xs text-slate-500">({graphData.stats.lexical_nodes})</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {viewMode !== 'all' && (
                  <span className="text-xs text-purple-400 bg-purple-500/20 px-2 py-1 rounded-full">
                    Visualizando: {viewMode === 'semantic' ? 'Campos Semânticos' : viewMode === 'lexical' ? 'Campos Lexicais' : 'Conceitos'}
                  </span>
                )}
                <p className="text-xs text-slate-500">Clique para fixar | Hover para conexões</p>
              </div>
            </div>
          </div>

          {/* Top Connections */}
          <div className="px-6 py-4 border-t border-white/10">
            <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" />
              Principais Conexões Identificadas
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
              {graphData.edges.slice(0, 8).map((edge, idx) => {
                const sourceName = edge.source.replace('concept_', '').replace('semantic_', '').replace('lexical_', '');
                const targetName = edge.target.replace('concept_', '').replace('semantic_', '').replace('lexical_', '');
                return (
                  <div
                    key={idx}
                    className="flex items-center gap-2 text-xs p-3 bg-white/5 hover:bg-white/10 rounded-xl border border-white/5 transition-colors cursor-pointer"
                  >
                    <span className="font-medium text-slate-300 truncate flex-1">{sourceName}</span>
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-gradient-to-r from-purple-500/30 to-blue-500/30 flex items-center justify-center">
                      <ArrowUpRight className="h-3 w-3 text-purple-400" />
                    </div>
                    <span className="text-slate-400 truncate flex-1">{targetName}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
