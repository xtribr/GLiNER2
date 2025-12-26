'use client';

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
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
  ZoomIn,
  ZoomOut,
  Maximize2,
  Filter,
  Layers,
  Check,
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
// Features: Entity filters, Flow animations, Zoom/Pan, Thematic clusters
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
  const [viewMode, setViewMode] = useState<'all' | 'clusters'>('all');

  // Entity type filters (multi-select)
  const [entityFilters, setEntityFilters] = useState({
    conceito_cientifico: true,
    campo_semantico: true,
    campo_lexical: true,
  });
  const [showFilterDropdown, setShowFilterDropdown] = useState(false);

  // Zoom and Pan state
  const [transform, setTransform] = useState({ scale: 1, x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Animation state
  const [animationEnabled, setAnimationEnabled] = useState(true);

  const { data: graphData, isLoading } = useQuery({
    queryKey: ['glinerGraph', codigoInep, networkArea],
    queryFn: () => api.getGlinerKnowledgeGraph(codigoInep, networkArea),
  });

  // Filter nodes based on entity type selection
  const filteredNodes = useMemo(() => {
    if (!graphData?.nodes) return [];
    return graphData.nodes.filter(node => entityFilters[node.type as keyof typeof entityFilters]);
  }, [graphData?.nodes, entityFilters]);

  // Filter edges to only include those with visible nodes
  const filteredEdges = useMemo(() => {
    if (!graphData?.edges) return [];
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    return graphData.edges.filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  }, [graphData?.edges, filteredNodes]);

  // Toggle entity filter
  const toggleEntityFilter = (type: keyof typeof entityFilters) => {
    setEntityFilters(prev => ({ ...prev, [type]: !prev[type] }));
  };

  // Zoom handlers
  const handleZoom = useCallback((delta: number) => {
    setTransform(prev => ({
      ...prev,
      scale: Math.min(Math.max(prev.scale + delta, 0.5), 3),
    }));
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    handleZoom(delta);
  }, [handleZoom]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click
    setIsDragging(true);
    setDragStart({ x: e.clientX - transform.x, y: e.clientY - transform.y });
  }, [transform.x, transform.y]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setTransform(prev => ({
      ...prev,
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    }));
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const resetView = useCallback(() => {
    setTransform({ scale: 1, x: 0, y: 0 });
  }, []);

  // Area colors for clustering
  const areaColors: { [key: string]: { bg: string; border: string; label: string } } = {
    CN: { bg: 'rgba(34, 197, 94, 0.1)', border: 'rgba(34, 197, 94, 0.3)', label: 'Ciências da Natureza' },
    CH: { bg: 'rgba(234, 179, 8, 0.1)', border: 'rgba(234, 179, 8, 0.3)', label: 'Ciências Humanas' },
    LC: { bg: 'rgba(239, 68, 68, 0.1)', border: 'rgba(239, 68, 68, 0.3)', label: 'Linguagens' },
    MT: { bg: 'rgba(59, 130, 246, 0.1)', border: 'rgba(59, 130, 246, 0.3)', label: 'Matemática' },
  };

  // Enhanced node positioning with cluster support
  type GraphNode = {
    id: string;
    label: string;
    type: string;
    color: string;
    size: number;
    count: number;
    area?: string;
    area_name?: string;
    area_distribution?: { [key: string]: number };
    is_interdisciplinary?: boolean;
  };
  const calculatePositions = useCallback((nodes: GraphNode[]) => {
    if (!nodes || nodes.length === 0) return {};

    const positions: { [key: string]: { x: number; y: number; node: GraphNode; ring: number; emphasis: boolean; cluster?: string } } = {};
    const centerX = 50;
    const centerY = 50;

    if (viewMode === 'clusters') {
      // Cluster layout by area
      const clusterPositions: { [key: string]: { cx: number; cy: number } } = {
        CN: { cx: 25, cy: 30 },
        CH: { cx: 75, cy: 30 },
        LC: { cx: 25, cy: 70 },
        MT: { cx: 75, cy: 70 },
      };

      // Group nodes by area (now using actual area from backend)
      const nodesByArea: { [key: string]: GraphNode[] } = { CN: [], CH: [], LC: [], MT: [], other: [] };

      nodes.forEach(node => {
        // Use the area provided by backend (primary area based on frequency)
        const area = node.area || 'other';
        if (clusterPositions[area]) {
          nodesByArea[area].push(node);
        } else {
          nodesByArea.other.push(node);
        }
      });

      // Position nodes within each cluster
      Object.entries(nodesByArea).forEach(([area, areaNodes]) => {
        if (area === 'other' || areaNodes.length === 0) return;
        const cluster = clusterPositions[area];
        const radius = 15;

        areaNodes.forEach((node, i) => {
          const angle = (i / areaNodes.length) * Math.PI * 2 - Math.PI / 2;
          const r = areaNodes.length > 6 ? radius * (0.5 + (i % 2) * 0.5) : radius * 0.7;
          positions[node.id] = {
            x: cluster.cx + Math.cos(angle) * r,
            y: cluster.cy + Math.sin(angle) * r,
            node,
            ring: 1,
            emphasis: node.type === 'campo_semantico',
            cluster: area,
          };
        });
      });

      // Position "other" nodes in the center
      nodesByArea.other.forEach((node, i) => {
        const angle = (i / Math.max(nodesByArea.other.length, 1)) * Math.PI * 2;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * 8,
          y: centerY + Math.sin(angle) * 8,
          node,
          ring: 0,
          emphasis: false,
        };
      });
    } else {
      // Standard hierarchical layout
      const semanticNodes = nodes.filter(n => n.type === 'campo_semantico');
      const lexicalNodes = nodes.filter(n => n.type === 'campo_lexical');
      const conceptNodes = nodes.filter(n => n.type === 'conceito_cientifico');

      // Core ring: Semantic fields
      const semanticCount = Math.min(semanticNodes.length, 8);
      semanticNodes.slice(0, semanticCount).forEach((node, i) => {
        const angle = (i / semanticCount) * Math.PI * 2 - Math.PI / 2;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * 12,
          y: centerY + Math.sin(angle) * 12,
          node,
          ring: 1,
          emphasis: true,
        };
      });

      // Middle ring: Lexical fields
      const lexicalCount = Math.min(lexicalNodes.length, 14);
      lexicalNodes.slice(0, lexicalCount).forEach((node, i) => {
        const angle = (i / lexicalCount) * Math.PI * 2 - Math.PI / 6;
        const jitter = (i % 2) * 1;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * (24 + jitter),
          y: centerY + Math.sin(angle) * (24 + jitter),
          node,
          ring: 2,
          emphasis: false,
        };
      });

      // Outer ring: Concepts
      const conceptCount = Math.min(conceptNodes.length, 20);
      conceptNodes.slice(0, conceptCount).forEach((node, i) => {
        const angle = (i / conceptCount) * Math.PI * 2;
        positions[node.id] = {
          x: centerX + Math.cos(angle) * 36,
          y: centerY + Math.sin(angle) * 36,
          node,
          ring: 3,
          emphasis: false,
        };
      });
    }

    return positions;
  }, [viewMode]);

  const nodePositions = useMemo(() =>
    calculatePositions(filteredNodes),
    [filteredNodes, calculatePositions]
  );

  // Get connected nodes for highlighting
  const getConnectedNodes = useCallback((nodeId: string) => {
    const connected = new Set<string>();
    filteredEdges.forEach(edge => {
      if (edge.source === nodeId) connected.add(edge.target);
      if (edge.target === nodeId) connected.add(edge.source);
    });
    return connected;
  }, [filteredEdges]);

  const connectedNodes = hoveredNode ? getConnectedNodes(hoveredNode) : new Set<string>();
  const selectedConnections = selectedNode ? getConnectedNodes(selectedNode) : new Set<string>();

  // Count active filters
  const activeFilterCount = Object.values(entityFilters).filter(Boolean).length;

  return (
    <div className="space-y-4">
      {/* Enhanced Filters Row */}
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

        {/* Entity Type Filter Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowFilterDropdown(!showFilterDropdown)}
            className={`flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg transition-all ${
              activeFilterCount < 3
                ? 'bg-purple-100 text-purple-700 border border-purple-200'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <Filter className="w-3.5 h-3.5" />
            <span>Filtrar Entidades</span>
            {activeFilterCount < 3 && (
              <span className="px-1.5 py-0.5 bg-purple-600 text-white rounded-full text-[10px]">
                {activeFilterCount}
              </span>
            )}
          </button>

          {showFilterDropdown && (
            <div className="absolute top-full right-0 mt-1 bg-white rounded-xl shadow-xl border border-gray-200 p-2 z-50 min-w-[200px]">
              <div className="text-xs text-gray-500 px-2 py-1 border-b border-gray-100 mb-1">
                Tipos de Entidade
              </div>
              {[
                { key: 'conceito_cientifico', label: 'Conceitos Científicos', color: 'blue' },
                { key: 'campo_semantico', label: 'Campos Semânticos', color: 'purple' },
                { key: 'campo_lexical', label: 'Campos Lexicais', color: 'emerald' },
              ].map(({ key, label, color }) => (
                <button
                  key={key}
                  onClick={() => toggleEntityFilter(key as keyof typeof entityFilters)}
                  className="w-full flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div
                    className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all ${
                      entityFilters[key as keyof typeof entityFilters]
                        ? `bg-${color}-500 border-${color}-500`
                        : 'border-gray-300'
                    }`}
                    style={{
                      backgroundColor: entityFilters[key as keyof typeof entityFilters]
                        ? color === 'blue' ? '#3b82f6' : color === 'purple' ? '#a855f7' : '#10b981'
                        : 'transparent',
                      borderColor: entityFilters[key as keyof typeof entityFilters]
                        ? color === 'blue' ? '#3b82f6' : color === 'purple' ? '#a855f7' : '#10b981'
                        : '#d1d5db',
                    }}
                  >
                    {entityFilters[key as keyof typeof entityFilters] && (
                      <Check className="w-3 h-3 text-white" />
                    )}
                  </div>
                  <span className="text-sm text-gray-700">{label}</span>
                  <div
                    className="w-2.5 h-2.5 rounded-full ml-auto"
                    style={{
                      backgroundColor: color === 'blue' ? '#3b82f6' : color === 'purple' ? '#a855f7' : '#10b981',
                    }}
                  />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Layout:</span>
          <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5">
            <button
              onClick={() => setViewMode('all')}
              className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1 ${
                viewMode === 'all'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Network className="w-3 h-3" />
              Radial
            </button>
            <button
              onClick={() => setViewMode('clusters')}
              className={`px-3 py-1 text-xs rounded-md transition-all flex items-center gap-1 ${
                viewMode === 'clusters'
                  ? 'bg-purple-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Layers className="w-3 h-3" />
              Clusters
            </button>
          </div>
        </div>

        {/* Animation Toggle */}
        <button
          onClick={() => setAnimationEnabled(!animationEnabled)}
          className={`px-3 py-1 text-xs rounded-lg transition-all flex items-center gap-1 ${
            animationEnabled
              ? 'bg-amber-100 text-amber-700 border border-amber-200'
              : 'bg-gray-100 text-gray-500'
          }`}
        >
          <Zap className={`w-3.5 h-3.5 ${animationEnabled ? 'animate-pulse' : ''}`} />
          Fluxo
        </button>
      </div>

      {/* Click outside to close filter dropdown */}
      {showFilterDropdown && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowFilterDropdown(false)}
        />
      )}

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
                <p className="text-xs text-slate-400">
                  Visualização interativa • {filteredNodes.length} nós • {filteredEdges.length} conexões
                </p>
              </div>
            </div>

            {/* Zoom Controls */}
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 bg-white/10 rounded-lg p-1">
                <button
                  onClick={() => handleZoom(-0.2)}
                  className="p-1.5 rounded hover:bg-white/10 transition-colors text-white"
                  title="Diminuir zoom"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <span className="text-xs text-slate-300 px-2 min-w-[50px] text-center">
                  {Math.round(transform.scale * 100)}%
                </span>
                <button
                  onClick={() => handleZoom(0.2)}
                  className="p-1.5 rounded hover:bg-white/10 transition-colors text-white"
                  title="Aumentar zoom"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <button
                  onClick={resetView}
                  className="p-1.5 rounded hover:bg-white/10 transition-colors text-white ml-1"
                  title="Resetar visualização"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>

              <div className="flex items-center gap-4 text-xs ml-4">
                {entityFilters.conceito_cientifico && (
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500 shadow-lg shadow-blue-500/50" />
                    <span className="text-slate-300">{filteredNodes.filter(n => n.type === 'conceito_cientifico').length}</span>
                  </div>
                )}
                {entityFilters.campo_semantico && (
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-purple-500 shadow-lg shadow-purple-500/50" />
                    <span className="text-slate-300">{filteredNodes.filter(n => n.type === 'campo_semantico').length}</span>
                  </div>
                )}
                {entityFilters.campo_lexical && (
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/50" />
                    <span className="text-slate-300">{filteredNodes.filter(n => n.type === 'campo_lexical').length}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Neural Network Visualization with Zoom/Pan */}
          <div
            ref={containerRef}
            className="relative h-[500px] overflow-hidden cursor-grab active:cursor-grabbing"
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* Background Grid Effect */}
            <div className="absolute inset-0 opacity-10 pointer-events-none">
              <div className="absolute inset-0" style={{
                backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.3) 1px, transparent 0)`,
                backgroundSize: '40px 40px'
              }} />
            </div>

            {/* Glow Effects */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-purple-500/20 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute top-1/4 left-1/4 w-48 h-48 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute bottom-1/4 right-1/4 w-48 h-48 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

            {/* Cluster backgrounds (when in cluster mode) */}
            {viewMode === 'clusters' && (
              <div
                className="absolute inset-0 pointer-events-none"
                style={{
                  transform: `scale(${transform.scale}) translate(${transform.x / transform.scale}px, ${transform.y / transform.scale}px)`,
                  transformOrigin: 'center center',
                }}
              >
                {Object.entries(areaColors).map(([area, colors]) => {
                  const positions: { [key: string]: { cx: number; cy: number } } = {
                    CN: { cx: 25, cy: 30 },
                    CH: { cx: 75, cy: 30 },
                    LC: { cx: 25, cy: 70 },
                    MT: { cx: 75, cy: 70 },
                  };
                  const pos = positions[area];
                  return (
                    <div
                      key={area}
                      className="absolute rounded-3xl border-2 border-dashed"
                      style={{
                        left: `${pos.cx - 20}%`,
                        top: `${pos.cy - 18}%`,
                        width: '40%',
                        height: '36%',
                        backgroundColor: colors.bg,
                        borderColor: colors.border,
                      }}
                    >
                      <div className="absolute -top-3 left-4 px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                        {colors.label}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Transformable container for nodes and edges */}
            <div
              className="absolute inset-0"
              style={{
                transform: `scale(${transform.scale}) translate(${transform.x / transform.scale}px, ${transform.y / transform.scale}px)`,
                transformOrigin: 'center center',
                transition: isDragging ? 'none' : 'transform 0.1s ease-out',
              }}
            >
              {/* SVG for connections with flow animation */}
              <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 1 }}>
                <defs>
                  <linearGradient id="lineGradientEnhanced" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="rgba(139, 92, 246, 0.6)" />
                    <stop offset="100%" stopColor="rgba(59, 130, 246, 0.6)" />
                  </linearGradient>
                  <filter id="glowEnhanced">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge>
                      <feMergeNode in="coloredBlur"/>
                      <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                  </filter>
                  {/* Flow animation marker */}
                  <marker
                    id="flowDot"
                    markerWidth="6"
                    markerHeight="6"
                    refX="3"
                    refY="3"
                  >
                    <circle cx="3" cy="3" r="2" fill="rgba(168, 85, 247, 0.8)" />
                  </marker>
                </defs>

                {/* Draw edges with optional flow animation */}
                {filteredEdges.slice(0, 80).map((edge, idx) => {
                  const sourcePos = nodePositions[edge.source];
                  const targetPos = nodePositions[edge.target];
                  if (!sourcePos || !targetPos) return null;

                  const isHighlighted = hoveredNode === edge.source || hoveredNode === edge.target ||
                                       selectedNode === edge.source || selectedNode === edge.target;

                  // Calculate path for curved connections
                  const midX = (sourcePos.x + targetPos.x) / 2;
                  const midY = (sourcePos.y + targetPos.y) / 2;
                  const curvature = 2 + (idx % 3);
                  const curveX = midX + curvature;
                  const curveY = midY - curvature;

                  return (
                    <g key={idx}>
                      {/* Main edge line */}
                      <path
                        d={`M ${sourcePos.x}% ${sourcePos.y}% Q ${curveX}% ${curveY}% ${targetPos.x}% ${targetPos.y}%`}
                        fill="none"
                        stroke={isHighlighted ? 'rgba(168, 85, 247, 0.8)' : 'rgba(148, 163, 184, 0.15)'}
                        strokeWidth={isHighlighted ? 2 : 1}
                        filter={isHighlighted ? 'url(#glowEnhanced)' : undefined}
                        className="transition-all duration-300"
                      />

                      {/* Flow animation particles */}
                      {animationEnabled && isHighlighted && (
                        <>
                          <circle r="3" fill="rgba(168, 85, 247, 0.9)">
                            <animateMotion
                              dur={`${1.5 + (idx % 3) * 0.5}s`}
                              repeatCount="indefinite"
                              path={`M ${sourcePos.x * 10} ${sourcePos.y * 5} Q ${curveX * 10} ${curveY * 5} ${targetPos.x * 10} ${targetPos.y * 5}`}
                            />
                          </circle>
                          <circle r="2" fill="rgba(59, 130, 246, 0.7)">
                            <animateMotion
                              dur={`${2 + (idx % 2) * 0.5}s`}
                              repeatCount="indefinite"
                              begin="0.5s"
                              path={`M ${sourcePos.x * 10} ${sourcePos.y * 5} Q ${curveX * 10} ${curveY * 5} ${targetPos.x * 10} ${targetPos.y * 5}`}
                            />
                          </circle>
                        </>
                      )}
                    </g>
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

                const isSemantic = node.type === 'campo_semantico';
                const isLexical = node.type === 'campo_lexical';

                // Dynamic sizing
                let baseSize = 40;
                if (isSemantic) baseSize = 60;
                else if (isLexical) baseSize = 50;

                const size = isHovered || isSelected ? baseSize + 10 : baseSize;
                const labelMaxLength = baseSize > 55 ? 14 : 10;

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
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedNode(selectedNode === id ? null : id);
                    }}
                  >
                    {/* Ambient glow for semantic fields */}
                    {isSemantic && (
                      <div
                        className={`absolute rounded-full ${animationEnabled ? 'animate-pulse' : ''}`}
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

                    {/* Type indicator badge */}
                    {isSemantic && viewMode === 'all' && (
                      <div
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-purple-400 border-2 border-slate-900 flex items-center justify-center"
                        title="Campo Semântico"
                      >
                        <Brain className="w-2 h-2 text-white" />
                      </div>
                    )}

                    {/* Interdisciplinary indicator badge */}
                    {node.is_interdisciplinary && viewMode === 'clusters' && (
                      <div
                        className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-amber-400 border-2 border-slate-900 flex items-center justify-center animate-pulse"
                        title="Conceito Interdisciplinar"
                      >
                        <span className="text-[8px] font-bold text-slate-900">+</span>
                      </div>
                    )}

                    {/* Tooltip */}
                    {y < 50 ? (
                      <div className={`absolute left-1/2 -translate-x-1/2 transition-all duration-200 pointer-events-none ${
                        isHovered ? 'opacity-100 top-full mt-2' : 'opacity-0 top-full mt-0'
                      }`} style={{ zIndex: 50 }}>
                        <div className="bg-slate-900/95 backdrop-blur-sm border border-white/20 rounded-xl px-4 py-3 shadow-2xl min-w-[220px]">
                          <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-slate-900/95 border-l border-t border-white/20 transform rotate-45" />
                          <p className="font-semibold text-white text-sm mb-1">{node.label}</p>
                          <div className="flex items-center gap-2 mb-2 flex-wrap">
                            <span
                              className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                              style={{ backgroundColor: `${node.color}30`, color: node.color }}
                            >
                              {node.type === 'conceito_cientifico' ? 'Conceito' :
                               node.type === 'campo_semantico' ? 'Semântico' : 'Lexical'}
                            </span>
                            <span className="text-[10px] text-slate-400">{node.count}x</span>
                            {node.area && (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-slate-700 text-slate-300">
                                {node.area_name || node.area}
                              </span>
                            )}
                          </div>
                          {node.is_interdisciplinary && node.area_distribution && (
                            <div className="mt-2 pt-2 border-t border-white/10">
                              <p className="text-[10px] text-amber-400 mb-1">Interdisciplinar:</p>
                              <div className="flex gap-1 flex-wrap">
                                {Object.entries(node.area_distribution).map(([area, count]) => (
                                  <span key={area} className="text-[9px] px-1.5 py-0.5 bg-slate-700/50 rounded text-slate-400">
                                    {area}: {count}x
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className={`absolute left-1/2 -translate-x-1/2 transition-all duration-200 pointer-events-none ${
                        isHovered ? 'opacity-100 bottom-full mb-2' : 'opacity-0 bottom-full mb-0'
                      }`} style={{ zIndex: 50 }}>
                        <div className="bg-slate-900/95 backdrop-blur-sm border border-white/20 rounded-xl px-4 py-3 shadow-2xl min-w-[220px]">
                          <p className="font-semibold text-white text-sm mb-1">{node.label}</p>
                          <div className="flex items-center gap-2 mb-2 flex-wrap">
                            <span
                              className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                              style={{ backgroundColor: `${node.color}30`, color: node.color }}
                            >
                              {node.type === 'conceito_cientifico' ? 'Conceito' :
                               node.type === 'campo_semantico' ? 'Semântico' : 'Lexical'}
                            </span>
                            <span className="text-[10px] text-slate-400">{node.count}x</span>
                            {node.area && (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-slate-700 text-slate-300">
                                {node.area_name || node.area}
                              </span>
                            )}
                          </div>
                          {node.is_interdisciplinary && node.area_distribution && (
                            <div className="mt-2 pt-2 border-t border-white/10">
                              <p className="text-[10px] text-amber-400 mb-1">Interdisciplinar:</p>
                              <div className="flex gap-1 flex-wrap">
                                {Object.entries(node.area_distribution).map(([area, count]) => (
                                  <span key={area} className="text-[9px] px-1.5 py-0.5 bg-slate-700/50 rounded text-slate-400">
                                    {area}: {count}x
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-slate-900/95 border-r border-b border-white/20 transform rotate-45" />
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Center decoration */}
              {viewMode === 'all' && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
                  <div className="relative">
                    <div className="absolute inset-0 w-20 h-20 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 blur-xl" />
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border border-white/10 flex items-center justify-center">
                      <Brain className={`w-8 h-8 text-purple-400 ${animationEnabled ? 'animate-pulse' : ''}`} />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Zoom level indicator (bottom right) */}
            <div className="absolute bottom-4 right-4 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-1.5 text-xs text-slate-300">
              {transform.scale !== 1 || transform.x !== 0 || transform.y !== 0 ? (
                <span>Zoom: {Math.round(transform.scale * 100)}% • Scroll para zoom, arraste para mover</span>
              ) : (
                <span>Scroll para zoom • Arraste para mover</span>
              )}
            </div>
          </div>

          {/* Legend & Controls */}
          <div className="px-6 py-4 border-t border-white/10 bg-black/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                {entityFilters.conceito_cientifico && (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-blue-500 shadow-lg shadow-blue-500/30" />
                    <span className="text-xs text-slate-400">Conceito Científico</span>
                  </div>
                )}
                {entityFilters.campo_semantico && (
                  <div className="flex items-center gap-2">
                    <div className={`w-5 h-5 rounded-full bg-purple-500 shadow-lg shadow-purple-500/30 ${animationEnabled ? 'animate-pulse' : ''}`} />
                    <span className="text-xs text-slate-300 font-medium">Campo Semântico</span>
                  </div>
                )}
                {entityFilters.campo_lexical && (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/30" />
                    <span className="text-xs text-slate-400">Campo Lexical</span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-3">
                {viewMode === 'clusters' && (
                  <span className="text-xs text-purple-400 bg-purple-500/20 px-2 py-1 rounded-full">
                    Modo Clusters por Área
                  </span>
                )}
                <p className="text-xs text-slate-500">Clique para fixar • Hover para conexões</p>
              </div>
            </div>
          </div>

          {/* Top Connections */}
          <div className="px-6 py-4 border-t border-white/10">
            <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
              <Zap className={`w-4 h-4 text-amber-400 ${animationEnabled ? 'animate-pulse' : ''}`} />
              Principais Conexões Identificadas
            </h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
              {filteredEdges.slice(0, 8).map((edge, idx) => {
                const sourceName = edge.source.replace('concept_', '').replace('semantic_', '').replace('lexical_', '');
                const targetName = edge.target.replace('concept_', '').replace('semantic_', '').replace('lexical_', '');
                return (
                  <div
                    key={idx}
                    className="flex items-center gap-2 text-xs p-3 bg-white/5 hover:bg-white/10 rounded-xl border border-white/5 transition-colors cursor-pointer"
                    onMouseEnter={() => {
                      setHoveredNode(edge.source);
                    }}
                    onMouseLeave={() => setHoveredNode(null)}
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
