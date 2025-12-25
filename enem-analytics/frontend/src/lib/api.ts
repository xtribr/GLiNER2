const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081';

export interface SchoolScore {
  ano: number;
  nota_cn: number | null;
  nota_ch: number | null;
  nota_lc: number | null;
  nota_mt: number | null;
  nota_redacao: number | null;
  nota_media: number | null;
  ranking_brasil: number | null;
  desempenho_habilidades: number | null;
  competencia_redacao_media: number | null;
}

export interface SchoolSummary {
  codigo_inep: string;
  nome_escola: string;
  uf: string | null;
  tipo_escola: string | null;
  localizacao: string | null;
  porte: number | null;
  porte_label: string | null;
  qt_matriculas: number | null;
  ultimo_ranking: number | null;
  ultima_nota: number | null;
  anos_participacao: number;
}

export interface SchoolDetail {
  codigo_inep: string;
  nome_escola: string;
  uf: string | null;
  tipo_escola: string | null;
  historico: SchoolScore[];
  tendencia: string | null;
  melhor_ano: number | null;
  melhor_ranking: number | null;
}

export interface TopSchool {
  ranking: number;
  codigo_inep: string;
  nome_escola: string;
  uf: string | null;
  tipo_escola: string | null;
  localizacao: string | null;
  porte: number | null;
  porte_label: string | null;
  qt_matriculas: number | null;
  nota_media: number | null;
  nota_cn: number | null;
  nota_ch: number | null;
  nota_lc: number | null;
  nota_mt: number | null;
  nota_redacao: number | null;
  desempenho_habilidades: number | null;
  competencia_redacao_media: number | null;
}

export interface Stats {
  total_records: number;
  total_schools: number;
  years: number[];
  states: string[];
  avg_scores: {
    nota_cn: number;
    nota_ch: number;
    nota_lc: number;
    nota_mt: number;
    nota_redacao: number;
  };
}

// ML Types - Predictions
export interface PredictionResult {
  codigo_inep: string;
  target_year: number;
  scores: Record<string, number>;
  confidence_intervals: Record<string, { low: number; high: number }>;
  model_info: Record<string, unknown>;
}

export interface PredictionComparison {
  codigo_inep: string;
  historical: {
    year: number;
    scores: Record<string, number | null>;
  };
  predicted: {
    year: number;
    scores: Record<string, number>;
  };
  expected_change: Record<string, number>;
  confidence_intervals: Record<string, { low: number; high: number }>;
}

// ML Types - Diagnosis
export interface AreaAnalysis {
  area: string;
  area_name: string;
  school_score: number;
  national_avg: number;
  peer_avg: number;
  gap_to_national: number;
  gap_to_peer: number;
  z_score: number;
  status: 'excellent' | 'good' | 'needs_attention' | 'critical';
  priority_score: number;
}

export interface DiagnosisResult {
  codigo_inep: string;
  school_info: {
    codigo_inep: string;
    nome_escola: string;
    porte: number | null;
    tipo_escola: string | null;
    localizacao: string | null;
    ano: number;
  };
  overall_health: 'excellent' | 'good' | 'needs_attention' | 'critical';
  health_summary: {
    avg_z_score: number;
    critical_areas: number;
    excellent_areas: number;
    total_areas: number;
  };
  area_analysis: AreaAnalysis[];
  priority_areas: AreaAnalysis[];
  strengths: AreaAnalysis[];
  skill_gaps: {
    skill_code: string;
    area: string;
    skill_number: number;
    national_avg: number;
    estimated_school: number;
    gap: number;
    priority_score: number;
  }[];
  peer_comparison: {
    comparison_group: string;
    peer_count: number;
  };
}

// ML Types - Clustering
export interface ClusterPersona {
  name: string;
  description: string;
  color: string;
}

export interface ClusterResult {
  codigo_inep: string;
  cluster: number;
  persona: ClusterPersona;
  scores: Record<string, number>;
  cluster_center: Record<string, number>;
  distance_to_center: number;
}

export interface SimilarSchool {
  codigo_inep: string;
  nome_escola: string;
  distance: number;
  scores: Record<string, number>;
  porte: number | null;
  tipo_escola: string | null;
}

// ML Types - Recommendations
export interface RecommendationEvidence {
  available: boolean;
  schools_improved?: number;
  avg_improvement?: number;
  examples?: {
    escola: string;
    antes: number;
    depois: number;
    melhoria: number;
  }[];
  insight?: string;
}

export interface Recommendation {
  area: string;
  area_name: string;
  priority: number;
  current_score: number;
  target_score: number;
  expected_gain: number;
  difficulty: 'low' | 'medium' | 'high';
  gap_to_mean: number;
  evidence: RecommendationEvidence;
  action_items: string[];
}

export interface SuccessStory {
  codigo_inep: string;
  nome_escola: string;
  similarity_score: number;
  improvement: number;
  area_changes: Record<string, {
    before: number;
    after: number;
    change: number;
  }>;
  tipo_escola: string | null;
  porte: number | null;
}

export interface RecommendationResult {
  codigo_inep: string;
  school_info: {
    codigo_inep: string;
    nome_escola: string;
    porte: number | null;
    tipo_escola: string | null;
    localizacao: string | null;
    ano: number;
  };
  all_recommendations: Recommendation[];
  high_priority_recommendations: Recommendation[];
  quick_wins: Recommendation[];
  long_term_priorities: Recommendation[];
  success_stories: SuccessStory[];
  summary: {
    total_recommendations: number;
    high_priority_count: number;
    quick_wins_count: number;
    success_stories_count: number;
  };
}

export interface RoadmapPhase {
  phase: number;
  name: string;
  description: string;
  focus_areas: string[];
  expected_gain: number;
  action_items: string[];
}

export interface RoadmapResult {
  codigo_inep: string;
  school_info: {
    codigo_inep: string;
    nome_escola: string;
    porte: number | null;
    tipo_escola: string | null;
    localizacao: string | null;
    ano: number;
  };
  current_position: {
    nota_media_estimada: number;
    areas_criticas: number;
    areas_excelentes: number;
  };
  target_position: {
    nota_media_alvo: number;
    melhoria_esperada: number;
  };
  phases: RoadmapPhase[];
  total_phases: number;
  success_stories: SuccessStory[];
}

export interface SchoolHistory {
  codigo_inep: string;
  nome_escola: string;
  uf: string | null;
  tipo_escola: string | null;
  anos_participacao: number;
  history: {
    ano: number;
    ranking_brasil: number | null;
    ranking_change: number | null;
    nota_media: number | null;
    nota_change: number | null;
    nota_cn: number | null;
    nota_ch: number | null;
    nota_lc: number | null;
    nota_mt: number | null;
    nota_redacao: number | null;
    desempenho_habilidades: number | null;
    competencia_redacao_media: number | null;
  }[];
}

async function fetchAPI<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`);
  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }
  return response.json();
}

export const api = {
  getStats: () => fetchAPI<Stats>('/api/stats'),

  getTopSchools: (limit = 10, ano?: number, uf?: string) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (ano) params.set('ano', ano.toString());
    if (uf) params.set('uf', uf);
    return fetchAPI<{ ano: number; total: number; schools: TopSchool[] }>(
      `/api/schools/top?${params}`
    );
  },

  searchSchools: (q: string, limit = 20) =>
    fetchAPI<{ codigo_inep: string; nome_escola: string; uf: string | null; ultimo_ano: number }[]>(
      `/api/schools/search?q=${encodeURIComponent(q)}&limit=${limit}`
    ),

  getSchool: (codigo_inep: string) =>
    fetchAPI<SchoolDetail>(`/api/schools/${codigo_inep}`),

  getSchoolHistory: (codigo_inep: string) =>
    fetchAPI<SchoolHistory>(`/api/schools/${codigo_inep}/history`),

  listSchools: (params: {
    page?: number;
    limit?: number;
    search?: string;
    uf?: string;
    tipo_escola?: 'Privada' | 'Pública';
    localizacao?: 'Urbana' | 'Rural';
    porte?: number;
    ano?: number;
    order_by?: 'ranking' | 'nota' | 'nome';
    order?: 'asc' | 'desc';
  }) => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.set(key, value.toString());
      }
    });
    return fetchAPI<SchoolSummary[]>(`/api/schools?${searchParams}`);
  },

  compareSchools: (inep1: string, inep2: string) =>
    fetchAPI<{
      escola1: { codigo_inep: string; nome_escola: string; uf: string | null };
      escola2: { codigo_inep: string; nome_escola: string; uf: string | null };
      common_years: number[];
      comparison: {
        ano: number;
        escola1: { nota_media: number | null; ranking: number | null };
        escola2: { nota_media: number | null; ranking: number | null };
      }[];
    }>(`/api/schools/compare/${inep1}/${inep2}`),

  getWorstSkills: (area?: string, limit = 10) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (area) params.set('area', area);
    return fetchAPI<{
      ano: number;
      skills_by_area: Record<string, { skill_num: number; performance: number; descricao: string }[]>;
    }>(`/api/schools/skills/worst?${params}`);
  },

  getAllSkills: (area?: string) => {
    const params = new URLSearchParams();
    if (area) params.set('area', area);
    return fetchAPI<{
      ano: number;
      total: number;
      skills: { area: string; skill_num: number; performance: number; descricao: string }[];
    }>(`/api/schools/skills/all?${params}`);
  },

  getSchoolSkills: (codigo_inep: string, limit = 10) =>
    fetchAPI<{
      codigo_inep: string;
      ano: number;
      total_skills: number;
      worst_overall: {
        area: string;
        skill_num: number;
        performance: number;
        national_avg: number | null;
        diff: number | null;
        descricao: string;
        status: 'above' | 'below' | 'equal';
      }[];
      by_area: Record<string, {
        skill_num: number;
        performance: number;
        national_avg: number | null;
        diff: number | null;
        descricao: string;
        status: 'above' | 'below' | 'equal';
      }[]>;
    }>(`/api/schools/${codigo_inep}/skills?limit=${limit}`),

  // ML APIs - Predictions
  getPredictions: (codigo_inep: string, target_year = 2025) =>
    fetchAPI<PredictionResult>(`/api/predictions/${codigo_inep}?target_year=${target_year}`),

  getPredictionComparison: (codigo_inep: string) =>
    fetchAPI<PredictionComparison>(`/api/predictions/comparison/${codigo_inep}`),

  // ML APIs - Diagnosis
  getDiagnosis: (codigo_inep: string) =>
    fetchAPI<DiagnosisResult>(`/api/diagnosis/${codigo_inep}`),

  getAreaDiagnosis: (codigo_inep: string, area: string) =>
    fetchAPI<{
      codigo_inep: string;
      area: string;
      area_name: string;
      analysis: AreaAnalysis;
      skill_gaps: DiagnosisResult['skill_gaps'];
      peer_comparison: DiagnosisResult['peer_comparison'];
    }>(`/api/diagnosis/${codigo_inep}/area/${area}`),

  getImprovementPotential: (codigo_inep: string) =>
    fetchAPI<{
      codigo_inep: string;
      improvements: {
        area: string;
        area_name: string;
        current_score: number;
        peer_avg: number;
        potential_gain: number;
        effort_level: 'high' | 'medium';
      }[];
      total_potential_gain: number;
      priority_area: string | null;
    }>(`/api/diagnosis/${codigo_inep}/improvement-potential`),

  // ML APIs - Clustering
  getSchoolCluster: (codigo_inep: string) =>
    fetchAPI<ClusterResult>(`/api/clusters/${codigo_inep}/cluster`),

  getSimilarSchools: (codigo_inep: string, limit = 10, same_cluster = true) =>
    fetchAPI<{
      codigo_inep: string;
      school_cluster: ClusterResult;
      similar_schools: SimilarSchool[];
    }>(`/api/clusters/${codigo_inep}/similar?limit=${limit}&same_cluster=${same_cluster}`),

  getSimilarImprovedSchools: (codigo_inep: string, limit = 10, min_improvement = 10) =>
    fetchAPI<{
      codigo_inep: string;
      school_cluster: ClusterResult;
      improved_similar_schools: {
        codigo_inep: string;
        nome_escola: string;
        similarity_distance: number;
        improvement: number;
        scores_2023: Record<string, number>;
        scores_2024: Record<string, number>;
        tipo_escola: string | null;
        porte: number | null;
      }[];
      insight: string;
    }>(`/api/clusters/${codigo_inep}/similar-improved?limit=${limit}&min_improvement=${min_improvement}`),

  getClusterPersonas: () =>
    fetchAPI<{
      personas: {
        cluster: number;
        persona: ClusterPersona;
        center_scores: Record<string, number>;
        avg_media: number;
      }[];
    }>('/api/clusters/personas'),

  // ML APIs - Recommendations
  getRecommendations: (codigo_inep: string) =>
    fetchAPI<RecommendationResult>(`/api/recommendations/${codigo_inep}`),

  getRoadmap: (codigo_inep: string) =>
    fetchAPI<RoadmapResult>(`/api/recommendations/${codigo_inep}/roadmap`),

  getSuccessStories: (codigo_inep: string, limit = 10) =>
    fetchAPI<{
      codigo_inep: string;
      school_info: RecommendationResult['school_info'];
      success_stories: (SuccessStory & {
        highlight_area: string | null;
        highlight_area_name: string | null;
        highlight_improvement: number;
        key_insight: string | null;
      })[];
      total_found: number;
      insight: string;
    }>(`/api/recommendations/${codigo_inep}/success-stories?limit=${limit}`),

  getQuickWins: (codigo_inep: string, limit = 5) =>
    fetchAPI<{
      codigo_inep: string;
      school_info: RecommendationResult['school_info'];
      quick_wins: Recommendation[];
      total_available: number;
      recommendation: string;
    }>(`/api/recommendations/${codigo_inep}/quick-wins?limit=${limit}`),

  getActionPlan: (codigo_inep: string) =>
    fetchAPI<{
      codigo_inep: string;
      school_info: RecommendationResult['school_info'];
      action_plan: {
        immediate_actions: string[];
        short_term_goals: string[];
        long_term_objectives: string[];
      };
      expected_improvement: number;
      phases_count: number;
      success_stories_count: number;
    }>(`/api/recommendations/${codigo_inep}/action-plan`),
};
