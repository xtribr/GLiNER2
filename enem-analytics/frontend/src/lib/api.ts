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
};
