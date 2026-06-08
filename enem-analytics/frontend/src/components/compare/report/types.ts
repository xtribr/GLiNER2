import type {
  DiagnosisComparisonResult, SchoolHistory,
} from '@/lib/api';

export interface ReportSchoolMeta {
  codigo_inep: string;
  nome_escola: string;        // NOME REAL (não "Escola 1/2")
  uf: string | null;
  cidade?: string | null;
  tipo_escola: string | null;
  porte_label?: string | null;
  nota_media: number | null;
  ranking_brasil: number | null;
  ranking_uf: number | null;
  overall_health: 'excellent' | 'good' | 'needs_attention' | 'critical' | null;
}

export interface ComparisonYearRow {
  ano: number;
  a_media: number | null; a_rank: number | null;
  b_media: number | null; b_rank: number | null;
}

// Seções avançadas (Fases 2-3) são opcionais — degradação graciosa.
export interface ReportData {
  generatedAt: Date;
  baseYear: number;
  schoolA: ReportSchoolMeta;
  schoolB: ReportSchoolMeta;
  diagnosis: DiagnosisComparisonResult;     // area_comparison + status
  history: ComparisonYearRow[];
  // Fase 2+:
  projection?: ProjectionRow[];
  redacaoCompetencias?: RedacaoCompRow[];
  recommendations?: RecommendationRow[];
  skills?: { a: SkillRow[]; b: SkillRow[] };
}

export interface ProjectionRow {
  area: string; area_name: string;
  a_current: number | null; a_projected: number | null; a_gain: number | null;
  b_current: number | null; b_projected: number | null; b_gain: number | null;
  focus_content: string;       // conteúdo de maior gap
}
export interface RedacaoCompRow {
  comp: 'C1'|'C2'|'C3'|'C4'|'C5'; label: string;
  a: number | null; b: number | null; nacional: number | null;
  reading: string;
}
export interface RecommendationRow {
  scope: 'A'|'B'|'Ambas'|'Benchmark'; priority: 'Alta'|'Média'|'Baixa'|'—';
  action: string; impact: string;
}
export interface SkillRow { area: string; skill: string; kind: 'forte'|'fraca'; }

// Re-export for convenience
export type { DiagnosisComparisonResult, SchoolHistory };
