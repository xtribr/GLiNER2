import type {
  DiagnosisComparisonResult, SchoolHistory, TRIAreaProjection,
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

export interface ProjectionCell {
  current: number | null;
  recommended: number | null;
  potential_gain: number | null;
  scenarios: { conservative: number; realistic: number; optimistic: number } | null;
  official_next: number | null;
  official_change: number | null;
  risk_level: 'normal' | 'conservative' | 'outlier' | null;
  trend_dir: 'ascending' | 'descending' | 'stable' | 'insufficient_data' | null;
  trend_annual: number | null;
}
export interface ProjectionFocusItem { skill: string; gap: number; }
export interface ProjectionRow {
  area: string;
  area_name: string;
  target_year: number;
  a: ProjectionCell;
  b: ProjectionCell;
  a_focus: ProjectionFocusItem[];
  b_focus: ProjectionFocusItem[];
}

// Keep TRIAreaProjection re-exported for convenience
export type { TRIAreaProjection };
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
