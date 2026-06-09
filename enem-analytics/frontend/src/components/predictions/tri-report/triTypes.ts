import type { TRIAreaAnalysis, TRIAreaProjection } from '@/lib/api';

export interface TriReportArea {
  analysis: TRIAreaAnalysis;
  /** Detalhe de cenário; null se a chamada por área falhou (degradação graciosa). */
  projection: TRIAreaProjection | null;
}

export interface TriReportData {
  generatedAt: Date;
  codigoInep: string;
  schoolName: string;          // NOME REAL
  baseYear: number;            // ano do score atual (0 se indisponível)
  overallMastery: number;
  masteryInterpretation: string;
  recommendation: string;
  areas: TriReportArea[];
}
