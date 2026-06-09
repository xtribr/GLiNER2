import type { TriReportArea, TriReportData } from './triTypes';
import type { TRIAreaProjection } from '@/lib/api';
import { api } from '@/lib/api';

/**
 * Monta os dados do TRI REPORT: 1 chamada getTRIAnalysis (painel por área) +
 * 1 getAreaProjection por área (cenário). As projeções são buscadas em paralelo
 * (allSettled) — se uma área falhar, ela sai com `projection: null` e o relatório
 * ainda imprime o resumo daquela área.
 */
export async function buildTriReportData(codigoInep: string, schoolName: string): Promise<TriReportData> {
  const tri = await api.getTRIAnalysis(codigoInep);

  const settled = await Promise.allSettled(
    tri.area_analysis.map((a) => api.getAreaProjection(codigoInep, a.area.toLowerCase())),
  );

  const areas: TriReportArea[] = tri.area_analysis.map((analysis, i) => {
    const r = settled[i];
    const projection: TRIAreaProjection | null = r.status === 'fulfilled' ? r.value : null;
    return { analysis, projection };
  });

  const baseYear = areas.find((a) => a.projection)?.projection?.current_year ?? 0;

  return {
    generatedAt: new Date(),
    codigoInep,
    schoolName,
    baseYear,
    overallMastery: tri.overall_tri_mastery,
    masteryInterpretation: tri.insights.mastery_interpretation,
    recommendation: tri.insights.recommendation,
    areas,
  };
}
