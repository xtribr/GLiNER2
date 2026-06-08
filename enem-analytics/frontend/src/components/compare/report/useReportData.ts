import type { ReportData, ComparisonYearRow, ReportSchoolMeta } from './types';
import type { DiagnosisComparisonResult, SchoolHistory } from '@/lib/api';

interface BuildArgs {
  diagnosis: DiagnosisComparisonResult;
  history1?: SchoolHistory; history2?: SchoolHistory;
  comparison?: { comparison: { ano: number; escola1: { nota_media: number|null; ranking: number|null }|null; escola2: { nota_media: number|null; ranking: number|null }|null }[] };
  nameA: string; nameB: string;
  ufA?: string|null; ufB?: string|null;
}

export function buildPhase1ReportData(args: BuildArgs): ReportData {
  const { diagnosis, history1, history2, comparison } = args;
  const lastA = history1?.history?.at(-1);
  const lastB = history2?.history?.at(-1);
  const meta = (info: DiagnosisComparisonResult['school_1'], name: string, uf: string|null|undefined, last: typeof lastA): ReportSchoolMeta => ({
    codigo_inep: info.codigo_inep,
    nome_escola: name,                          // NOME REAL
    uf: uf ?? info.info.localizacao ?? null,
    cidade: info.info.localizacao ?? null,
    tipo_escola: info.info.tipo_escola,
    porte_label: info.info.porte != null ? String(info.info.porte) : null,
    nota_media: last?.nota_media ?? null,
    ranking_brasil: last?.ranking_brasil ?? null,
    ranking_uf: last?.ranking_uf ?? null,
    overall_health: info.overall_health ?? null,
  });
  const history: ComparisonYearRow[] = (comparison?.comparison ?? []).map((y) => ({
    ano: y.ano,
    a_media: y.escola1?.nota_media ?? null, a_rank: y.escola1?.ranking ?? null,
    b_media: y.escola2?.nota_media ?? null, b_rank: y.escola2?.ranking ?? null,
  }));
  return {
    generatedAt: new Date(),
    baseYear: diagnosis.school_1.info.ano,
    schoolA: meta(diagnosis.school_1, args.nameA, args.ufA, lastA),
    schoolB: meta(diagnosis.school_2, args.nameB, args.ufB, lastB),
    diagnosis,
    history,
  };
}
