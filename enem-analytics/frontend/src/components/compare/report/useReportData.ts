import type { ReportData, ComparisonYearRow, ReportSchoolMeta, ProjectionRow, RedacaoCompRow, SkillRow } from './types';
import type { DiagnosisComparisonResult, SchoolHistory, TRIAreaProjection } from '@/lib/api';
import { api } from '@/lib/api';
import { buildProjectionRows, buildRedacaoRows, buildSkillRows } from './reportMetrics';
import type { StudyFocusResult, SchoolSkillsResult } from './reportMetrics';

interface BuildArgs {
  diagnosis: DiagnosisComparisonResult;
  history1?: SchoolHistory; history2?: SchoolHistory;
  comparison?: { comparison: { ano: number; escola1: { nota_media: number|null; ranking: number|null }|null; escola2: { nota_media: number|null; ranking: number|null }|null }[] };
  nameA: string; nameB: string;
  ufA?: string|null; ufB?: string|null;
}

// Map diagnosis area codes (e.g. 'CH', 'RED', 'REDACAO') → backend lowercase (e.g. 'ch', 're')
const DIAG_TO_BACKEND: Record<string, string> = { CN: 'cn', CH: 'ch', LC: 'lc', MT: 'mt', RED: 're', REDACAO: 're', RE: 're' };
function diagAreaToBackendCode(code: string): string {
  const u = (code || '').toUpperCase();
  return DIAG_TO_BACKEND[u] ?? u.toLowerCase();
}

export async function collectProjectionRows(
  inepA: string,
  inepB: string,
  areaCodes: string[],
): Promise<ProjectionRow[]> {
  const backendCodes = areaCodes.map(diagAreaToBackendCode);

  const settledA = await Promise.allSettled(
    backendCodes.map(code => api.getAreaProjection(inepA, code)),
  );
  const settledB = await Promise.allSettled(
    backendCodes.map(code => api.getAreaProjection(inepB, code)),
  );

  const projA: TRIAreaProjection[] = settledA
    .filter((r): r is PromiseFulfilledResult<TRIAreaProjection> => r.status === 'fulfilled')
    .map(r => r.value);
  const projB: TRIAreaProjection[] = settledB
    .filter((r): r is PromiseFulfilledResult<TRIAreaProjection> => r.status === 'fulfilled')
    .map(r => r.value);

  return buildProjectionRows(projA, projB);
}

// Returns RedacaoCompRow[] (empty on total failure)
export async function collectRedacaoRows(inepA: string, inepB: string): Promise<RedacaoCompRow[]> {
  const [a, b] = await Promise.allSettled([
    api.getGlinerStudyFocus(inepA),
    api.getGlinerStudyFocus(inepB),
  ]);
  const sa: StudyFocusResult | null = a.status === 'fulfilled' ? a.value : null;
  const sb: StudyFocusResult | null = b.status === 'fulfilled' ? b.value : null;
  return buildRedacaoRows(sa, sb);
}

// Returns { a: SkillRow[]; b: SkillRow[] }
export async function collectSkills(inepA: string, inepB: string): Promise<{ a: SkillRow[]; b: SkillRow[] }> {
  const [a, b] = await Promise.allSettled([
    api.getSchoolSkills(inepA),
    api.getSchoolSkills(inepB),
  ]);
  return {
    a: buildSkillRows(a.status === 'fulfilled' ? (a.value as SchoolSkillsResult) : null),
    b: buildSkillRows(b.status === 'fulfilled' ? (b.value as SchoolSkillsResult) : null),
  };
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
