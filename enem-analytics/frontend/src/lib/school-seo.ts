const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PublicSchoolSeoSummary {
  codigo_inep: string;
  nome_escola: string;
  uf: string | null;
  municipio: string | null;
  tipo_escola: string | null;
  anos_participacao: number;
  ultimo_ano: number | null;
  ranking_brasil: number | null;
  ranking_uf: number | null;
  nota_media: number | null;
  nota_cn: number | null;
  nota_ch: number | null;
  nota_lc: number | null;
  nota_mt: number | null;
  nota_redacao: number | null;
}

export interface SchoolSeoIndexResponse {
  ano: number;
  offset: number;
  limit: number;
  total: number;
  schools: Array<{ codigo_inep: string; ano: number }>;
}

export async function getPublicSchoolSeoSummary(
  codigoInep: string,
): Promise<PublicSchoolSeoSummary | null> {
  const response = await fetch(`${API_BASE}/api/schools/${encodeURIComponent(codigoInep)}/summary`, {
    next: { revalidate: 86_400 },
  });

  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`School summary request failed: ${response.status}`);
  return response.json() as Promise<PublicSchoolSeoSummary>;
}

export async function getSchoolSeoIndex(offset = 0, limit = 5000): Promise<SchoolSeoIndexResponse> {
  const response = await fetch(`${API_BASE}/api/schools/seo-index?offset=${offset}&limit=${limit}`, {
    next: { revalidate: 86_400 },
  });

  if (!response.ok) throw new Error(`School SEO index request failed: ${response.status}`);
  return response.json() as Promise<SchoolSeoIndexResponse>;
}
