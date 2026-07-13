import type { MetadataRoute } from 'next';
import { getSchoolSeoIndex } from '@/lib/school-seo';

const BASE_URL = 'https://app.rankingenem.com';
const BATCH_SIZE = 5000;

export const revalidate = 86_400;

const corePages: MetadataRoute.Sitemap = [
  { url: BASE_URL, changeFrequency: 'weekly', priority: 1 },
  { url: `${BASE_URL}/redacao`, changeFrequency: 'monthly', priority: 0.8 },
  { url: `${BASE_URL}/termos-de-uso`, changeFrequency: 'yearly', priority: 0.2 },
  { url: `${BASE_URL}/politica-de-privacidade`, changeFrequency: 'yearly', priority: 0.2 },
  { url: `${BASE_URL}/politica-de-compliance-ia`, changeFrequency: 'yearly', priority: 0.2 },
];

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  try {
    const firstBatch = await getSchoolSeoIndex(0, BATCH_SIZE);
    const remainingOffsets = Array.from(
      { length: Math.max(0, Math.ceil(firstBatch.total / BATCH_SIZE) - 1) },
      (_, index) => (index + 1) * BATCH_SIZE,
    );
    const remainingBatches = await Promise.all(
      remainingOffsets.map((offset) => getSchoolSeoIndex(offset, BATCH_SIZE)),
    );
    const schools = [firstBatch, ...remainingBatches].flatMap((batch) => batch.schools);

    return [
      ...corePages,
      ...schools.map(({ codigo_inep }) => ({
        url: `${BASE_URL}/ranking-enem/escola/${codigo_inep}`,
        changeFrequency: 'yearly' as const,
        priority: 0.6,
      })),
    ];
  } catch {
    return corePages;
  }
}
