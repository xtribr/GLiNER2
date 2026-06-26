import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Redação ENEM 2025 — a dupla correção em números | X-TRI',
  description:
    'As 5 competências caíram em 2025. Análise anônima da redação do ENEM: distribuição de notas, divergência entre avaliadores por competência e por estado. Microdados oficiais do INEP.',
  openGraph: {
    title: 'Redação ENEM 2025 — a dupla correção em números',
    description:
      'As 5 competências caíram em 2025; a C2 teve a maior queda (−16,0). Veja a análise completa da redação por avaliador, competência e estado.',
    type: 'article',
  },
  twitter: { card: 'summary_large_image' },
};

export default function RedacaoLayout({ children }: { children: React.ReactNode }) {
  return children;
}
