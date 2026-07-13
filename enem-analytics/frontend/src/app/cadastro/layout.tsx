import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Cadastro gratuito',
  robots: { index: false, follow: true },
};

export default function CadastroLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
