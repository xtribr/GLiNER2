import Link from "next/link";
import Image from "next/image";
import { ArrowLeft } from "lucide-react";

export default function LegalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-white">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-5 py-4 md:px-8">
          <Link
            href="/"
            className="flex items-center gap-3 text-slate-800 transition hover:text-slate-950"
          >
            <Image
              src='/logo-xtri.png'
              alt="XTRI"
              width={28}
              height={30}
              className="h-7 w-auto"
            />
            <span className="text-sm font-semibold">
              Mentoria ENEM
            </span>
          </Link>
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-slate-500 transition hover:text-slate-800"
          >
            <ArrowLeft className="h-4 w-4" />
            Voltar
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-5 py-12 md:px-8 md:py-16">
        {children}
      </main>

      <footer className="border-t border-slate-200 bg-slate-50 px-5 py-8 md:px-8">
        <div className="mx-auto flex max-w-4xl flex-col items-center gap-4 text-center text-sm text-slate-500 sm:flex-row sm:justify-between sm:text-left">
          <p>© 2026 XTRI EdTech. Todos os direitos reservados.</p>
          <nav className="flex gap-4">
            <Link
              href="/termos-de-uso"
              className="hover:text-slate-800 transition"
            >
              Termos de Uso
            </Link>
            <Link
              href="/politica-de-privacidade"
              className="hover:text-slate-800 transition"
            >
              Privacidade
            </Link>
            <Link
              href="/politica-de-compliance-ia"
              className="hover:text-slate-800 transition"
            >
              Compliance IA
            </Link>
          </nav>
        </div>
      </footer>
    </div>
  );
}
