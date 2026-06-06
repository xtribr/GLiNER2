'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';
import Image from 'next/image';
import Sidebar from './Sidebar';
import { useAuth } from '@/lib/auth-context';
import { useSidebar } from '@/lib/sidebar-context';
import { LogOut, School } from 'lucide-react';

// Rotas públicas (vitrine / funil). Acessíveis SEM sessão.
// A página de detalhe /schools/[inep] permanece gated (resumo público virá em etapa
// própria); aqui liberamos a vitrine `/`, a lista `/schools`, o cadastro e o login.
function isPublicPath(pathname: string | null): boolean {
  if (!pathname) return false;
  if (pathname === '/') return true;
  if (pathname === '/schools') return true;
  // Detalhe da escola (/schools/{inep}) = resumo público; /roadmap segue gated.
  if (/^\/schools\/[^/]+$/.test(pathname)) return true;
  if (pathname.startsWith('/cadastro')) return true;
  if (pathname === '/login') return true;
  return false;
}

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, session, isAdmin, isLoading, logout } = useAuth();
  const { collapsed } = useSidebar();
  const isLoginPage = pathname === '/login';
  const isPublic = isPublicPath(pathname);

  useEffect(() => {
    if (isLoading) return;

    // Visitante deslogado: liberar rotas públicas, mandar o resto para login.
    if (!session) {
      if (!isPublic) {
        router.push('/login');
      }
      return;
    }

    // Sessão existe mas o perfil ainda está carregando.
    if (!user) return;

    // Modelo B (fail-closed): não-admin sem codigo_inep é um perfil malformado
    // e não deve navegar o app — manda para o login em vez de renderizar o shell.
    if (!isAdmin && !user.codigo_inep) {
      router.push('/login');
      return;
    }

    // Modelo B: usuário-escola (não admin) fica restrito ao próprio painel,
    // inclusive longe da vitrine pública e da lista.
    if (!isAdmin) {
      const allowedPaths = [
        `/schools/${user.codigo_inep}`,
        `/schools/${user.codigo_inep}/roadmap`,
      ];
      const isAllowed = allowedPaths.some((p) => pathname?.startsWith(p));

      if (!isAllowed) {
        router.push(`/schools/${user.codigo_inep}`);
      }
    }
  }, [isLoading, isAdmin, isPublic, pathname, router, session, user]);

  // Página de login: sem chrome do app.
  if (isLoginPage) {
    return <>{children}</>;
  }

  // Rota pública sem sessão (ou ainda carregando a auth): renderiza a vitrine
  // imediatamente — ela traz o próprio header/hero e não depende de login.
  if (isPublic && !session) {
    return <>{children}</>;
  }

  // Carregando sessão/perfil (rotas autenticadas).
  if (isLoading || (session && !user)) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-50">
        <div className="h-8 w-8 border-4 border-sky-400 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Rota gated sem sessão → o effect acima já redireciona para /login.
  if (!session) {
    return null;
  }

  // Usuário-escola (não admin): layout enxuto, sem sidebar.
  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-slate-50">
        {/* Minimal Header for School Users */}
        <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo and School Name */}
              <div className="flex items-center gap-4">
                <div className="h-10 w-10 bg-gradient-to-br from-sky-400 to-orange-500 rounded-xl flex items-center justify-center p-1">
                  <Image src="/logo-xtri.png" alt="X-TRI" width={32} height={32} />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-gray-900">X-TRI Escolas</h1>
                  {user && (
                    <p className="text-xs text-gray-500 flex items-center gap-1">
                      <School className="h-3 w-3" />
                      {user.nome_escola}
                    </p>
                  )}
                </div>
              </div>

              {/* Logout Button */}
              <button
                onClick={logout}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-xl transition-colors"
              >
                <LogOut className="h-4 w-4" />
                Sair
              </button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
      </div>
    );
  }

  // Admin: layout completo com sidebar.
  return (
    <div className="flex bg-slate-50 min-h-screen">
      <Sidebar />
      <main className={`flex-1 min-h-screen transition-all duration-300 ease-in-out ${
        collapsed ? 'ml-16' : 'ml-56'
      }`}>
        {children}
      </main>
    </div>
  );
}
