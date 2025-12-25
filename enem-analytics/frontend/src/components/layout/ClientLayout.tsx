'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';
import Sidebar from './Sidebar';
import { useAuth } from '@/lib/auth-context';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isAuthenticated, isAdmin, isLoading } = useAuth();
  const isLoginPage = pathname === '/login';

  useEffect(() => {
    // Don't redirect while loading or on login page
    if (isLoading || isLoginPage) return;

    // Redirect to login if not authenticated
    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    // For non-admin users, redirect to their school page
    if (!isAdmin && user?.codigo_inep) {
      const allowedPaths = [
        `/schools/${user.codigo_inep}`,
        `/schools/${user.codigo_inep}/roadmap`,
      ];
      const isAllowed = allowedPaths.some(p => pathname?.startsWith(p));

      if (!isAllowed) {
        router.push(`/schools/${user.codigo_inep}`);
      }
    }
  }, [isLoading, isAuthenticated, isAdmin, user, pathname, router, isLoginPage]);

  if (isLoginPage) {
    return <>{children}</>;
  }

  // Show loading while checking auth
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-50">
        <div className="h-8 w-8 border-4 border-sky-400 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Don't render anything if not authenticated (will redirect)
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="flex">
      <Sidebar />
      <main className="flex-1 ml-64 min-h-screen transition-all duration-300 p-6">
        {children}
      </main>
    </div>
  );
}
