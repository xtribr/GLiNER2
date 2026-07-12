'use client';

import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import type { Session } from '@supabase/supabase-js';

import { isProfileValidationError } from './auth-profile';
import { supabase, getUserFromSession, signIn, signOut } from './supabase';
import type { User } from './supabase';

export interface AuthError {
  kind: 'session_expired' | 'access_denied' | 'unavailable';
  message: string;
}

interface AuthContextType {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  authError: AuthError | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [authError, setAuthError] = useState<AuthError | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [isValidatingProfile, setIsValidatingProfile] = useState(false);
  const sessionRef = useRef<Session | null>(null);
  const validationIdRef = useRef(0);
  const inFlightValidationRef = useRef<{
    accessToken: string;
    promise: Promise<User>;
  } | null>(null);

  const validateSession = useCallback(async (currentSession: Session): Promise<User> => {
    const existing = inFlightValidationRef.current;
    if (existing?.accessToken === currentSession.access_token) {
      return existing.promise;
    }

    const validationId = validationIdRef.current + 1;
    validationIdRef.current = validationId;
    setIsValidatingProfile(true);

    const request = getUserFromSession(currentSession);
    inFlightValidationRef.current = {
      accessToken: currentSession.access_token,
      promise: request,
    };

    try {
      const currentUser = await request;
      if (validationIdRef.current === validationId) {
        setUser(currentUser);
        setAuthError(null);
      }
      return currentUser;
    } catch (error) {
      const issue: AuthError = isProfileValidationError(error)
        ? { kind: error.kind, message: error.message }
        : {
            kind: 'unavailable',
            message: 'Não foi possível validar seu perfil agora. Tente novamente.',
          };

      if (validationIdRef.current === validationId) {
        setUser(null);
        setAuthError(issue);
      }

      if (issue.kind === 'session_expired' || issue.kind === 'access_denied') {
        try {
          await signOut();
        } catch {
          // A sessão local ainda é invalidada abaixo mesmo se o logout remoto falhar.
        } finally {
          sessionRef.current = null;
          if (validationIdRef.current === validationId) {
            setSession(null);
            setAuthError(issue);
          }
        }
      }
      throw error;
    } finally {
      if (inFlightValidationRef.current?.accessToken === currentSession.access_token) {
        inFlightValidationRef.current = null;
      }
      if (validationIdRef.current === validationId) {
        setIsValidatingProfile(false);
      }
    }
  }, []);

  const refreshUser = useCallback(async () => {
    const currentSession = sessionRef.current;
    if (!currentSession) {
      setUser(null);
      return;
    }
    try {
      await validateSession(currentSession);
    } catch {
      // validateSession já materializa o erro seguro para a interface.
    }
  }, [validateSession]);

  useEffect(() => {
    let mounted = true;

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, newSession) => {
        if (!mounted) return;
        sessionRef.current = newSession;
        setSession(newSession);

        if (event === 'INITIAL_SESSION' || event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
          if (newSession) {
            void validateSession(newSession).catch(() => undefined);
          } else {
            setUser(null);
            setAuthError(null);
          }
        } else if (event === 'SIGNED_OUT') {
          validationIdRef.current += 1;
          setUser(null);
          setIsValidatingProfile(false);
        }

        setIsInitializing(false);
      }
    );

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, [validateSession]);

  const login = async (email: string, password: string) => {
    try {
      const { session: signedInSession } = await signIn(email, password);
      if (!signedInSession) throw new Error('Não foi possível iniciar a sessão.');
      sessionRef.current = signedInSession;
      setSession(signedInSession);
      await validateSession(signedInSession);
    } catch (error) {
      if (error instanceof Error && error.message === 'Invalid login credentials') {
        throw new Error('Email ou senha incorretos');
      }
      throw error instanceof Error ? error : new Error('Erro ao fazer login');
    }
  };

  const logout = async () => {
    try {
      await signOut();
    } finally {
      validationIdRef.current += 1;
      sessionRef.current = null;
      setUser(null);
      setSession(null);
      setAuthError(null);
      setIsValidatingProfile(false);
    }
  };

  const isLoading = isInitializing || isValidatingProfile;
  const isAuthenticated = Boolean(session && user && !isLoading && !authError);
  const isAdmin = isAuthenticated ? (user?.is_admin ?? false) : false;

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        isLoading,
        isAuthenticated,
        isAdmin,
        authError,
        login,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
