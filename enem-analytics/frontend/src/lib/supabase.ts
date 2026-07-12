/**
 * Supabase client and auth utilities for the ranking ENEM frontend.
 */

import { createClient } from '@supabase/supabase-js';
import type { Session } from '@supabase/supabase-js';

import { fetchValidatedProfile } from './auth-profile';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Environment variables
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

// Validate environment
if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.warn('Supabase environment variables not set. Auth will not work.');
}

// Create Supabase client
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
  },
});

// User profile type (matches Supabase profiles table)
export interface UserProfile {
  id: string;
  codigo_inep: string;
  nome_escola: string;
  is_admin: boolean;
  is_active?: boolean;
  created_at?: string;
  updated_at?: string;
}

// Extended user type with email from auth
export interface User {
  id: string;
  email: string;
  codigo_inep: string;
  nome_escola: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
}

/**
 * Sign in with email and password
 */
export async function signIn(email: string, password: string) {
  const isLocalSupabase = SUPABASE_URL.includes('127.0.0.1') || SUPABASE_URL.includes('localhost');

  if (isLocalSupabase) {
    try {
      await fetch(`${SUPABASE_URL}/auth/v1/health`, {
        method: 'GET',
        cache: 'no-store',
      });
    } catch {
      throw new Error(
        'Não foi possível conectar ao Supabase Auth local. Inicie o Supabase local ou ajuste NEXT_PUBLIC_SUPABASE_URL.'
      );
    }
  }

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    throw new Error(error.message);
  }

  return data;
}

/**
 * Sign out the current user
 */
export async function signOut() {
  const { error } = await supabase.auth.signOut();
  if (error) {
    throw new Error(error.message);
  }
}

/**
 * Get current session
 */
export async function getSession() {
  const { data, error } = await supabase.auth.getSession();
  if (error) {
    throw new Error(error.message);
  }
  return data.session;
}

/**
 * Get user from session using the backend as the source of truth.
 */
export async function getUserFromSession(session: Session): Promise<User> {
  const profile = await fetchValidatedProfile(session.access_token, { apiBase: API_BASE });
  return {
    id: profile.id,
    email: profile.email || session.user.email || '',
    codigo_inep: profile.codigo_inep,
    nome_escola: profile.nome_escola,
    is_admin: profile.is_admin,
    is_active: profile.is_active,
    created_at: session.user.created_at,
  };
}

/**
 * Get access token for API calls
 */
export async function getAccessToken(): Promise<string | null> {
  const { data: { session } } = await supabase.auth.getSession();
  return session?.access_token || null;
}

/**
 * Request password reset email
 */
export async function resetPassword(email: string) {
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${window.location.origin}/reset-password`,
  });

  if (error) {
    throw new Error(error.message);
  }
}

/**
 * Update password (after reset)
 */
export async function updatePassword(newPassword: string) {
  const { error } = await supabase.auth.updateUser({
    password: newPassword,
  });

  if (error) {
    throw new Error(error.message);
  }
}

/**
 * Subscribe to auth state changes
 */
export function onAuthStateChange(
  callback: (event: string, session: unknown) => void
) {
  return supabase.auth.onAuthStateChange(callback);
}
