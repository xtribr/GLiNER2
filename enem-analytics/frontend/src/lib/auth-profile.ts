export interface ValidatedAuthProfile {
  id: string;
  email: string;
  codigo_inep: string;
  nome_escola: string;
  is_admin: boolean;
  is_active: boolean;
}

export type ProfileValidationFailure = 'session_expired' | 'access_denied' | 'unavailable';

export const SESSION_EXPIRED_MESSAGE =
  'Sua sessão expirou. Entre novamente para acessar sua área.';

export class ProfileValidationError extends Error {
  readonly kind: ProfileValidationFailure;

  constructor(kind: ProfileValidationFailure, message: string) {
    super(message);
    this.name = 'ProfileValidationError';
    this.kind = kind;
  }
}

export function isProfileValidationError(error: unknown): error is ProfileValidationError {
  return error instanceof ProfileValidationError;
}

interface ProfileRequestOptions {
  apiBase: string;
  fetchImpl?: typeof fetch;
  timeoutsMs?: number[];
}

const DEFAULT_TIMEOUTS_MS = [5000, 3000];

function isValidatedProfile(value: unknown): value is ValidatedAuthProfile {
  if (!value || typeof value !== 'object') return false;
  const profile = value as Record<string, unknown>;
  return (
    typeof profile.id === 'string' &&
    typeof profile.email === 'string' &&
    typeof profile.codigo_inep === 'string' &&
    typeof profile.nome_escola === 'string' &&
    typeof profile.is_admin === 'boolean' &&
    typeof profile.is_active === 'boolean'
  );
}

async function fetchWithTimeout(
  fetchImpl: typeof fetch,
  url: string,
  accessToken: string,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetchImpl(url, {
      headers: { Authorization: `Bearer ${accessToken}` },
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function fetchValidatedProfile(
  accessToken: string,
  {
    apiBase,
    fetchImpl = fetch,
    timeoutsMs = DEFAULT_TIMEOUTS_MS,
  }: ProfileRequestOptions,
): Promise<ValidatedAuthProfile> {
  const attempts = timeoutsMs.length ? timeoutsMs : DEFAULT_TIMEOUTS_MS;

  for (let index = 0; index < attempts.length; index += 1) {
    try {
      const response = await fetchWithTimeout(
        fetchImpl,
        `${apiBase}/api/auth/me`,
        accessToken,
        attempts[index],
      );

      if (response.status === 401) {
        throw new ProfileValidationError('session_expired', SESSION_EXPIRED_MESSAGE);
      }

      if (response.status === 403) {
        throw new ProfileValidationError(
          'access_denied',
          'Seu acesso foi desativado. Fale com o suporte da XTRI.',
        );
      }

      if (response.ok) {
        const profile: unknown = await response.json();
        if (!isValidatedProfile(profile) || !profile.is_active) {
          throw new ProfileValidationError(
            'access_denied',
            'Seu perfil está ausente ou inativo. Fale com o suporte da XTRI.',
          );
        }
        return profile;
      }
    } catch (error) {
      if (isProfileValidationError(error)) {
        throw error;
      }
    }
  }

  throw new ProfileValidationError(
    'unavailable',
    'Não foi possível validar seu perfil agora. Verifique sua conexão e tente novamente.',
  );
}
