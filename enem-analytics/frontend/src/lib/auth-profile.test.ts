import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  fetchValidatedProfile,
  ProfileValidationError,
  SESSION_EXPIRED_MESSAGE,
} from './auth-profile';

const REAL_PROFILE = {
  id: 'perfil-xtri',
  email: 'teste@xtri.online',
  codigo_inep: '31350664',
  nome_escola: 'Colégio Bernoulli',
  is_admin: false,
  is_active: true,
};

const API_BASE = 'https://api.rankingenem.com';

function response(status: number, body?: unknown): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
  });
}

afterEach(() => {
  vi.useRealTimers();
});

describe('fetchValidatedProfile', () => {
  it('retorna somente o perfil validado pelo backend', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response(200, REAL_PROFILE));

    await expect(fetchValidatedProfile('token', { apiBase: API_BASE, fetchImpl })).resolves.toEqual(
      REAL_PROFILE,
    );
    expect(fetchImpl).toHaveBeenCalledOnce();
  });

  it('trata 401 como sessão expirada sem fallback nem retry', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response(401));

    await expect(fetchValidatedProfile('token', { apiBase: API_BASE, fetchImpl })).rejects.toMatchObject({
      kind: 'session_expired',
      message: SESSION_EXPIRED_MESSAGE,
    });
    expect(fetchImpl).toHaveBeenCalledOnce();
  });

  it('trata 403 como acesso desativado sem confundir com sessão expirada', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response(403));

    await expect(fetchValidatedProfile('token', { apiBase: API_BASE, fetchImpl })).rejects.toMatchObject({
      kind: 'access_denied',
      message: 'Seu acesso foi desativado. Fale com o suporte da XTRI.',
    });
    expect(fetchImpl).toHaveBeenCalledOnce();
  });

  it('rejeita perfil inativo mesmo se a resposta for 200', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      response(200, { ...REAL_PROFILE, is_active: false }),
    );

    await expect(fetchValidatedProfile('token', { apiBase: API_BASE, fetchImpl })).rejects.toMatchObject({
      kind: 'access_denied',
    });
  });

  it('classifica falhas repetidas do backend como indisponibilidade', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(response(503));

    await expect(
      fetchValidatedProfile('token', {
        apiBase: API_BASE,
        fetchImpl,
        timeoutsMs: [100, 100],
      }),
    ).rejects.toMatchObject({ kind: 'unavailable' });
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it('classifica timeout como indisponibilidade', async () => {
    vi.useFakeTimers();
    const fetchImpl = vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'));
        });
      }),
    ) as typeof fetch;

    const validation = fetchValidatedProfile('token', {
      apiBase: API_BASE,
      fetchImpl,
      timeoutsMs: [10],
    });
    const expectation = expect(validation).rejects.toBeInstanceOf(ProfileValidationError);

    await vi.advanceTimersByTimeAsync(10);
    await expectation;
  });

  it('recupera quando uma nova tentativa recebe o perfil válido', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(response(503))
      .mockResolvedValueOnce(response(200, REAL_PROFILE));

    await expect(
      fetchValidatedProfile('token', {
        apiBase: API_BASE,
        fetchImpl,
        timeoutsMs: [100, 100],
      }),
    ).resolves.toEqual(REAL_PROFILE);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});
