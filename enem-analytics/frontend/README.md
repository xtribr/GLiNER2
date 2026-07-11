# Ranking ENEM Frontend

Frontend Next.js da plataforma `app.rankingenem.com`.

## Stack

- Next.js App Router
- React
- Tailwind CSS
- Supabase Auth

## Variáveis de ambiente

Crie `.env.local` em desenvolvimento ou configure no Coolify:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

`NEXT_PUBLIC_*` é embutido no build. Qualquer alteração exige novo deploy do frontend.

## Desenvolvimento local

```bash
cd enem-analytics/frontend
pnpm install
pnpm dev
```

## Testes

```bash
pnpm test          # unitários (Vitest)
pnpm test:e2e      # Playwright; requer frontend e backend locais
pnpm build         # TypeScript + build de produção
```

Os cenários autenticados do Playwright não guardam credenciais no repositório.
Configure `E2E_ADMIN_EMAIL`, `E2E_ADMIN_PASSWORD`, `E2E_SCHOOL_EMAIL`,
`E2E_SCHOOL_PASSWORD`, `E2E_SCHOOL_INEP` e `E2E_OTHER_SCHOOL_INEP` no ambiente
local/CI. Os dois códigos INEP devem existir na base real.

## Docker / Coolify

- Base Directory: `/enem-analytics/frontend`
- Dockerfile Location: `/Dockerfile`
- Port Expose: `3000`

No Coolify, configure as variáveis acima e faça novo deploy sempre que mudar:

- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`

## Fluxo de autenticação

- Login é feito diretamente pelo Supabase Auth.
- O frontend lê a sessão do Supabase e busca o perfil do usuário.
- Usuário admin acessa o dashboard completo.
- Usuário escola fica restrito à própria escola e ao roadmap.
