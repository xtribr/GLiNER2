-- =============================================================
-- MIGRATION 006: Lead capture fields on profiles
-- Suporte ao cadastro self-service (vitrine pública / funil de leads)
-- Date: 2026-06-06
-- =============================================================
-- Colunas aditivas — não quebram perfis existentes (admins/escolas).
-- A inserção de profiles continua via service key (bypassa RLS), igual ao
-- create_admin.py / POST /api/admin/users; nenhuma policy nova é necessária.

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS nome_contato   text,
  ADD COLUMN IF NOT EXISTS cargo          text,
  ADD COLUMN IF NOT EXISTS telefone       text,
  ADD COLUMN IF NOT EXISTS email_verified boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS origem         text;

COMMENT ON COLUMN public.profiles.nome_contato   IS 'Nome do responsável (lead) informado no cadastro';
COMMENT ON COLUMN public.profiles.cargo          IS 'Cargo na escola (Diretor/Coordenador/Professor/...)';
COMMENT ON COLUMN public.profiles.telefone       IS 'Celular/WhatsApp do lead';
COMMENT ON COLUMN public.profiles.email_verified IS 'Flag não-bloqueante: e-mail confirmado via link do welcome';
COMMENT ON COLUMN public.profiles.origem         IS 'Origem do cadastro (ex.: cadastro_publico) — distingue lead de admin';

-- =============================================================
-- END OF MIGRATION 006
-- =============================================================
