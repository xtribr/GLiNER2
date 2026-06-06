-- =============================================================
-- MIGRATION 007: RPC get_uf_stats — média TRI agregada por UF
-- Alimenta o mapa de calor (choropleth) da vitrine pública.
-- Date: 2026-06-06
-- =============================================================
-- Agrega o último ano disponível. Read-only; segura para acesso público
-- (dados derivados de microdados públicos do INEP).

CREATE OR REPLACE FUNCTION public.get_uf_stats()
RETURNS TABLE(uf text, media numeric, escolas bigint)
LANGUAGE sql
STABLE
SECURITY DEFINER
AS $$
  SELECT e.uf,
         round(avg(e.media_geral)::numeric, 1) AS media,
         count(DISTINCT e.codigo_inep)         AS escolas
  FROM public.enem_results e
  WHERE e.ano = (SELECT max(ano) FROM public.enem_results)
    AND e.uf IS NOT NULL
    AND e.media_geral IS NOT NULL
  GROUP BY e.uf
  ORDER BY media DESC;
$$;

GRANT EXECUTE ON FUNCTION public.get_uf_stats() TO anon, authenticated, service_role;

-- =============================================================
-- END OF MIGRATION 007
-- =============================================================
