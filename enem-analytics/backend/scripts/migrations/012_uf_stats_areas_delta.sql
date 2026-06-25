-- 012: estende get_uf_stats() com as 5 áreas + média do ano anterior (delta YoY).
-- Usado pelo tooltip rico do mapa da vitrine (BrazilChoropleth).
-- Aditivo: mantém as colunas uf/media/escolas (contrato atual intacto).
-- DROP antes do CREATE: RETURNS TABLE muda, e CREATE OR REPLACE não troca a assinatura.

DROP FUNCTION IF EXISTS public.get_uf_stats();

CREATE OR REPLACE FUNCTION public.get_uf_stats()
  RETURNS TABLE(
    uf            text,
    media         numeric,
    escolas       bigint,
    media_cn      numeric,
    media_ch      numeric,
    media_lc      numeric,
    media_mt      numeric,
    media_redacao numeric,
    media_prev    numeric,
    ano           integer,
    ano_prev      integer
  )
  LANGUAGE sql
  STABLE
  SECURITY DEFINER
AS $function$
  WITH latest AS (
    SELECT max(ano) AS y FROM public.enem_results
  ),
  prev AS (
    SELECT max(ano) AS y FROM public.enem_results
    WHERE ano < (SELECT y FROM latest)
  ),
  cur AS (
    SELECT e.uf,
           round(avg(e.media_geral)::numeric, 1)   AS media,
           count(DISTINCT e.codigo_inep)           AS escolas,
           round(avg(e.media_cn)::numeric, 1)      AS media_cn,
           round(avg(e.media_ch)::numeric, 1)      AS media_ch,
           round(avg(e.media_lc)::numeric, 1)      AS media_lc,
           round(avg(e.media_mt)::numeric, 1)      AS media_mt,
           round(avg(e.media_redacao)::numeric, 1) AS media_redacao
    FROM public.enem_results e
    WHERE e.ano = (SELECT y FROM latest)
      AND e.uf IS NOT NULL
      AND e.media_geral IS NOT NULL
    GROUP BY e.uf
  ),
  prv AS (
    SELECT e.uf, round(avg(e.media_geral)::numeric, 1) AS media_prev
    FROM public.enem_results e
    WHERE e.ano = (SELECT y FROM prev)
      AND e.uf IS NOT NULL
      AND e.media_geral IS NOT NULL
    GROUP BY e.uf
  )
  SELECT cur.uf, cur.media, cur.escolas,
         cur.media_cn, cur.media_ch, cur.media_lc, cur.media_mt, cur.media_redacao,
         prv.media_prev,
         (SELECT y FROM latest)::int AS ano,
         (SELECT y FROM prev)::int   AS ano_prev
  FROM cur
  LEFT JOIN prv ON prv.uf = cur.uf
  ORDER BY cur.media DESC;
$function$;
