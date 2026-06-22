-- Correção pontual de nome escolar ENEM 2025.
-- O arquivo DADOS/RESULTADOS_2025.csv do INEP não traz nome da escola;
-- esta escola nova não estava no Censo 2024 usado no enriquecimento inicial.
-- Fontes consultadas para o código INEP 24091260:
-- - https://qedu.org.br/escola/24091260-colegio-simples
-- - https://buscaescola.com.br/escolas/rn/mossoro/24091260-colegio-simples.html

BEGIN;

UPDATE public.enem_results
SET
  nome_escola = 'COLEGIO SIMPLES',
  inep_nome = '24091260-COLEGIO SIMPLES',
  updated_at = NOW()
WHERE ano = 2025
  AND codigo_inep = '24091260'
  AND (
    nome_escola IS DISTINCT FROM 'COLEGIO SIMPLES'
    OR inep_nome IS DISTINCT FROM '24091260-COLEGIO SIMPLES'
  );

COMMIT;
