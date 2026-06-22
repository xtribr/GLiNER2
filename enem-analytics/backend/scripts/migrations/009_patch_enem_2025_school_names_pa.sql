-- Correção pontual de nome escolar ENEM 2025.
-- O arquivo DADOS/RESULTADOS_2025.csv do INEP não traz nome da escola;
-- esta escola nova não estava no Censo 2024 usado no enriquecimento inicial.
-- Fontes consultadas para o código INEP 15181081:
-- - https://qedu.org.br/escola/15181081-colegio-integrado-in
-- - https://buscaescola.com.br/escolas/pa/belem/15181081-colegio-integrado-in.html
-- - https://escolasbrasil.org/para/belem/15181081

BEGIN;

UPDATE public.enem_results
SET
  nome_escola = 'COLEGIO INTEGRADO IN',
  inep_nome = '15181081-COLEGIO INTEGRADO IN',
  updated_at = NOW()
WHERE ano = 2025
  AND codigo_inep = '15181081'
  AND (
    nome_escola IS DISTINCT FROM 'COLEGIO INTEGRADO IN'
    OR inep_nome IS DISTINCT FROM '15181081-COLEGIO INTEGRADO IN'
  );

COMMIT;
