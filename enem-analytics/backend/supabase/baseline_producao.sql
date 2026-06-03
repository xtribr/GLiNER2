


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "hypopg" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "index_advisor" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE OR REPLACE FUNCTION "public"."calculate_tri_score"("theta" numeric, "mean" numeric DEFAULT 500, "sd" numeric DEFAULT 100) RETURNS numeric
    LANGUAGE "plpgsql" IMMUTABLE
    SET "search_path" TO 'public'
    AS $$
BEGIN
    RETURN ROUND(mean + (sd * theta), 2);
END;
$$;


ALTER FUNCTION "public"."calculate_tri_score"("theta" numeric, "mean" numeric, "sd" numeric) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_enem_stats"() RETURNS "jsonb"
    LANGUAGE "plpgsql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
    result jsonb;
BEGIN
    SELECT jsonb_build_object(
        'total_records', (SELECT COUNT(*) FROM public.enem_results),
        'total_schools', (SELECT COUNT(DISTINCT codigo_inep) FROM public.enem_results),
        'years', (
            SELECT COALESCE(jsonb_agg(ano ORDER BY ano), '[]'::jsonb)
            FROM (SELECT DISTINCT ano FROM public.enem_results) a
        ),
        'states', (
            SELECT COALESCE(jsonb_agg(uf ORDER BY uf), '[]'::jsonb)
            FROM (SELECT DISTINCT uf FROM public.enem_results WHERE uf IS NOT NULL) s
        ),
        'avg_scores', jsonb_build_object(
            'nota_cn',      (SELECT ROUND(AVG(media_cn)::numeric, 2)      FROM public.enem_results WHERE media_cn      IS NOT NULL),
            'nota_ch',      (SELECT ROUND(AVG(media_ch)::numeric, 2)      FROM public.enem_results WHERE media_ch      IS NOT NULL),
            'nota_lc',      (SELECT ROUND(AVG(media_lc)::numeric, 2)      FROM public.enem_results WHERE media_lc      IS NOT NULL),
            'nota_mt',      (SELECT ROUND(AVG(media_mt)::numeric, 2)      FROM public.enem_results WHERE media_mt      IS NOT NULL),
            'nota_redacao', (SELECT ROUND(AVG(media_redacao)::numeric, 2) FROM public.enem_results WHERE media_redacao IS NOT NULL),
            'nota_media',   (SELECT ROUND(AVG(media_geral)::numeric, 2)   FROM public.enem_results WHERE media_geral   IS NOT NULL)
        ),
        'avg_scores_by_year', (
            SELECT COALESCE(jsonb_object_agg(ano::text, avg_media), '{}'::jsonb)
            FROM (
                SELECT ano, ROUND(AVG(media_geral)::numeric, 2) AS avg_media
                FROM public.enem_results WHERE media_geral IS NOT NULL
                GROUP BY ano ORDER BY ano
            ) s
        ),
        'latest_year', (SELECT MAX(ano) FROM public.enem_results)
    ) INTO result;
    RETURN result;
END;
$$;


ALTER FUNCTION "public"."get_enem_stats"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_admin"() RETURNS boolean
    LANGUAGE "plpgsql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.profiles
    WHERE id = (select auth.uid()) AND is_admin = TRUE
  );
END;
$$;


ALTER FUNCTION "public"."is_admin"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."promote_enem_year_import"("p_import_job_id" "uuid", "p_year" integer, "p_expected_count" integer, "p_allow_existing_year" boolean DEFAULT false) RETURNS "jsonb"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
    v_staging_count INTEGER;
    v_duplicate_count INTEGER;
    v_existing_count INTEGER;
    v_inserted_count INTEGER;
BEGIN
    IF p_year < 2018 OR p_year > 2030 THEN
        RAISE EXCEPTION 'Ano fora do range permitido: %', p_year;
    END IF;

    IF p_expected_count < 1 THEN
        RAISE EXCEPTION 'p_expected_count deve ser maior que zero';
    END IF;

    SELECT COUNT(*)
      INTO v_staging_count
      FROM public.enem_results_import_staging
     WHERE import_job_id = p_import_job_id
       AND ano = p_year;

    IF v_staging_count <> p_expected_count THEN
        RAISE EXCEPTION 'Staging incompleto: esperado %, encontrado %',
            p_expected_count, v_staging_count;
    END IF;

    SELECT COUNT(*) - COUNT(DISTINCT codigo_inep)
      INTO v_duplicate_count
      FROM public.enem_results_import_staging
     WHERE import_job_id = p_import_job_id
       AND ano = p_year;

    IF v_duplicate_count > 0 THEN
        RAISE EXCEPTION 'Staging contem % codigo_inep duplicado(s)', v_duplicate_count;
    END IF;

    SELECT COUNT(*)
      INTO v_existing_count
      FROM public.enem_results
     WHERE ano = p_year;

    IF v_existing_count > 0 AND NOT p_allow_existing_year THEN
        RAISE EXCEPTION 'Ja existem % registros em enem_results para %',
            v_existing_count, p_year;
    END IF;

    IF p_allow_existing_year THEN
        DELETE FROM public.enem_results WHERE ano = p_year;
    END IF;

    INSERT INTO public.enem_results (
        codigo_inep,
        ano,
        nome_escola,
        uf,
        municipio,
        dependencia,
        media_cn,
        media_ch,
        media_lc,
        media_mt,
        media_redacao,
        media_geral,
        num_participantes,
        taxa_participacao,
        ranking_nacional,
        ranking_uf,
        ranking_municipio,
        localizacao,
        porte,
        porte_label,
        nota_tri_media,
        desempenho_habilidades,
        competencia_redacao_media,
        inep_nome,
        anos_participacao
    )
    SELECT
        codigo_inep,
        ano,
        nome_escola,
        uf,
        municipio,
        dependencia,
        media_cn,
        media_ch,
        media_lc,
        media_mt,
        media_redacao,
        media_geral,
        num_participantes,
        taxa_participacao,
        ranking_nacional,
        ranking_uf,
        ranking_municipio,
        localizacao,
        porte,
        porte_label,
        nota_tri_media,
        desempenho_habilidades,
        competencia_redacao_media,
        inep_nome,
        anos_participacao
      FROM public.enem_results_import_staging
     WHERE import_job_id = p_import_job_id
       AND ano = p_year
     ORDER BY ranking_nacional NULLS LAST, codigo_inep;

    GET DIAGNOSTICS v_inserted_count = ROW_COUNT;

    IF v_inserted_count <> p_expected_count THEN
        RAISE EXCEPTION 'Insercao incompleta: esperado %, inserido %',
            p_expected_count, v_inserted_count;
    END IF;

    DELETE FROM public.enem_results_import_staging
     WHERE import_job_id = p_import_job_id;

    IF to_regprocedure('public.refresh_materialized_views()') IS NOT NULL THEN
        PERFORM public.refresh_materialized_views();
    END IF;

    RETURN jsonb_build_object(
        'status', 'ok',
        'year', p_year,
        'inserted', v_inserted_count,
        'replaced_existing', p_allow_existing_year,
        'previous_count', v_existing_count
    );
END;
$$;


ALTER FUNCTION "public"."promote_enem_year_import"("p_import_job_id" "uuid", "p_year" integer, "p_expected_count" integer, "p_allow_existing_year" boolean) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."update_updated_at_column"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    SET "search_path" TO 'public'
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."update_updated_at_column"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."enem_results" (
    "id" integer NOT NULL,
    "codigo_inep" character varying(8) NOT NULL,
    "ano" integer NOT NULL,
    "nome_escola" character varying(255),
    "uf" character(2),
    "municipio" character varying(100),
    "dependencia" character varying(20),
    "media_cn" numeric(6,2),
    "media_ch" numeric(6,2),
    "media_lc" numeric(6,2),
    "media_mt" numeric(6,2),
    "media_redacao" numeric(6,2),
    "media_geral" numeric(6,2),
    "num_participantes" integer,
    "taxa_participacao" numeric(5,2),
    "ranking_nacional" integer,
    "ranking_uf" integer,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "nota_tri_media" numeric(7,2),
    "desempenho_habilidades" numeric(5,4),
    "competencia_redacao_media" numeric(8,2),
    "inep_nome" "text",
    "localizacao" character varying(20),
    "porte" integer,
    "porte_label" character varying(64),
    "anos_participacao" integer,
    "ranking_municipio" integer,
    CONSTRAINT "enem_results_ano_check" CHECK ((("ano" >= 2018) AND ("ano" <= 2030)))
);


ALTER TABLE "public"."enem_results" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."enem_results_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE "public"."enem_results_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."enem_results_id_seq" OWNED BY "public"."enem_results"."id";



CREATE TABLE IF NOT EXISTS "public"."enem_results_import_staging" (
    "import_job_id" "uuid" NOT NULL,
    "row_number" integer NOT NULL,
    "codigo_inep" character varying(8) NOT NULL,
    "ano" integer NOT NULL,
    "nome_escola" character varying(255),
    "uf" character(2),
    "municipio" character varying(100),
    "dependencia" character varying(20),
    "media_cn" numeric(6,2),
    "media_ch" numeric(6,2),
    "media_lc" numeric(6,2),
    "media_mt" numeric(6,2),
    "media_redacao" numeric(6,2),
    "media_geral" numeric(6,2),
    "num_participantes" integer,
    "taxa_participacao" numeric(5,2),
    "ranking_nacional" integer,
    "ranking_uf" integer,
    "ranking_municipio" integer,
    "localizacao" character varying(20),
    "porte" integer,
    "porte_label" "text",
    "nota_tri_media" numeric(6,2),
    "desempenho_habilidades" numeric(6,2),
    "competencia_redacao_media" numeric(6,2),
    "inep_nome" "text",
    "anos_participacao" integer,
    "created_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "enem_results_import_staging_ano_check" CHECK ((("ano" >= 2018) AND ("ano" <= 2030)))
);


ALTER TABLE "public"."enem_results_import_staging" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."profiles" (
    "id" "uuid" NOT NULL,
    "codigo_inep" character varying(8) NOT NULL,
    "nome_escola" character varying(255) NOT NULL,
    "is_admin" boolean DEFAULT false,
    "is_active" boolean DEFAULT true,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."profiles" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."school_skills" (
    "id" integer NOT NULL,
    "codigo_inep" character varying(8) NOT NULL,
    "nome_escola" character varying(255) NOT NULL,
    "area" character varying(2) NOT NULL,
    "skill_num" integer NOT NULL,
    "performance" numeric(5,2) NOT NULL,
    "descricao" "text",
    "ano" integer DEFAULT 2024 NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."school_skills" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."school_skills_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE "public"."school_skills_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."school_skills_id_seq" OWNED BY "public"."school_skills"."id";



CREATE TABLE IF NOT EXISTS "public"."schools" (
    "codigo_inep" character varying(8) NOT NULL,
    "nome" character varying(255) NOT NULL,
    "uf" character(2) NOT NULL,
    "municipio" character varying(100),
    "dependencia" character varying(20),
    "localizacao" character varying(20),
    "is_active" boolean DEFAULT true,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."schools" OWNER TO "postgres";


ALTER TABLE ONLY "public"."enem_results" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."enem_results_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."school_skills" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."school_skills_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."enem_results"
    ADD CONSTRAINT "enem_results_codigo_inep_ano_key" UNIQUE ("codigo_inep", "ano");



ALTER TABLE ONLY "public"."enem_results_import_staging"
    ADD CONSTRAINT "enem_results_import_staging_pkey" PRIMARY KEY ("import_job_id", "codigo_inep", "ano");



ALTER TABLE ONLY "public"."enem_results"
    ADD CONSTRAINT "enem_results_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_codigo_inep_key" UNIQUE ("codigo_inep");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."school_skills"
    ADD CONSTRAINT "school_skills_codigo_inep_area_skill_num_ano_key" UNIQUE ("codigo_inep", "area", "skill_num", "ano");



ALTER TABLE ONLY "public"."school_skills"
    ADD CONSTRAINT "school_skills_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."schools"
    ADD CONSTRAINT "schools_pkey" PRIMARY KEY ("codigo_inep");



CREATE INDEX "idx_enem_results_import_staging_job" ON "public"."enem_results_import_staging" USING "btree" ("import_job_id");



CREATE INDEX "idx_profiles_codigo_inep" ON "public"."profiles" USING "btree" ("codigo_inep");



CREATE INDEX "idx_school_skills_area" ON "public"."school_skills" USING "btree" ("area");



CREATE INDEX "idx_school_skills_inep" ON "public"."school_skills" USING "btree" ("codigo_inep");



CREATE OR REPLACE TRIGGER "update_enem_results_updated_at" BEFORE UPDATE ON "public"."enem_results" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_schools_updated_at" BEFORE UPDATE ON "public"."schools" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_id_fkey" FOREIGN KEY ("id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE "public"."enem_results" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "enem_results_delete" ON "public"."enem_results" FOR DELETE USING ("public"."is_admin"());



ALTER TABLE "public"."enem_results_import_staging" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "enem_results_insert" ON "public"."enem_results" FOR INSERT WITH CHECK ("public"."is_admin"());



CREATE POLICY "enem_results_select" ON "public"."enem_results" FOR SELECT USING (true);



CREATE POLICY "enem_results_update" ON "public"."enem_results" FOR UPDATE USING ("public"."is_admin"());



ALTER TABLE "public"."profiles" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "profiles_delete_policy" ON "public"."profiles" FOR DELETE USING ("public"."is_admin"());



CREATE POLICY "profiles_insert_policy" ON "public"."profiles" FOR INSERT WITH CHECK ("public"."is_admin"());



CREATE POLICY "profiles_select_policy" ON "public"."profiles" FOR SELECT USING (((( SELECT "auth"."uid"() AS "uid") = "id") OR "public"."is_admin"()));



CREATE POLICY "profiles_update_policy" ON "public"."profiles" FOR UPDATE USING (((( SELECT "auth"."uid"() AS "uid") = "id") OR "public"."is_admin"()));



ALTER TABLE "public"."school_skills" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "school_skills_delete" ON "public"."school_skills" FOR DELETE USING ("public"."is_admin"());



CREATE POLICY "school_skills_insert" ON "public"."school_skills" FOR INSERT WITH CHECK ("public"."is_admin"());



CREATE POLICY "school_skills_select" ON "public"."school_skills" FOR SELECT USING (true);



CREATE POLICY "school_skills_update" ON "public"."school_skills" FOR UPDATE USING ("public"."is_admin"());



ALTER TABLE "public"."schools" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "schools_delete" ON "public"."schools" FOR DELETE USING ("public"."is_admin"());



CREATE POLICY "schools_insert" ON "public"."schools" FOR INSERT WITH CHECK ("public"."is_admin"());



CREATE POLICY "schools_select" ON "public"."schools" FOR SELECT USING (true);



CREATE POLICY "schools_update" ON "public"."schools" FOR UPDATE USING ("public"."is_admin"());





ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";


























































































































































































GRANT ALL ON FUNCTION "public"."calculate_tri_score"("theta" numeric, "mean" numeric, "sd" numeric) TO "anon";
GRANT ALL ON FUNCTION "public"."calculate_tri_score"("theta" numeric, "mean" numeric, "sd" numeric) TO "authenticated";
GRANT ALL ON FUNCTION "public"."calculate_tri_score"("theta" numeric, "mean" numeric, "sd" numeric) TO "service_role";



GRANT ALL ON FUNCTION "public"."get_enem_stats"() TO "anon";
GRANT ALL ON FUNCTION "public"."get_enem_stats"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_enem_stats"() TO "service_role";



GRANT ALL ON FUNCTION "public"."is_admin"() TO "anon";
GRANT ALL ON FUNCTION "public"."is_admin"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_admin"() TO "service_role";



REVOKE ALL ON FUNCTION "public"."promote_enem_year_import"("p_import_job_id" "uuid", "p_year" integer, "p_expected_count" integer, "p_allow_existing_year" boolean) FROM PUBLIC;
GRANT ALL ON FUNCTION "public"."promote_enem_year_import"("p_import_job_id" "uuid", "p_year" integer, "p_expected_count" integer, "p_allow_existing_year" boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "anon";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "service_role";
























GRANT ALL ON TABLE "public"."enem_results" TO "anon";
GRANT ALL ON TABLE "public"."enem_results" TO "authenticated";
GRANT ALL ON TABLE "public"."enem_results" TO "service_role";



GRANT ALL ON SEQUENCE "public"."enem_results_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."enem_results_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."enem_results_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."enem_results_import_staging" TO "service_role";



GRANT ALL ON TABLE "public"."profiles" TO "anon";
GRANT ALL ON TABLE "public"."profiles" TO "authenticated";
GRANT ALL ON TABLE "public"."profiles" TO "service_role";



GRANT ALL ON TABLE "public"."school_skills" TO "anon";
GRANT ALL ON TABLE "public"."school_skills" TO "authenticated";
GRANT ALL ON TABLE "public"."school_skills" TO "service_role";



GRANT ALL ON SEQUENCE "public"."school_skills_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."school_skills_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."school_skills_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."schools" TO "anon";
GRANT ALL ON TABLE "public"."schools" TO "authenticated";
GRANT ALL ON TABLE "public"."schools" TO "service_role";









ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "service_role";































