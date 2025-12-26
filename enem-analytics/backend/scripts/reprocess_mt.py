"""
Script para reprocessar itens de Matemática (MT) no GLiNER
com retry logic e entity types específicos para MT.
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GLiNER API
gliner_path = Path(__file__).parent.parent.parent.parent / "gliner2"
sys.path.insert(0, str(gliner_path.parent))

import importlib.util
spec = importlib.util.spec_from_file_location("api_client", gliner_path / "api_client.py")
api_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_client)
GLiNER2API = api_client.GLiNER2API

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Entity types específicos para Matemática - mais abrangentes
ENTITY_TYPES_MT = {
    # Conceitos matemáticos
    "conceito_cientifico": "Conceito matemático, teorema, propriedade ou estrutura (ex: 'razão e proporção', 'progressão aritmética', 'função exponencial', 'teorema de Pitágoras', 'sistema linear', 'análise combinatória', 'probabilidade condicional', 'geometria analítica')",

    # Campo semântico - área temática matemática
    "campo_semantico": "Área ou ramo da matemática (ex: 'álgebra', 'geometria', 'estatística', 'trigonometria', 'cálculo', 'aritmética', 'probabilidade', 'análise de dados', 'matemática financeira')",

    # Campo lexical - domínio específico
    "campo_lexical": "Domínio ou contexto de aplicação (ex: 'geometria espacial', 'funções trigonométricas', 'equações do segundo grau', 'matrizes e determinantes', 'números complexos', 'sequências numéricas')",

    # Processo ou operação
    "processo_fenomeno": "Operação, procedimento ou processo matemático (ex: 'cálculo de área', 'resolução de sistema', 'análise gráfica', 'interpretação de dados', 'modelagem matemática', 'otimização de função')",

    # Contexto de aplicação
    "contexto_historico": "Contexto ou aplicação prática (ex: 'matemática financeira', 'estatística populacional', 'problemas de contagem', 'análise de gráficos', 'medidas e grandezas')",

    # Habilidade
    "habilidade_composta": "Habilidade ou competência matemática (ex: 'interpretação de gráfico', 'resolução de problemas', 'análise de dados', 'raciocínio lógico', 'cálculo mental', 'visualização espacial')"
}

# Entity types gerais (para outras áreas)
ENTITY_TYPES_GENERAL = {
    "conceito_cientifico": "Teoria científica, lei física, princípio químico ou fenômeno biológico composto por duas ou mais palavras",
    "campo_semantico": "Campo semântico ou área temática educacional composta",
    "campo_lexical": "Campo lexical ou domínio de conhecimento específico",
    "processo_fenomeno": "Processo, fenômeno ou transformação descrita em frase",
    "contexto_historico": "Período, movimento ou contexto histórico-social específico",
    "habilidade_composta": "Habilidade ou competência cognitiva composta"
}

GENERIC_WORDS = {
    'ciência', 'arte', 'cultura', 'sociedade', 'natureza', 'história', 'economia',
    'política', 'tecnologia', 'educação', 'saúde', 'ambiente', 'energia', 'vida',
    'mundo', 'tempo', 'espaço', 'forma', 'ideia', 'conceito', 'processo', 'sistema',
    'relação', 'análise', 'estudo', 'pesquisa', 'teoria', 'prática', 'método',
    'juntos', 'expectativa', 'cooperatividade', 'alteridade', 'umidade', 'violência',
    'número', 'valor', 'quantidade', 'total', 'resultado', 'dado', 'informação'
}


def filter_entity(entity_text: str) -> bool:
    """Filter entities to keep only meaningful phrases."""
    if not entity_text or not entity_text.strip():
        return False

    text = entity_text.strip().lower()

    if text in GENERIC_WORDS:
        return False

    words = text.split()

    # Para MT, ser mais permissivo com termos técnicos
    if len(words) < 2:
        if len(text) < 8:
            return False
        technical_patterns = ['ismo', 'ção', 'dade', 'logia', 'metria', 'grafia', 'nomia', 'triz', 'ente']
        if not any(text.endswith(p) for p in technical_patterns):
            return False

    if len(text) < 4:
        return False

    return True


def clean_entities(entities_dict):
    """Clean and filter extracted entities."""
    cleaned = {}
    for entity_type, entities in entities_dict.items():
        if isinstance(entities, list):
            filtered = [e for e in entities if filter_entity(e)]
            seen = set()
            unique = []
            for e in filtered:
                e_lower = e.lower().strip()
                if e_lower not in seen:
                    seen.add(e_lower)
                    unique.append(e)
            if unique:
                cleaned[entity_type] = unique
    return cleaned


def batch_extract_with_retry(client, batch, schema, max_retries=3, base_delay=2):
    """Extract with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            results = client.batch_extract(batch, schema, threshold=0.25)
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed: {e}")
                raise


def reprocess_area(area_code: str, force_reprocess: bool = False):
    """Reprocess a specific area with appropriate entity types."""

    api_key = os.environ.get("PIONEER_API_KEY")
    if not api_key:
        raise ValueError("PIONEER_API_KEY required")

    client = GLiNER2API(api_key=api_key)

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    cache_file = data_dir / "gliner_cache_v2.json"
    csv_file = data_dir / "conteudos_tri_final.csv"
    output_file = data_dir / "conteudos_tri_gliner.csv"

    # Load cache
    cache = {}
    if cache_file.exists() and not force_reprocess:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        logger.info(f"Loaded {len(cache)} cached items")

    # Load data
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} total items")

    # Filter by area
    area_df = df[df['area_code'] == area_code].copy()
    logger.info(f"Found {len(area_df)} items for area {area_code}")

    # Find uncached items
    uncached = []
    uncached_indices = []

    for idx, row in area_df.iterrows():
        cache_key = row['descricao'][:100]
        if cache_key not in cache:
            uncached.append(row['descricao'])
            uncached_indices.append(idx)

    logger.info(f"Found {len(uncached)} uncached items to process")

    if not uncached:
        logger.info("All items already in cache!")
        return

    # Select entity types based on area
    entity_types = ENTITY_TYPES_MT if area_code == 'MT' else ENTITY_TYPES_GENERAL

    # Build schema
    schema = client.create_schema()
    schema.entities(entity_types, threshold=0.25)

    # Process in batches
    batch_size = 15  # Smaller batches for reliability
    total_processed = 0

    for batch_start in range(0, len(uncached), batch_size):
        batch_end = min(batch_start + batch_size, len(uncached))
        batch = uncached[batch_start:batch_end]
        batch_indices = uncached_indices[batch_start:batch_end]

        try:
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(uncached) + batch_size - 1)//batch_size}")

            results = batch_extract_with_retry(client, batch, schema)

            # Store in cache
            for j, res in enumerate(results):
                if 'entities' in res:
                    res['entities'] = clean_entities(res['entities'])

                original_idx = batch_indices[j]
                cache_key = df.loc[original_idx, 'descricao'][:100]
                cache[cache_key] = res

            total_processed += len(batch)

            # Save cache after each batch
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

            logger.info(f"Processed {total_processed}/{len(uncached)} items. Cache size: {len(cache)}")

            # Small delay between batches
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            # Continue with next batch
            continue

    logger.info(f"Finished processing. Total cache size: {len(cache)}")

    # Now regenerate the enriched CSV
    logger.info("Regenerating enriched CSV...")
    regenerate_enriched_csv(cache, csv_file, output_file)


def regenerate_enriched_csv(cache, csv_file, output_file):
    """Regenerate the enriched CSV from cache."""
    df = pd.read_csv(csv_file)

    # Initialize columns
    df['conceitos_cientificos'] = ''
    df['campos_semanticos'] = ''
    df['campos_lexicais'] = ''
    df['processos_fenomenos'] = ''
    df['contextos_historicos'] = ''
    df['habilidades_compostas'] = ''
    df['all_entities'] = ''

    # Fill from cache
    cached_count = 0
    for idx, row in df.iterrows():
        cache_key = row['descricao'][:100]
        if cache_key in cache:
            res = cache[cache_key]
            entities = res.get('entities', {})

            df.loc[idx, 'conceitos_cientificos'] = ', '.join(entities.get('conceito_cientifico', []))
            df.loc[idx, 'campos_semanticos'] = ', '.join(entities.get('campo_semantico', []))
            df.loc[idx, 'campos_lexicais'] = ', '.join(entities.get('campo_lexical', []))
            df.loc[idx, 'processos_fenomenos'] = ', '.join(entities.get('processo_fenomeno', []))
            df.loc[idx, 'contextos_historicos'] = ', '.join(entities.get('contexto_historico', []))
            df.loc[idx, 'habilidades_compostas'] = ', '.join(entities.get('habilidade_composta', []))

            all_ents = []
            for ent_list in entities.values():
                if isinstance(ent_list, list):
                    all_ents.extend(ent_list)
            df.loc[idx, 'all_entities'] = ', '.join(all_ents)

            cached_count += 1

    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved enriched CSV with {cached_count}/{len(df)} items from cache")

    # Print coverage stats
    for area in ['CN', 'CH', 'LC', 'MT']:
        area_df = df[df['area_code'] == area]
        with_concepts = len(area_df[area_df['conceitos_cientificos'] != ''])
        logger.info(f"  {area}: {with_concepts}/{len(area_df)} ({100*with_concepts/len(area_df):.1f}%) with concepts")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Reprocess GLiNER for specific area')
    parser.add_argument('--area', default='MT', help='Area code to reprocess (CN, CH, LC, MT)')
    parser.add_argument('--force', action='store_true', help='Force reprocess all items')
    parser.add_argument('--all', action='store_true', help='Reprocess all areas')

    args = parser.parse_args()

    if args.all:
        for area in ['MT', 'CN', 'CH', 'LC']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing area: {area}")
            logger.info(f"{'='*50}")
            reprocess_area(area, force_reprocess=args.force)
    else:
        reprocess_area(args.area, force_reprocess=args.force)
