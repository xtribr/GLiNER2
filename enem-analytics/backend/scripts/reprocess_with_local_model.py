"""
Reprocess all TRI content using the local fine-tuned GLiNER2 model.
This replaces the API-based extraction with local inference.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Entity types with descriptions
ENTITY_TYPES = {
    "campo_semantico": "Área temática ou campo de conhecimento amplo, como 'Ecologia e meio ambiente', 'Tecnologia e sociedade', 'Direitos humanos', 'Matemática financeira', 'Física e energia'",

    "campo_lexical": "Domínio lexical específico com vocabulário técnico, como 'Ciclo hidrológico', 'Equilíbrio químico', 'Progressão aritmética', 'Figuras de linguagem', 'Revolução Industrial'",

    "conceito_cientifico": "Conceito científico composto, teoria, lei ou princípio, como 'Efeito estufa', 'Seleção natural', 'Teorema de Pitágoras', 'Lei de Newton', 'Síntese proteica'",

    "processo_fenomeno": "Processo, transformação ou fenômeno descrito em frase, como 'Urbanização acelerada', 'Erosão do solo', 'Mutação genética', 'Fluxo de energia', 'Ciclo do carbono'",

    "contexto_historico": "Período, movimento ou contexto histórico-social específico, como 'Brasil Colonial', 'Ditadura Militar', 'Iluminismo', 'Revolução Francesa', 'Semana de 22'",

    "habilidade_composta": "Habilidade cognitiva ou competência composta, como 'Análise crítica de textos', 'Interpretação de gráficos', 'Modelagem matemática', 'Argumentação fundamentada'"
}

# Words to filter out
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

    # Be more permissive with technical terms
    if len(words) < 2:
        if len(text) < 8:
            return False
        technical_patterns = ['ismo', 'ção', 'dade', 'logia', 'metria', 'grafia', 'nomia', 'triz', 'ente']
        if not any(text.endswith(p) for p in technical_patterns):
            return False

    if len(text) < 4:
        return False

    return True


def clean_entities(entities_dict: Dict) -> Dict:
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


def load_model():
    """Load GLiNER2 with fine-tuned LoRA adapter."""
    from gliner2 import GLiNER2

    logger.info("Loading GLiNER2 base model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    adapter_path = Path(__file__).parent.parent / "models" / "gliner2-enem-semantic-v2" / "best"
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model.load_adapter(str(adapter_path))

    return model


def process_batch(model, texts: List[str], schema, batch_size: int = 8) -> List[Dict]:
    """Process a batch of texts."""
    results = model.batch_extract(texts, schema, batch_size=batch_size, threshold=0.6)
    return results


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    csv_file = data_dir / "conteudos_tri_final.csv"
    output_file = data_dir / "conteudos_tri_gliner_local.csv"
    cache_file = data_dir / "gliner_cache_v2.json"

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} items")

    # Load model
    model = load_model()

    # Create schema
    schema = model.create_schema()
    schema.entities(ENTITY_TYPES, threshold=0.6)

    # Process in batches
    batch_size = 16
    all_results = {}

    texts = df['descricao'].tolist()

    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
        batch_texts = texts[i:i + batch_size]

        try:
            results = process_batch(model, batch_texts, schema, batch_size)

            for j, res in enumerate(results):
                idx = i + j
                text = batch_texts[j]
                cache_key = text[:100]

                # Clean entities
                if 'entities' in res:
                    res['entities'] = clean_entities(res['entities'])

                all_results[cache_key] = res

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue

    # Save cache
    logger.info(f"Saving cache with {len(all_results)} items...")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Create enriched CSV
    logger.info("Creating enriched CSV...")

    df['conceitos_cientificos'] = ''
    df['campos_semanticos'] = ''
    df['campos_lexicais'] = ''
    df['processos_fenomenos'] = ''
    df['contextos_historicos'] = ''
    df['habilidades_compostas'] = ''
    df['all_entities'] = ''

    for idx, row in df.iterrows():
        cache_key = row['descricao'][:100]
        if cache_key in all_results:
            res = all_results[cache_key]
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

    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved enriched CSV to: {output_file}")

    # Print coverage stats
    logger.info("\nCoverage by area:")
    for area in ['CN', 'CH', 'LC', 'MT']:
        area_df = df[df['area_code'] == area]
        with_concepts = len(area_df[area_df['conceitos_cientificos'] != ''])
        with_semantic = len(area_df[area_df['campos_semanticos'] != ''])
        with_lexical = len(area_df[area_df['campos_lexicais'] != ''])
        logger.info(f"  {area}: concepts={with_concepts}/{len(area_df)}, semantic={with_semantic}, lexical={with_lexical}")

    # Compare entity counts
    logger.info("\nEntity type distribution:")
    for entity_type in ENTITY_TYPES.keys():
        count = sum(1 for res in all_results.values()
                   if entity_type in res.get('entities', {}))
        logger.info(f"  {entity_type}: {count} items")


if __name__ == "__main__":
    main()
