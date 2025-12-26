"""
GLiNER Integration for TRI Content Processing

Extracts educational concepts, topics, and skills from TRI question descriptions
to enable semantic matching for recommendations.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Import GLiNER2 API client directly (avoid importing torch-dependent modules)
gliner_path = Path(__file__).parent.parent.parent.parent / "gliner2"
sys.path.insert(0, str(gliner_path.parent))

# Direct import of api_client module content to avoid __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("api_client", gliner_path / "api_client.py")
api_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_client)
GLiNER2API = api_client.GLiNER2API

logger = logging.getLogger(__name__)

# Educational entity types for ENEM content - ENHANCED for compound phrases and semantic fields
ENTITY_TYPES = {
    # Conceitos compostos - teorias, leis, princípios, fenômenos
    "conceito_cientifico": "Teoria científica, lei física, princípio químico ou fenômeno biológico composto por duas ou mais palavras (ex: 'teoria da evolução', 'lei de gravitação universal', 'efeito estufa', 'cadeia alimentar', 'transformação de energia', 'equilíbrio químico', 'seleção natural')",

    # Campo semântico - área temática ampla
    "campo_semantico": "Campo semântico ou área temática educacional composta (ex: 'movimentos sociais no Brasil', 'revolução industrial', 'literatura modernista', 'direitos humanos', 'desenvolvimento sustentável', 'globalização econômica', 'filosofia iluminista')",

    # Campo lexical - conjunto de palavras relacionadas
    "campo_lexical": "Campo lexical ou domínio de conhecimento específico (ex: 'termodinâmica', 'genética molecular', 'história colonial', 'geometria espacial', 'literatura contemporânea', 'ecossistemas aquáticos', 'economia política')",

    # Processo ou fenômeno - ações e transformações
    "processo_fenomeno": "Processo, fenômeno ou transformação descrita em frase (ex: 'urbanização acelerada', 'degradação ambiental', 'exclusão social', 'migração campo-cidade', 'aquecimento global', 'industrialização tardia', 'democratização do acesso')",

    # Contexto histórico-social
    "contexto_historico": "Período, movimento ou contexto histórico-social específico (ex: 'Era Vargas', 'Ditadura Militar', 'Período Colonial', 'Belle Époque', 'Guerra Fria', 'Revolução Francesa', 'Renascimento Cultural')",

    # Habilidade composta
    "habilidade_composta": "Habilidade ou competência cognitiva composta (ex: 'análise crítica de textos', 'interpretação de gráficos', 'resolução de problemas', 'comparação entre períodos', 'síntese de informações', 'argumentação fundamentada')"
}

# Minimum word count for extracted entities (filter single words)
MIN_ENTITY_WORDS = 2

# Generic words to filter out (single words that are too vague)
GENERIC_WORDS = {
    'ciência', 'arte', 'cultura', 'sociedade', 'natureza', 'história', 'economia',
    'política', 'tecnologia', 'educação', 'saúde', 'ambiente', 'energia', 'vida',
    'mundo', 'tempo', 'espaço', 'forma', 'ideia', 'conceito', 'processo', 'sistema',
    'relação', 'análise', 'estudo', 'pesquisa', 'teoria', 'prática', 'método',
    'juntos', 'expectativa', 'cooperatividade', 'alteridade', 'umidade', 'violência'
}

# Classification labels for difficulty and type
CLASSIFICATION_TASKS = {
    "tipo_questao": {
        "labels": [
            "interpretação de texto",
            "aplicação de fórmula",
            "análise de dados",
            "contextualização histórica",
            "compreensão de conceito",
            "resolução de problema"
        ],
        "multi_label": True,
        "cls_threshold": 0.4
    }
}


def filter_entity(entity_text: str) -> bool:
    """
    Filter entities to keep only meaningful compound phrases.

    Args:
        entity_text: The extracted entity text

    Returns:
        True if entity should be kept, False if it should be filtered out
    """
    if not entity_text or not entity_text.strip():
        return False

    text = entity_text.strip().lower()

    # Filter out generic single words
    if text in GENERIC_WORDS:
        return False

    # Count words (considering compound words)
    words = text.split()

    # Allow single words only if they are technical/specific terms (longer than 10 chars)
    if len(words) < MIN_ENTITY_WORDS:
        # Keep single words only if they are technical terms
        if len(text) < 10:
            return False
        # Or if they contain technical suffixes/prefixes
        technical_patterns = ['ismo', 'ção', 'dade', 'logia', 'metria', 'grafia', 'nomia']
        if not any(text.endswith(p) for p in technical_patterns):
            return False

    # Filter out very short phrases
    if len(text) < 5:
        return False

    return True


def clean_entities(entities_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Clean and filter extracted entities.

    Args:
        entities_dict: Dictionary of entity type -> list of entities

    Returns:
        Cleaned dictionary with filtered entities
    """
    cleaned = {}
    for entity_type, entities in entities_dict.items():
        if isinstance(entities, list):
            # Filter entities
            filtered = [e for e in entities if filter_entity(e)]
            # Remove duplicates while preserving order
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


class GLiNERProcessor:
    """Process TRI content using GLiNER for entity extraction and classification."""

    def __init__(self, api_key: Optional[str] = None, clear_cache: bool = False):
        """Initialize GLiNER processor."""
        self.api_key = api_key or os.environ.get("PIONEER_API_KEY")
        if not self.api_key:
            raise ValueError("PIONEER_API_KEY required for GLiNER processing")

        self.client = GLiNER2API(api_key=self.api_key)
        self._cache = {}
        # Use local fine-tuned model cache (97% more semantic fields, 127% more lexical fields)
        self._cache_file = Path(__file__).parent.parent / "data" / "gliner_cache_local.json"

        if clear_cache and self._cache_file.exists():
            self._cache_file.unlink()
            logger.info("Cache cleared")

        self._load_cache()

    def _load_cache(self):
        """Load cached results."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached GLiNER results")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def extract_from_description(self, description: str) -> Dict[str, Any]:
        """
        Extract entities and classify a single description.

        Args:
            description: TRI question description

        Returns:
            Dictionary with extracted entities and classifications
        """
        # Check cache
        cache_key = description[:100]  # Use first 100 chars as key
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Build extraction schema with enhanced descriptions
            schema = self.client.create_schema()
            schema.entities(ENTITY_TYPES, threshold=0.25)  # Lower threshold for compound phrases

            # Extract
            result = self.client.extract(description, schema, threshold=0.25)

            # Clean and filter entities
            if 'entities' in result:
                result['entities'] = clean_entities(result['entities'])

            # Cache result
            self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"GLiNER extraction error: {e}")
            return {"entities": {}, "error": str(e)}

    def batch_extract(self, descriptions: List[str], batch_size: int = 20) -> List[Dict[str, Any]]:
        """
        Batch extract entities from multiple descriptions.

        Args:
            descriptions: List of TRI question descriptions
            batch_size: Number of descriptions per batch

        Returns:
            List of extraction results
        """
        results = []
        uncached = []
        uncached_indices = []

        # Check cache first
        for i, desc in enumerate(descriptions):
            cache_key = desc[:100]
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                uncached.append(desc)
                uncached_indices.append(i)
                results.append(None)  # Placeholder

        if not uncached:
            logger.info("All descriptions found in cache")
            return results

        logger.info(f"Processing {len(uncached)} uncached descriptions in batches of {batch_size}")

        # Build schema once
        schema = self.client.create_schema()
        schema.entities(ENTITY_TYPES, threshold=0.3)

        # Process in batches
        for batch_start in range(0, len(uncached), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached))
            batch = uncached[batch_start:batch_end]

            try:
                batch_results = self.client.batch_extract(batch, schema, threshold=0.3)

                # Store results and update cache
                for j, res in enumerate(batch_results):
                    # Clean entities
                    if 'entities' in res:
                        res['entities'] = clean_entities(res['entities'])

                    idx = uncached_indices[batch_start + j]
                    results[idx] = res
                    cache_key = descriptions[idx][:100]
                    self._cache[cache_key] = res

                logger.info(f"Processed batch {batch_start//batch_size + 1}/{(len(uncached) + batch_size - 1)//batch_size}")

            except Exception as e:
                logger.error(f"Batch extraction error: {e}")
                # Fill with empty results
                for j in range(len(batch)):
                    idx = uncached_indices[batch_start + j]
                    results[idx] = {"entities": {}, "error": str(e)}

        # Save cache
        self._save_cache()

        return results

    def process_tri_content(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process all TRI content and add GLiNER-extracted fields.

        Args:
            csv_path: Path to TRI content CSV

        Returns:
            DataFrame with added GLiNER columns
        """
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "data" / "conteudos_tri_final.csv"

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} TRI content items")

        # Extract entities
        descriptions = df['descricao'].tolist()
        extraction_results = self.batch_extract(descriptions)

        # Add extracted fields to dataframe - new entity types for compound phrases
        df['conceitos_cientificos'] = [
            ', '.join(r.get('entities', {}).get('conceito_cientifico', [])) if r else ''
            for r in extraction_results
        ]
        df['campos_semanticos'] = [
            ', '.join(r.get('entities', {}).get('campo_semantico', [])) if r else ''
            for r in extraction_results
        ]
        df['campos_lexicais'] = [
            ', '.join(r.get('entities', {}).get('campo_lexical', [])) if r else ''
            for r in extraction_results
        ]
        df['processos_fenomenos'] = [
            ', '.join(r.get('entities', {}).get('processo_fenomeno', [])) if r else ''
            for r in extraction_results
        ]
        df['contextos_historicos'] = [
            ', '.join(r.get('entities', {}).get('contexto_historico', [])) if r else ''
            for r in extraction_results
        ]
        df['habilidades_compostas'] = [
            ', '.join(r.get('entities', {}).get('habilidade_composta', [])) if r else ''
            for r in extraction_results
        ]

        # Also store all entities combined for search
        df['all_entities'] = [
            ', '.join([
                item for entities in r.get('entities', {}).values()
                for item in (entities if isinstance(entities, list) else [])
            ]) if r else ''
            for r in extraction_results
        ]

        # Save enriched data
        output_path = Path(__file__).parent.parent / "data" / "conteudos_tri_gliner.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved enriched data to {output_path}")

        return df

    def find_similar_content(
        self,
        query_entities: Dict[str, List[str]],
        df: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Find content similar to extracted entities.

        Args:
            query_entities: Dictionary of entity types to values
            df: DataFrame with GLiNER-enriched TRI content
            top_k: Number of similar items to return

        Returns:
            DataFrame with top similar content
        """
        # Calculate similarity scores
        scores = []

        query_conceitos = set(query_entities.get('conceito_cientifico', []))
        query_campos_sem = set(query_entities.get('campo_semantico', []))
        query_campos_lex = set(query_entities.get('campo_lexical', []))
        query_processos = set(query_entities.get('processo_fenomeno', []))
        query_contextos = set(query_entities.get('contexto_historico', []))

        for _, row in df.iterrows():
            score = 0

            # Match scientific concepts (highest weight)
            row_conceitos = set(row.get('conceitos_cientificos', '').split(', ')) if row.get('conceitos_cientificos') else set()
            score += len(query_conceitos & row_conceitos) * 4

            # Match semantic fields (high weight)
            row_campos_sem = set(row.get('campos_semanticos', '').split(', ')) if row.get('campos_semanticos') else set()
            score += len(query_campos_sem & row_campos_sem) * 3

            # Match lexical fields (high weight)
            row_campos_lex = set(row.get('campos_lexicais', '').split(', ')) if row.get('campos_lexicais') else set()
            score += len(query_campos_lex & row_campos_lex) * 3

            # Match processes/phenomena
            row_processos = set(row.get('processos_fenomenos', '').split(', ')) if row.get('processos_fenomenos') else set()
            score += len(query_processos & row_processos) * 2

            # Match historical contexts
            row_contextos = set(row.get('contextos_historicos', '').split(', ')) if row.get('contextos_historicos') else set()
            score += len(query_contextos & row_contextos) * 2

            scores.append(score)

        df = df.copy()
        df['similarity_score'] = scores

        return df.nlargest(top_k, 'similarity_score')


def process_all_content():
    """Process all TRI content with GLiNER."""
    processor = GLiNERProcessor()
    df = processor.process_tri_content()
    print(f"\nProcessed {len(df)} items")
    print(f"\nSample enriched row:")
    print(df.iloc[0].to_dict())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_all_content()
