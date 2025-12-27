"""
Data Augmentation for GLiNER ENEM Training.

Expands the training dataset using various augmentation techniques:
1. Synonym replacement
2. Entity swapping (same type)
3. Template-based generation
4. Paraphrasing patterns
5. Context variation

Usage:
    python scripts/augment_training_data.py --multiplier 3
    python scripts/augment_training_data.py --multiplier 5 --dry-run
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import argparse


# Portuguese synonyms for common terms
SYNONYMS = {
    "análise": ["avaliação", "exame", "estudo", "investigação"],
    "compreender": ["entender", "assimilar", "interpretar", "apreender"],
    "identificar": ["reconhecer", "detectar", "determinar", "distinguir"],
    "relacionar": ["associar", "conectar", "vincular", "correlacionar"],
    "interpretar": ["analisar", "decifrar", "traduzir", "explicar"],
    "avaliar": ["julgar", "apreciar", "examinar", "ponderar"],
    "processo": ["procedimento", "método", "mecanismo", "dinâmica"],
    "fenômeno": ["evento", "ocorrência", "manifestação", "acontecimento"],
    "conceito": ["noção", "ideia", "princípio", "fundamento"],
    "contexto": ["cenário", "ambiente", "situação", "conjuntura"],
    "problema": ["questão", "desafio", "dificuldade", "situação-problema"],
    "fundamental": ["essencial", "primordial", "básico", "crucial"],
    "importante": ["relevante", "significativo", "substancial", "expressivo"],
    "estudo": ["análise", "pesquisa", "investigação", "exame"],
    "questão": ["problema", "pergunta", "item", "exercício"],
    "capacidade": ["habilidade", "aptidão", "competência", "perícia"],
}

# Templates for generating new examples
TEMPLATES = {
    "conceito_cientifico": [
        "O conceito de {entity} é fundamental na compreensão de fenômenos naturais.",
        "A aplicação de {entity} permite resolver problemas complexos.",
        "{entity} é um princípio científico aplicado em diversas situações.",
        "O estudo de {entity} envolve a análise de dados e experimentos.",
        "Questão que exige conhecimento sobre {entity}.",
        "Problema contextualizado envolvendo {entity}.",
        "A compreensão de {entity} é essencial para interpretar o fenômeno.",
    ],
    "campo_semantico": [
        "No campo de {entity}, são estudados diversos conceitos.",
        "A área de {entity} abrange múltiplos temas relacionados.",
        "{entity} é um domínio do conhecimento que engloba vários aspectos.",
        "O campo temático de {entity} é explorado nesta questão.",
        "Questão sobre {entity} no contexto do ENEM.",
    ],
    "campo_lexical": [
        "O vocabulário técnico de {entity} inclui termos específicos.",
        "No domínio de {entity}, encontramos expressões características.",
        "{entity} apresenta um léxico especializado.",
        "A terminologia de {entity} é aplicada neste contexto.",
    ],
    "processo_fenomeno": [
        "O processo de {entity} envolve transformações significativas.",
        "{entity} é um fenômeno observado em diversos contextos.",
        "A ocorrência de {entity} tem consequências importantes.",
        "O fenômeno de {entity} pode ser analisado cientificamente.",
        "A dinâmica de {entity} é estudada por especialistas.",
    ],
    "contexto_historico": [
        "No período de {entity}, ocorreram mudanças significativas.",
        "{entity} representa um momento histórico importante.",
        "O contexto de {entity} influenciou eventos posteriores.",
        "Durante {entity}, transformações sociais ocorreram.",
        "A análise de {entity} revela aspectos históricos relevantes.",
    ],
    "habilidade_composta": [
        "A habilidade de {entity} é avaliada nesta questão.",
        "{entity} é uma competência fundamental para o ENEM.",
        "O candidato deve demonstrar {entity}.",
        "A questão exige {entity} do estudante.",
    ],
}

# Patterns for entity extraction from existing examples
AREA_PREFIXES = {
    "MT": ["Matemática", "Cálculo", "Geometria", "Álgebra", "Estatística"],
    "CN": ["Biologia", "Física", "Química", "Ciências Naturais"],
    "CH": ["História", "Geografia", "Sociologia", "Filosofia"],
    "LC": ["Linguagem", "Literatura", "Gramática", "Interpretação"],
}


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def apply_synonym_replacement(text: str, prob: float = 0.3) -> str:
    """Replace words with synonyms probabilistically."""
    words = text.split()
    new_words = []

    for word in words:
        word_lower = word.lower().strip('.,;:!?')
        if word_lower in SYNONYMS and random.random() < prob:
            synonym = random.choice(SYNONYMS[word_lower])
            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            new_words.append(synonym)
        else:
            new_words.append(word)

    return ' '.join(new_words)


def build_entity_bank(examples: List[Dict]) -> Dict[str, List[str]]:
    """Build a bank of entities by type from existing examples."""
    entity_bank = defaultdict(set)

    for example in examples:
        entities = example.get('entities', {})
        for entity_type, entity_list in entities.items():
            if not entity_type.startswith('_'):
                for entity in entity_list:
                    entity_bank[entity_type].add(entity)

    return {k: list(v) for k, v in entity_bank.items()}


def augment_with_entity_swap(
    example: Dict,
    entity_bank: Dict[str, List[str]],
    prob: float = 0.5
) -> List[Dict]:
    """Create new examples by swapping entities of the same type."""
    augmented = []
    text = example['text']
    entities = example.get('entities', {})
    meta = example.get('_meta', {})

    for entity_type, entity_list in entities.items():
        if entity_type.startswith('_'):
            continue

        if entity_type not in entity_bank or len(entity_bank[entity_type]) < 2:
            continue

        for entity in entity_list:
            if random.random() > prob:
                continue

            # Find a replacement entity
            candidates = [e for e in entity_bank[entity_type] if e != entity]
            if not candidates:
                continue

            replacement = random.choice(candidates)

            # Create new text with replacement
            new_text = text.replace(entity, replacement)
            if new_text != text:
                new_entities = {}
                for et, el in entities.items():
                    if et.startswith('_'):
                        continue
                    new_entities[et] = [
                        replacement if e == entity else e for e in el
                    ]

                augmented.append({
                    'text': new_text,
                    'entities': new_entities,
                    '_meta': {**meta, 'source': 'entity_swap', 'original_entity': entity}
                })

    return augmented


def augment_with_templates(
    entity_bank: Dict[str, List[str]],
    count_per_type: int = 50
) -> List[Dict]:
    """Generate new examples using templates."""
    augmented = []

    for entity_type, templates in TEMPLATES.items():
        if entity_type not in entity_bank:
            continue

        entities = entity_bank[entity_type]

        for _ in range(min(count_per_type, len(entities) * 2)):
            entity = random.choice(entities)
            template = random.choice(templates)

            text = template.format(entity=entity)

            # Determine area from entity content
            area = None
            for area_code, prefixes in AREA_PREFIXES.items():
                if any(prefix.lower() in entity.lower() for prefix in prefixes):
                    area = area_code
                    break

            augmented.append({
                'text': text,
                'entities': {entity_type: [entity]},
                '_meta': {'source': 'template', 'area': area}
            })

    return augmented


def augment_with_paraphrase(example: Dict) -> List[Dict]:
    """Create paraphrased versions of examples."""
    augmented = []
    text = example['text']
    entities = example.get('entities', {})
    meta = example.get('_meta', {})

    # Paraphrase patterns
    patterns = [
        # Add introductory phrases
        ("", "Questão sobre "),
        ("", "Análise de "),
        ("", "Problema envolvendo "),
        ("", "O estudo de "),
        # Modify endings
        (".", " no contexto do ENEM."),
        (".", ", avaliando competências específicas."),
        (".", " é fundamental para a resolução."),
    ]

    for old, new in patterns:
        if old == "":
            # Add prefix
            new_text = new + text[0].lower() + text[1:]
        else:
            # Replace ending
            new_text = text.rstrip('.') + new[1:]

        if new_text != text:
            augmented.append({
                'text': new_text,
                'entities': {k: v for k, v in entities.items() if not k.startswith('_')},
                '_meta': {**meta, 'source': 'paraphrase'}
            })

    return augmented[:2]  # Limit paraphrases per example


def augment_with_context_variation(example: Dict) -> List[Dict]:
    """Add contextual variations to examples."""
    augmented = []
    text = example['text']
    entities = example.get('entities', {})
    meta = example.get('_meta', {})
    area = meta.get('area', 'CN')

    # Context variations by area
    contexts = {
        'MT': [
            "Em uma situação-problema de matemática, ",
            "Para resolver esta questão quantitativa, ",
            "Aplicando conceitos matemáticos, ",
        ],
        'CN': [
            "No contexto científico, ",
            "Considerando fenômenos naturais, ",
            "Do ponto de vista biológico/físico/químico, ",
        ],
        'CH': [
            "Sob perspectiva histórica, ",
            "Considerando o contexto social, ",
            "Na análise geográfica/sociológica, ",
        ],
        'LC': [
            "No contexto linguístico, ",
            "Analisando o texto apresentado, ",
            "Sob perspectiva literária, ",
        ],
    }

    area_contexts = contexts.get(area, contexts['CN'])

    for context in area_contexts[:1]:  # Limit to 1 variation
        new_text = context + text[0].lower() + text[1:]
        augmented.append({
            'text': new_text,
            'entities': {k: v for k, v in entities.items() if not k.startswith('_')},
            '_meta': {**meta, 'source': 'context_variation'}
        })

    return augmented


def combine_entities(examples: List[Dict], entity_bank: Dict[str, List[str]]) -> List[Dict]:
    """Create examples combining multiple entities."""
    augmented = []

    # Combine conceito_cientifico with campo_semantico
    if 'conceito_cientifico' in entity_bank and 'campo_semantico' in entity_bank:
        concepts = entity_bank['conceito_cientifico']
        fields = entity_bank['campo_semantico']

        templates = [
            "O conceito de {concept} no campo de {field} é fundamental.",
            "{field} envolve o estudo de {concept}.",
            "Na área de {field}, {concept} é um tema central.",
        ]

        for _ in range(100):
            concept = random.choice(concepts)
            field = random.choice(fields)
            template = random.choice(templates)

            text = template.format(concept=concept, field=field)
            augmented.append({
                'text': text,
                'entities': {
                    'conceito_cientifico': [concept],
                    'campo_semantico': [field],
                },
                '_meta': {'source': 'combined_entities'}
            })

    return augmented


def augment_dataset(
    examples: List[Dict],
    multiplier: float = 2.0,
    entity_bank: Dict[str, List[str]] = None
) -> List[Dict]:
    """Apply all augmentation techniques to expand the dataset."""
    if entity_bank is None:
        entity_bank = build_entity_bank(examples)

    print(f"Entity bank: {', '.join(f'{k}: {len(v)}' for k, v in entity_bank.items())}")

    augmented = []

    # 1. Template-based generation (independent of existing examples)
    template_examples = augment_with_templates(entity_bank, count_per_type=100)
    augmented.extend(template_examples)
    print(f"  Template-based: {len(template_examples)} examples")

    # 2. Combined entities
    combined_examples = combine_entities(examples, entity_bank)
    augmented.extend(combined_examples)
    print(f"  Combined entities: {len(combined_examples)} examples")

    # 3. Per-example augmentations
    entity_swaps = []
    paraphrases = []
    context_variations = []
    synonym_replacements = []

    sample_size = min(len(examples), int(len(examples) * multiplier / 4))
    sampled = random.sample(examples, sample_size)

    for example in sampled:
        # Entity swapping
        swaps = augment_with_entity_swap(example, entity_bank, prob=0.5)
        entity_swaps.extend(swaps[:1])  # Limit per example

        # Paraphrasing
        paras = augment_with_paraphrase(example)
        paraphrases.extend(paras[:1])

        # Context variation
        contexts = augment_with_context_variation(example)
        context_variations.extend(contexts)

        # Synonym replacement
        if random.random() < 0.3:
            new_text = apply_synonym_replacement(example['text'], prob=0.3)
            if new_text != example['text']:
                entities = {k: v for k, v in example.get('entities', {}).items()
                          if not k.startswith('_')}
                synonym_replacements.append({
                    'text': new_text,
                    'entities': entities,
                    '_meta': {**example.get('_meta', {}), 'source': 'synonym'}
                })

    augmented.extend(entity_swaps)
    augmented.extend(paraphrases)
    augmented.extend(context_variations)
    augmented.extend(synonym_replacements)

    print(f"  Entity swaps: {len(entity_swaps)} examples")
    print(f"  Paraphrases: {len(paraphrases)} examples")
    print(f"  Context variations: {len(context_variations)} examples")
    print(f"  Synonym replacements: {len(synonym_replacements)} examples")

    return augmented


def deduplicate_by_text(examples: List[Dict]) -> List[Dict]:
    """Remove duplicate examples by text."""
    seen = set()
    unique = []

    for example in examples:
        text = example['text'].strip().lower()
        if text not in seen:
            seen.add(text)
            unique.append(example)

    return unique


def main():
    parser = argparse.ArgumentParser(description="Augment GLiNER training data")
    parser.add_argument("--multiplier", type=float, default=2.0,
                       help="Target multiplier for dataset size")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show statistics without saving")
    parser.add_argument("--output-suffix", type=str, default="_augmented",
                       help="Suffix for output files")

    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "training"

    print("=" * 60)
    print("Data Augmentation for GLiNER ENEM Training")
    print("=" * 60)

    # Load existing data
    print("\n1. Loading existing training data...")
    train = load_jsonl(data_dir / "train.jsonl")
    val = load_jsonl(data_dir / "val.jsonl")
    test = load_jsonl(data_dir / "test.jsonl")

    print(f"   Train: {len(train)} examples")
    print(f"   Val: {len(val)} examples")
    print(f"   Test: {len(test)} examples")

    # Build entity bank from all data
    print("\n2. Building entity bank...")
    all_examples = train + val + test
    entity_bank = build_entity_bank(all_examples)

    # Augment training data
    print("\n3. Augmenting training data...")
    augmented_train = augment_dataset(train, multiplier=args.multiplier, entity_bank=entity_bank)

    # Augment validation data (less aggressive)
    print("\n4. Augmenting validation data...")
    augmented_val = augment_dataset(val, multiplier=args.multiplier / 2, entity_bank=entity_bank)

    # Combine and deduplicate
    print("\n5. Combining and deduplicating...")
    combined_train = train + augmented_train
    combined_val = val + augmented_val

    combined_train = deduplicate_by_text(combined_train)
    combined_val = deduplicate_by_text(combined_val)

    print(f"   Final Train: {len(combined_train)} examples (was {len(train)})")
    print(f"   Final Val: {len(combined_val)} examples (was {len(val)})")
    print(f"   Test: {len(test)} examples (unchanged)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original dataset: {len(train)} train + {len(val)} val + {len(test)} test")
    print(f"Augmented dataset: {len(combined_train)} train + {len(combined_val)} val + {len(test)} test")
    print(f"Growth: {len(combined_train) / len(train):.1f}x train, {len(combined_val) / len(val):.1f}x val")

    if not args.dry_run:
        # Save augmented data
        print("\n6. Saving augmented data...")

        # Backup original files
        backup_dir = data_dir / "backup"
        backup_dir.mkdir(exist_ok=True)

        for fname in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            src = data_dir / fname
            dst = backup_dir / fname
            if src.exists() and not dst.exists():
                import shutil
                shutil.copy(src, dst)
                print(f"   Backed up {fname}")

        # Save augmented files
        save_jsonl(combined_train, data_dir / "train.jsonl")
        save_jsonl(combined_val, data_dir / "val.jsonl")

        print(f"\nAugmented data saved to {data_dir}")
        print("Original files backed up to backup/")
    else:
        print("\n[DRY RUN] No files were modified.")


if __name__ == "__main__":
    random.seed(42)
    main()
