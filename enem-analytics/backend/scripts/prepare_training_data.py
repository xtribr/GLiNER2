"""
Prepare training data for GLiNER fine-tuning with LoRA.
Converts cached extractions to training format and adds quality filters.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import random
import re

# Entity type definitions - IMPROVED for semantic/lexical fields
ENTITY_TYPES_IMPROVED = {
    # Campos Semânticos - áreas temáticas amplas
    "campo_semantico": [
        "Ecologia e meio ambiente",
        "Tecnologia e sociedade",
        "Saúde pública",
        "Direitos humanos",
        "Globalização econômica",
        "Cultura brasileira",
        "Movimentos sociais",
        "Urbanização",
        "Relações de trabalho",
        "Comunicação e mídia",
        "Educação e cidadania",
        "Arte e literatura",
        "Ciência e ética",
        "Política e democracia",
        "Identidade cultural",
        "Sustentabilidade",
        "Diversidade cultural",
        "Conflitos territoriais",
        "Transformações sociais",
        "Patrimônio histórico",
        # Matemática
        "Álgebra e funções",
        "Geometria e medidas",
        "Estatística e probabilidade",
        "Aritmética e números",
        "Matemática financeira",
        # Ciências
        "Física e energia",
        "Química e materiais",
        "Biologia e vida",
        "Astronomia e cosmos",
        "Genética e evolução",
    ],

    # Campos Lexicais - domínios específicos de vocabulário
    "campo_lexical": [
        "Ecossistema florestal",
        "Ciclo hidrológico",
        "Cadeia alimentar",
        "Poluição atmosférica",
        "Aquecimento global",
        "Energia renovável",
        "Recursos naturais",
        "Biodiversidade",
        "Desmatamento",
        "Reciclagem de materiais",
        "Metabolismo celular",
        "Sistema imunológico",
        "Reprodução humana",
        "Genética molecular",
        "Evolução das espécies",
        "Termodinâmica",
        "Eletromagnetismo",
        "Mecânica clássica",
        "Óptica geométrica",
        "Ondas sonoras",
        "Reações químicas",
        "Ligações químicas",
        "Soluções aquosas",
        "Ácidos e bases",
        "Equilíbrio químico",
        # Humanas
        "Revolução Industrial",
        "Guerra Fria",
        "Colonização portuguesa",
        "Escravidão africana",
        "República Velha",
        "Era Vargas",
        "Ditadura Militar",
        "Redemocratização",
        "Globalização neoliberal",
        "Imperialismo europeu",
        # Linguagens
        "Gêneros textuais",
        "Figuras de linguagem",
        "Variação linguística",
        "Intertextualidade",
        "Coesão textual",
        # Matemática
        "Progressão aritmética",
        "Progressão geométrica",
        "Função exponencial",
        "Função logarítmica",
        "Função quadrática",
        "Geometria analítica",
        "Trigonometria",
        "Análise combinatória",
        "Probabilidade condicional",
        "Estatística descritiva",
    ],

    # Conceitos científicos compostos
    "conceito_cientifico": [
        "Efeito estufa",
        "Camada de ozônio",
        "Fotossíntese",
        "Respiração celular",
        "Divisão celular",
        "Síntese proteica",
        "Seleção natural",
        "Deriva genética",
        "Lei de Mendel",
        "Teorema de Pitágoras",
        "Lei de Newton",
        "Lei de Ohm",
        "Princípio de Arquimedes",
        "Efeito Doppler",
        "Fusão nuclear",
        "Fissão nuclear",
        "Oxidação e redução",
        "Catálise enzimática",
        "Equilíbrio iônico",
        "Cinética química",
    ],

    # Processos e fenômenos
    "processo_fenomeno": [
        "Urbanização acelerada",
        "Migração rural-urbana",
        "Industrialização tardia",
        "Expansão agrícola",
        "Desertificação",
        "Erosão do solo",
        "Assoreamento de rios",
        "Lixiviação",
        "Eutrofização",
        "Bioacumulação",
        "Mutação genética",
        "Especiação",
        "Sucessão ecológica",
        "Fluxo de energia",
        "Ciclo do carbono",
        "Ciclo do nitrogênio",
        "Condução térmica",
        "Convecção",
        "Irradiação",
        "Difração da luz",
    ],

    # Contextos histórico-sociais
    "contexto_historico": [
        "Brasil Colonial",
        "Período Imperial",
        "Primeira República",
        "Estado Novo",
        "Governo JK",
        "Regime Militar",
        "Nova República",
        "Antiguidade Clássica",
        "Idade Média",
        "Renascimento",
        "Iluminismo",
        "Revolução Francesa",
        "Belle Époque",
        "Entreguerras",
        "Pós-guerra",
        "Guerra do Paraguai",
        "Revolta da Vacina",
        "Semana de 22",
        "Movimento Tropicalista",
        "Diretas Já",
    ],

    # Habilidades compostas
    "habilidade_composta": [
        "Análise crítica de textos",
        "Interpretação de gráficos",
        "Resolução de problemas",
        "Argumentação fundamentada",
        "Síntese de informações",
        "Comparação histórica",
        "Contextualização social",
        "Inferência textual",
        "Modelagem matemática",
        "Análise de dados",
    ],
}


def load_data() -> tuple:
    """Load cache and CSV data."""
    data_dir = Path(__file__).parent.parent / "data"

    with open(data_dir / "gliner_cache_v2.json", 'r', encoding='utf-8') as f:
        cache = json.load(f)

    df = pd.read_csv(data_dir / "conteudos_tri_final.csv")

    return cache, df


def extract_quality_examples(cache: dict, df: pd.DataFrame) -> List[Dict]:
    """Extract high-quality examples for training."""
    examples = []

    for idx, row in df.iterrows():
        cache_key = row['descricao'][:100]

        if cache_key not in cache:
            continue

        cached = cache[cache_key]
        entities = cached.get('entities', {})

        # Skip if no entities
        if not entities:
            continue

        # Quality filters
        total_entities = sum(len(v) for v in entities.values() if isinstance(v, list))

        # Skip if too few or too many entities
        if total_entities < 2 or total_entities > 15:
            continue

        # Check for multi-word entities (compound phrases)
        has_compound = False
        for ent_list in entities.values():
            if isinstance(ent_list, list):
                for ent in ent_list:
                    if len(ent.split()) >= 2:
                        has_compound = True
                        break

        if not has_compound:
            continue

        example = {
            "text": row['descricao'],
            "entities": entities,
            "area": row.get('area_code', 'unknown'),
            "habilidade": row.get('habilidade', ''),
        }
        examples.append(example)

    return examples


def convert_to_training_format(examples: List[Dict]) -> List[Dict]:
    """Convert to GLiNER training format (InputExample style)."""
    training_data = []

    for ex in examples:
        # Create training example
        train_ex = {
            "text": ex["text"],
            "entities": ex["entities"],
            # Metadata for filtering/analysis
            "_meta": {
                "area": ex["area"],
                "habilidade": ex["habilidade"],
            }
        }
        training_data.append(train_ex)

    return training_data


def create_augmented_examples(base_examples: List[Dict], entity_types: Dict) -> List[Dict]:
    """Create augmented examples using entity type knowledge."""
    augmented = []

    # For each entity type, create examples where we know the entities
    for entity_type, examples_list in entity_types.items():
        for entity_example in examples_list:
            # Create a synthetic text that contains this entity
            text = f"O estudo sobre {entity_example} é fundamental para compreender os conceitos relacionados."

            augmented.append({
                "text": text,
                "entities": {
                    entity_type: [entity_example]
                },
                "_meta": {"source": "augmented"}
            })

    return augmented


def split_data(data: List[Dict], train_ratio=0.8, val_ratio=0.1) -> tuple:
    """Split data into train/val/test sets."""
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return data[:train_end], data[train_end:val_end], data[val_end:]


def analyze_entity_distribution(data: List[Dict]) -> Dict:
    """Analyze entity type distribution in dataset."""
    distribution = {}

    for ex in data:
        for entity_type, entities in ex.get("entities", {}).items():
            if entity_type not in distribution:
                distribution[entity_type] = {"count": 0, "examples": []}

            if isinstance(entities, list):
                distribution[entity_type]["count"] += len(entities)
                distribution[entity_type]["examples"].extend(entities[:3])

    return distribution


def main():
    """Main function to prepare training data."""
    print("Loading data...")
    cache, df = load_data()
    print(f"  Cache: {len(cache)} items")
    print(f"  CSV: {len(df)} rows")

    print("\nExtracting quality examples...")
    examples = extract_quality_examples(cache, df)
    print(f"  Quality examples: {len(examples)}")

    print("\nConverting to training format...")
    training_data = convert_to_training_format(examples)

    print("\nCreating augmented examples...")
    augmented = create_augmented_examples([], ENTITY_TYPES_IMPROVED)
    print(f"  Augmented examples: {len(augmented)}")

    # Combine
    all_data = training_data + augmented
    print(f"\nTotal examples: {len(all_data)}")

    print("\nSplitting data...")
    train, val, test = split_data(all_data)
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

    # Analyze distribution
    print("\nEntity distribution in training set:")
    dist = analyze_entity_distribution(train)
    for etype, info in sorted(dist.items(), key=lambda x: -x[1]["count"]):
        print(f"  {etype}: {info['count']} entities")
        if info["examples"]:
            print(f"    Examples: {info['examples'][:3]}")

    # Save datasets
    output_dir = Path(__file__).parent.parent / "data" / "training"
    output_dir.mkdir(exist_ok=True)

    def save_jsonl(data, filename):
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_jsonl(train, "train.jsonl")
    save_jsonl(val, "val.jsonl")
    save_jsonl(test, "test.jsonl")

    # Save entity types
    with open(output_dir / "entity_types.json", 'w', encoding='utf-8') as f:
        json.dump(ENTITY_TYPES_IMPROVED, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_dir}/")
    print("  - train.jsonl")
    print("  - val.jsonl")
    print("  - test.jsonl")
    print("  - entity_types.json")


if __name__ == "__main__":
    main()
