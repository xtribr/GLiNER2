"""
Generate training data from ENEM Reference Matrix (matriz_referencia_enem.json).

Creates enriched training examples for GLiNER fine-tuning with:
- habilidade_enem: Specific ENEM skills (H1-H30 per area)
- competencia_enem: ENEM competencies
- objeto_conhecimento: Knowledge objects/topics
- eixo_cognitivo: Cognitive axes

Usage:
    python scripts/generate_matriz_training_data.py
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any


def load_matriz(filepath: Path) -> Dict:
    """Load the ENEM reference matrix."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_habilidade_examples(area: Dict) -> List[Dict]:
    """Generate training examples from habilidades."""
    examples = []
    area_code = area['codigo']
    area_name = area['nome']

    for competencia in area['competencias']:
        comp_num = competencia['numero']
        comp_desc = competencia['descricao']

        for hab in competencia['habilidades']:
            hab_code = hab['codigo']
            hab_desc = hab['descricao']

            # Example 1: Direct habilidade mention
            text = f"A habilidade {hab_code} de {area_name} avalia a capacidade de {hab_desc.lower()}"
            examples.append({
                "text": text,
                "entities": {
                    "habilidade_enem": [f"{hab_code} - {hab_desc[:50]}..."],
                    "campo_semantico": [area_name],
                },
                "_meta": {"area": area_code, "habilidade": hab_code, "source": "matriz"}
            })

            # Example 2: Question-style
            text = f"Questão que exige {hab_desc}"
            examples.append({
                "text": text,
                "entities": {
                    "habilidade_enem": [hab_desc],
                    "habilidade_composta": [hab_desc[:60]],
                },
                "_meta": {"area": area_code, "habilidade": hab_code, "source": "matriz"}
            })

            # Example 3: Competency context
            text = f"Na competência {comp_num} ({comp_desc[:80]}...), a habilidade {hab_code} requer {hab_desc}"
            examples.append({
                "text": text,
                "entities": {
                    "competencia_enem": [comp_desc],
                    "habilidade_enem": [hab_desc],
                },
                "_meta": {"area": area_code, "habilidade": hab_code, "competencia": comp_num, "source": "matriz"}
            })

    return examples


def generate_objeto_conhecimento_examples(area: Dict) -> List[Dict]:
    """Generate training examples from objetos de conhecimento."""
    examples = []
    area_code = area['codigo']
    area_name = area['nome']

    objetos = area.get('objetos_conhecimento', [])

    # Handle different structures (list or dict with sub-areas)
    if isinstance(objetos, dict):
        # CN has sub-areas: fisica, quimica, biologia
        for sub_area, temas in objetos.items():
            for tema in temas:
                tema_name = tema.get('tema', '')
                conteudos = tema.get('conteudos', [])

                # Example with tema
                text = f"O tema {tema_name} em {sub_area.capitalize()} aborda conceitos fundamentais."
                examples.append({
                    "text": text,
                    "entities": {
                        "objeto_conhecimento": [tema_name],
                        "campo_semantico": [sub_area.capitalize()],
                    },
                    "_meta": {"area": area_code, "sub_area": sub_area, "source": "matriz"}
                })

                # Examples with conteudos
                for conteudo in conteudos[:5]:  # Limit per tema
                    text = f"Conteúdo de {area_name}: {conteudo}"
                    examples.append({
                        "text": text,
                        "entities": {
                            "objeto_conhecimento": [conteudo],
                            "conceito_cientifico": [conteudo] if area_code in ['CN', 'MT'] else [],
                            "campo_lexical": [tema_name],
                        },
                        "_meta": {"area": area_code, "tema": tema_name, "source": "matriz"}
                    })
    else:
        # LC, CH, MT have list structure
        for obj in objetos:
            tema = obj.get('tema', '')
            descricao = obj.get('descricao', '')
            conteudos = obj.get('conteudos', [])

            # Example with tema and description
            if descricao:
                text = f"{tema}: {descricao[:150]}"
                examples.append({
                    "text": text,
                    "entities": {
                        "objeto_conhecimento": [tema],
                        "campo_semantico": [tema],
                    },
                    "_meta": {"area": area_code, "source": "matriz"}
                })

            # Examples with conteudos (for MT)
            for conteudo in conteudos[:5]:
                text = f"O estudo de {conteudo} é fundamental em {area_name}."
                entities = {
                    "objeto_conhecimento": [conteudo],
                    "campo_lexical": [tema] if tema else [],
                }
                if area_code == 'MT':
                    entities["conceito_cientifico"] = [conteudo]

                examples.append({
                    "text": text,
                    "entities": entities,
                    "_meta": {"area": area_code, "tema": tema, "source": "matriz"}
                })

    return examples


def generate_eixo_cognitivo_examples(eixos: List[Dict]) -> List[Dict]:
    """Generate training examples from eixos cognitivos."""
    examples = []

    for eixo in eixos:
        codigo = eixo['codigo']
        nome = eixo['nome']
        descricao = eixo['descricao']

        # Example 1: Direct eixo mention
        text = f"O eixo cognitivo '{nome}' ({codigo}) visa: {descricao}"
        examples.append({
            "text": text,
            "entities": {
                "eixo_cognitivo": [nome],
                "habilidade_composta": [nome],
            },
            "_meta": {"eixo": codigo, "source": "matriz"}
        })

        # Example 2: Applied context
        text = f"Questões que exigem {nome.lower()}: {descricao[:100]}"
        examples.append({
            "text": text,
            "entities": {
                "eixo_cognitivo": [f"{codigo} - {nome}"],
                "habilidade_composta": [descricao[:60]],
            },
            "_meta": {"eixo": codigo, "source": "matriz"}
        })

    return examples


def generate_cross_reference_examples(matriz: Dict) -> List[Dict]:
    """Generate examples that cross-reference multiple elements."""
    examples = []
    areas = matriz['matriz_referencia_enem']['areas']
    eixos = matriz['matriz_referencia_enem']['eixos_cognitivos']

    # Cross-reference habilidades with eixos
    for area in areas:
        area_code = area['codigo']
        area_name = area['nome']

        for competencia in area['competencias']:
            for hab in competencia['habilidades']:
                # Pick a random eixo for variety
                eixo = random.choice(eixos)

                text = f"A {hab['codigo']} de {area_name} mobiliza o eixo '{eixo['nome']}': {hab['descricao'][:80]}"
                examples.append({
                    "text": text,
                    "entities": {
                        "habilidade_enem": [hab['descricao'][:60]],
                        "eixo_cognitivo": [eixo['nome']],
                        "campo_semantico": [area_name],
                    },
                    "_meta": {"area": area_code, "habilidade": hab['codigo'], "eixo": eixo['codigo'], "source": "matriz_cross"}
                })

    return examples


def generate_practical_examples(matriz: Dict) -> List[Dict]:
    """Generate practical/applied examples simulating real ENEM questions."""
    examples = []

    # Templates for practical examples
    templates = {
        'LC': [
            "Análise de texto literário que exige {hab} no contexto de {tema}.",
            "Questão sobre gênero textual envolvendo {tema} e {hab}.",
            "Interpretação de charge/tirinha relacionada a {tema}.",
        ],
        'MT': [
            "Problema de {tema} que requer {hab}.",
            "Questão contextualizada sobre {tema} envolvendo cálculo de {conteudo}.",
            "Análise de gráfico/tabela sobre {tema} exigindo {hab}.",
        ],
        'CN': [
            "Questão sobre {tema} que avalia {hab}.",
            "Problema envolvendo {conteudo} no contexto de {tema}.",
            "Análise de experimento sobre {tema} exigindo {hab}.",
        ],
        'CH': [
            "Questão sobre {tema} no contexto histórico-geográfico.",
            "Análise de fonte histórica relacionada a {tema}.",
            "Problema envolvendo {tema} e transformações sociais.",
        ],
    }

    areas = matriz['matriz_referencia_enem']['areas']

    for area in areas:
        area_code = area['codigo']
        area_templates = templates.get(area_code, templates['CH'])

        # Get temas from objetos_conhecimento
        temas = []
        objetos = area.get('objetos_conhecimento', [])
        if isinstance(objetos, dict):
            for sub_area, items in objetos.items():
                for item in items:
                    temas.append(item.get('tema', ''))
        else:
            for obj in objetos:
                temas.append(obj.get('tema', ''))

        # Generate examples
        for competencia in area['competencias'][:3]:  # Limit
            for hab in competencia['habilidades'][:3]:  # Limit
                tema = random.choice(temas) if temas else area['nome']
                template = random.choice(area_templates)

                text = template.format(
                    hab=hab['descricao'][:50],
                    tema=tema,
                    conteudo=tema
                )

                examples.append({
                    "text": text,
                    "entities": {
                        "habilidade_enem": [hab['descricao'][:60]],
                        "objeto_conhecimento": [tema],
                        "campo_semantico": [area['nome']],
                    },
                    "_meta": {"area": area_code, "habilidade": hab['codigo'], "source": "matriz_practical"}
                })

    return examples


def split_data(examples: List[Dict], train_ratio=0.8, val_ratio=0.1) -> tuple:
    """Split data into train/val/test sets."""
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        examples[:train_end],
        examples[train_end:val_end],
        examples[val_end:]
    )


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            # Clean empty entity lists
            if 'entities' in item:
                item['entities'] = {k: v for k, v in item['entities'].items() if v}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def merge_with_existing(new_data: List[Dict], existing_path: Path) -> List[Dict]:
    """Merge new data with existing training data."""
    existing = []
    if existing_path.exists():
        with open(existing_path, 'r', encoding='utf-8') as f:
            for line in f:
                existing.append(json.loads(line))

    # Combine and deduplicate by text
    seen_texts = set(item['text'] for item in existing)
    merged = existing.copy()

    for item in new_data:
        if item['text'] not in seen_texts:
            merged.append(item)
            seen_texts.add(item['text'])

    return merged


def update_entity_types(output_dir: Path):
    """Update entity_types.json with new types from matriz."""
    entity_types_path = output_dir / "entity_types.json"

    # Load existing
    if entity_types_path.exists():
        with open(entity_types_path, 'r', encoding='utf-8') as f:
            entity_types = json.load(f)
    else:
        entity_types = {}

    # Add new types
    new_types = {
        "habilidade_enem": [
            "Identificar diferentes linguagens",
            "Reconhecer recursos expressivos",
            "Resolver situação-problema",
            "Analisar informações em gráficos",
            "Avaliar propostas de intervenção",
            "Interpretar fenômenos naturais",
            "Compreender processos históricos",
            "Relacionar conceitos científicos",
            "Contextualizar produção artística",
            "Aplicar conhecimentos matemáticos",
        ],
        "competencia_enem": [
            "Dominar linguagens e seus recursos expressivos",
            "Compreender fenômenos naturais e sociais",
            "Construir argumentação consistente",
            "Elaborar propostas de intervenção",
            "Utilizar conhecimento geométrico",
            "Apropriar-se de conhecimentos científicos",
        ],
        "objeto_conhecimento": [
            "Estudo do texto e gêneros textuais",
            "Conhecimentos numéricos e algébricos",
            "Transformações químicas e energia",
            "Ecologia e ciências ambientais",
            "Diversidade cultural e movimentos sociais",
            "Hereditariedade e evolução",
            "Fenômenos elétricos e magnéticos",
            "Organização social e política",
        ],
        "eixo_cognitivo": [
            "Dominar linguagens",
            "Compreender fenômenos",
            "Enfrentar situações-problema",
            "Construir argumentação",
            "Elaborar propostas",
        ],
    }

    # Merge
    for key, values in new_types.items():
        if key not in entity_types:
            entity_types[key] = values
        else:
            existing = set(entity_types[key])
            entity_types[key] = list(existing | set(values))

    # Save
    with open(entity_types_path, 'w', encoding='utf-8') as f:
        json.dump(entity_types, f, ensure_ascii=False, indent=2)

    print(f"Updated entity_types.json with {len(new_types)} new types")


def main():
    # Paths
    matriz_path = Path("/Volumes/notebook/GLiNER2/dados/matriz_referencia_enem.json")
    output_dir = Path(__file__).parent.parent / "data" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Training Data from ENEM Reference Matrix")
    print("=" * 60)

    # Load matriz
    print("\n1. Loading matriz_referencia_enem.json...")
    matriz = load_matriz(matriz_path)

    # Generate examples
    print("\n2. Generating training examples...")
    all_examples = []

    # From habilidades
    for area in matriz['matriz_referencia_enem']['areas']:
        examples = generate_habilidade_examples(area)
        all_examples.extend(examples)
        print(f"   - {area['codigo']}: {len(examples)} habilidade examples")

    # From objetos de conhecimento
    for area in matriz['matriz_referencia_enem']['areas']:
        examples = generate_objeto_conhecimento_examples(area)
        all_examples.extend(examples)
        print(f"   - {area['codigo']}: {len(examples)} objeto_conhecimento examples")

    # From eixos cognitivos
    eixo_examples = generate_eixo_cognitivo_examples(
        matriz['matriz_referencia_enem']['eixos_cognitivos']
    )
    all_examples.extend(eixo_examples)
    print(f"   - Eixos cognitivos: {len(eixo_examples)} examples")

    # Cross-reference examples
    cross_examples = generate_cross_reference_examples(matriz)
    all_examples.extend(cross_examples)
    print(f"   - Cross-reference: {len(cross_examples)} examples")

    # Practical examples
    practical_examples = generate_practical_examples(matriz)
    all_examples.extend(practical_examples)
    print(f"   - Practical: {len(practical_examples)} examples")

    print(f"\n   Total new examples: {len(all_examples)}")

    # Merge with existing data
    print("\n3. Merging with existing training data...")
    train_merged = merge_with_existing(all_examples, output_dir / "train.jsonl")

    # Split new data only for validation/test augmentation
    _, new_val, new_test = split_data(all_examples, train_ratio=0.7, val_ratio=0.15)

    val_merged = merge_with_existing(new_val, output_dir / "val.jsonl")
    test_merged = merge_with_existing(new_test, output_dir / "test.jsonl")

    print(f"   - Train: {len(train_merged)} total examples")
    print(f"   - Val: {len(val_merged)} total examples")
    print(f"   - Test: {len(test_merged)} total examples")

    # Save
    print("\n4. Saving enriched training data...")
    save_jsonl(train_merged, output_dir / "train.jsonl")
    save_jsonl(val_merged, output_dir / "val.jsonl")
    save_jsonl(test_merged, output_dir / "test.jsonl")

    # Update entity types
    print("\n5. Updating entity_types.json...")
    update_entity_types(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"New examples generated from matriz: {len(all_examples)}")
    print(f"Final training set size: {len(train_merged)}")
    print(f"Output directory: {output_dir}")
    print("\nNew entity types added:")
    print("  - habilidade_enem")
    print("  - competencia_enem")
    print("  - objeto_conhecimento")
    print("  - eixo_cognitivo")
    print("\nReady for GLiNER fine-tuning!")


if __name__ == "__main__":
    random.seed(42)
    main()
