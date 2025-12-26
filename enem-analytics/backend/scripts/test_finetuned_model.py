"""
Test the fine-tuned GLiNER2 model with LoRA adapter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gliner2 import GLiNER2

# Sample texts for testing
TEST_TEXTS = [
    "O estudo da progressão aritmética e geométrica é fundamental para compreender a matemática financeira e juros compostos.",
    "A fotossíntese é um processo biológico que ocorre nos cloroplastos das células vegetais, convertendo energia luminosa em energia química.",
    "Durante o período da Ditadura Militar no Brasil, houve censura à imprensa e restrição de direitos civis.",
    "A análise de gráficos estatísticos requer habilidades de interpretação de dados e raciocínio lógico-matemático.",
    "O aquecimento global está relacionado ao efeito estufa causado pela emissão de gases como o dióxido de carbono.",
]

# Entity types
ENTITY_TYPES = {
    "campo_semantico": "Área temática ou campo de conhecimento amplo",
    "campo_lexical": "Domínio lexical específico com vocabulário técnico",
    "conceito_cientifico": "Conceito científico composto, teoria, lei ou princípio",
    "processo_fenomeno": "Processo, transformação ou fenômeno",
    "contexto_historico": "Período, movimento ou contexto histórico-social",
    "habilidade_composta": "Habilidade cognitiva ou competência composta",
}


def main():
    # Load base model
    print("Loading base model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    # Load LoRA adapter
    adapter_path = Path(__file__).parent.parent / "models" / "gliner2-enem-semantic" / "best"
    print(f"Loading LoRA adapter from: {adapter_path}")
    model.load_adapter(str(adapter_path))

    print("\n" + "=" * 60)
    print("Testing Fine-tuned Model")
    print("=" * 60)

    # Create schema with entity types
    schema = model.create_schema()
    schema.entities(ENTITY_TYPES, threshold=0.25)

    for i, text in enumerate(TEST_TEXTS, 1):
        print(f"\n[{i}] {text[:80]}...")

        # Extract entities
        result = model.extract(text, schema, threshold=0.25)

        if result and 'entities' in result:
            for entity_type, entities in result['entities'].items():
                if entities:
                    print(f"  {entity_type}: {entities}")
        else:
            print("  (no entities found)")

    print("\n" + "=" * 60)
    print("Model test complete!")


if __name__ == "__main__":
    main()
