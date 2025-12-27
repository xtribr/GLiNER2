"""
Evaluate the fine-tuned GLiNER2 model on validation data.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Entity types with descriptions
ENTITY_TYPES = {
    "campo_semantico": "Área temática ou campo de conhecimento amplo, como 'Ecologia e meio ambiente', 'Tecnologia e sociedade', 'Direitos humanos'",
    "campo_lexical": "Domínio lexical específico com vocabulário técnico, como 'Ciclo hidrológico', 'Equilíbrio químico', 'Progressão aritmética'",
    "conceito_cientifico": "Conceito científico, teoria, lei ou princípio, como 'Efeito estufa', 'Seleção natural', 'Teorema de Pitágoras'",
    "processo_fenomeno": "Processo, transformação ou fenômeno, como 'Urbanização acelerada', 'Erosão do solo', 'Mutação genética'",
    "contexto_historico": "Período, movimento ou contexto histórico-social, como 'Brasil Colonial', 'Ditadura Militar', 'Iluminismo'",
    "habilidade_composta": "Habilidade cognitiva ou competência composta, como 'Análise crítica de textos', 'Interpretação de gráficos'"
}

def load_model():
    """Load GLiNER2 with fine-tuned LoRA adapter."""
    from gliner2 import GLiNER2

    print("Loading GLiNER2 base model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    adapter_path = Path(__file__).parent.parent / "models" / "gliner2-enem-semantic" / "best"
    print(f"Loading LoRA adapter from: {adapter_path}")
    model.load_adapter(str(adapter_path))

    return model

def load_validation_data(limit=50):
    """Load validation examples."""
    val_file = Path(__file__).parent.parent / "data" / "training" / "val.jsonl"
    examples = []
    with open(val_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            examples.append(data)
    return examples

def normalize_text(text):
    """Normalize text for comparison."""
    return text.lower().strip()

def compute_metrics(predicted: list, expected: list):
    """Compute precision, recall, F1 for a single entity type."""
    pred_set = set(normalize_text(p) for p in predicted)
    exp_set = set(normalize_text(e) for e in expected)

    if not pred_set and not exp_set:
        return 1.0, 1.0, 1.0  # Both empty = perfect

    if not pred_set:
        return 0.0, 0.0, 0.0  # No predictions

    if not exp_set:
        return 0.0, 1.0, 0.0  # No expected but had predictions

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(exp_set) if exp_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def evaluate_with_threshold(model, examples, threshold):
    """Evaluate model with a specific threshold."""
    schema = model.create_schema()
    schema.entities(ENTITY_TYPES, threshold=threshold)

    metrics_per_type = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})

    for example in examples:
        text = example['text']
        expected_entities = example.get('entities', {})
        result = model.extract(text, schema)
        predicted_entities = result.get('entities', {})

        all_types = set(expected_entities.keys()) | set(predicted_entities.keys())
        for etype in all_types:
            pred = predicted_entities.get(etype, [])
            exp = expected_entities.get(etype, [])
            p, r, f = compute_metrics(pred, exp)
            metrics_per_type[etype]['precision'].append(p)
            metrics_per_type[etype]['recall'].append(r)
            metrics_per_type[etype]['f1'].append(f)

    # Calculate overall metrics
    total_p, total_r, total_f = [], [], []
    for etype in metrics_per_type:
        total_p.extend(metrics_per_type[etype]['precision'])
        total_r.extend(metrics_per_type[etype]['recall'])
        total_f.extend(metrics_per_type[etype]['f1'])

    overall_p = sum(total_p) / len(total_p) if total_p else 0
    overall_r = sum(total_r) / len(total_r) if total_r else 0
    overall_f = sum(total_f) / len(total_f) if total_f else 0

    return overall_p, overall_r, overall_f, metrics_per_type


def evaluate():
    model = load_model()
    examples = load_validation_data(limit=100)

    print(f"\nEvaluating on {len(examples)} examples...\n")

    # Test different thresholds
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)
    print(f"{'Threshold':>10} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 50)

    best_threshold = 0.3
    best_f1 = 0
    results = {}

    for threshold in thresholds:
        p, r, f, metrics = evaluate_with_threshold(model, examples, threshold)
        results[threshold] = (p, r, f, metrics)
        print(f"{threshold:>10.2f} {p:>12.2%} {r:>12.2%} {f:>12.2%}")
        if f > best_f1:
            best_f1 = f
            best_threshold = threshold

    print("-" * 50)
    print(f"\n✅ Best threshold: {best_threshold} (F1: {best_f1:.2%})")

    # Show detailed results for best threshold
    p, r, f, metrics_per_type = results[best_threshold]

    print(f"\n{'=' * 80}")
    print(f"DETAILED METRICS @ threshold={best_threshold}")
    print("=" * 80)
    print(f"{'Entity Type':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)

    for etype in sorted(metrics_per_type.keys()):
        m = metrics_per_type[etype]
        avg_p = sum(m['precision']) / len(m['precision']) if m['precision'] else 0
        avg_r = sum(m['recall']) / len(m['recall']) if m['recall'] else 0
        avg_f = sum(m['f1']) / len(m['f1']) if m['f1'] else 0
        print(f"{etype:<25} {avg_p:>10.2%} {avg_r:>10.2%} {avg_f:>10.2%}")

    print("-" * 55)
    print(f"{'OVERALL':<25} {p:>10.2%} {r:>10.2%} {f:>10.2%}")

    # Show sample predictions with best threshold
    print(f"\n{'=' * 80}")
    print(f"SAMPLE PREDICTIONS @ threshold={best_threshold}")
    print("=" * 80)

    schema = model.create_schema()
    schema.entities(ENTITY_TYPES, threshold=best_threshold)

    for i, example in enumerate(examples[:10]):
        text = example['text']
        expected_entities = example.get('entities', {})
        result = model.extract(text, schema)
        predicted_entities = result.get('entities', {})

        # Filter empty lists for cleaner output
        pred_clean = {k: v for k, v in predicted_entities.items() if v}

        print(f"\n[{i+1}] {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"  Expected:  {expected_entities}")
        print(f"  Predicted: {pred_clean}")

if __name__ == "__main__":
    evaluate()
