"""
Fine-tune GLiNER2 with LoRA for ENEM educational content extraction.

Focuses on improving:
- campo_semantico (semantic fields)
- campo_lexical (lexical domains)
- conceito_cientifico (scientific concepts)

Usage:
    python scripts/finetune_gliner_enem.py --epochs 10 --batch-size 4
    python scripts/finetune_gliner_enem.py --eval-only  # Evaluate existing adapter
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy imports for GPU
Extractor = None
GLiNER2Trainer = None
TrainingConfig = None
InputExample = None


def lazy_import():
    """Import heavy modules only when needed."""
    global Extractor, GLiNER2Trainer, TrainingConfig, InputExample

    from gliner2 import Extractor
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
    from gliner2.training.data import InputExample

    return Extractor, GLiNER2Trainer, TrainingConfig, InputExample


# Entity type descriptions for ENEM - IMPROVED
ENTITY_TYPES = {
    "campo_semantico": "Área temática ou campo de conhecimento amplo, como 'Ecologia e meio ambiente', 'Tecnologia e sociedade', 'Direitos humanos', 'Matemática financeira', 'Física e energia'",

    "campo_lexical": "Domínio lexical específico com vocabulário técnico, como 'Ciclo hidrológico', 'Equilíbrio químico', 'Progressão aritmética', 'Figuras de linguagem', 'Revolução Industrial'",

    "conceito_cientifico": "Conceito científico composto, teoria, lei ou princípio, como 'Efeito estufa', 'Seleção natural', 'Teorema de Pitágoras', 'Lei de Newton', 'Síntese proteica'",

    "processo_fenomeno": "Processo, transformação ou fenômeno descrito em frase, como 'Urbanização acelerada', 'Erosão do solo', 'Mutação genética', 'Fluxo de energia', 'Ciclo do carbono'",

    "contexto_historico": "Período, movimento ou contexto histórico-social específico, como 'Brasil Colonial', 'Ditadura Militar', 'Iluminismo', 'Revolução Francesa', 'Semana de 22'",

    "habilidade_composta": "Habilidade cognitiva ou competência composta, como 'Análise crítica de textos', 'Interpretação de gráficos', 'Modelagem matemática', 'Argumentação fundamentada'"
}


def load_training_data(data_dir: Path) -> tuple:
    """Load training data from JSONL files."""
    def load_jsonl(filepath):
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                examples.append(data)
        return examples

    train = load_jsonl(data_dir / "train.jsonl")
    val = load_jsonl(data_dir / "val.jsonl")
    test = load_jsonl(data_dir / "test.jsonl")

    return train, val, test


def convert_to_input_examples(data: List[Dict]) -> List:
    """Convert raw data to InputExample objects."""
    InputExample = lazy_import()[3]

    examples = []
    for item in data:
        # Remove metadata
        entities = {k: v for k, v in item.get("entities", {}).items()
                   if not k.startswith("_")}

        if entities:
            examples.append(InputExample(
                text=item["text"],
                entities=entities
            ))

    return examples


def create_model(model_name: str = "fastino/gliner2-base-v1"):
    """Create or load GLiNER2 model."""
    Extractor = lazy_import()[0]

    logger.info(f"Loading model: {model_name}")
    model = Extractor.from_pretrained(model_name)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to("mps")

    return model


def train_model(
    model,
    train_examples,
    val_examples,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
):
    """Train model with LoRA."""
    _, GLiNER2Trainer, TrainingConfig, _ = lazy_import()

    config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name="gliner2-enem-semantic",
        num_epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,

        # Learning rates
        encoder_lr=1e-5,  # Lower for encoder (pretrained)
        task_lr=5e-4,     # Higher for task heads

        # LoRA configuration
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        lora_target_modules=[
            "encoder",      # Encoder attention layers
            "span_rep",     # Span representation (critical for phrase boundaries)
            "classifier",   # Entity type classification
        ],
        save_adapter_only=True,

        # Training settings
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        weight_decay=0.01,

        # Evaluation
        eval_strategy="epoch",
        save_best=True,
        metric_for_best="val_loss",
        greater_is_better=False,
        save_total_limit=3,

        # Mixed precision
        fp16=torch.cuda.is_available(),

        # Logging
        logging_steps=10,
        report_to_wandb=False,  # Set True if you have wandb

        # Early stopping
        early_stopping=True,
        early_stopping_patience=3,

        seed=42,
    )

    trainer = GLiNER2Trainer(model, config)

    logger.info("Starting training...")
    logger.info(f"  Train examples: {len(train_examples)}")
    logger.info(f"  Val examples: {len(val_examples)}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  LoRA rank: {lora_r}")

    trainer.train(
        train_data=train_examples,
        eval_data=val_examples,
    )

    return trainer


def evaluate_model(model, test_examples: List, entity_types: Dict) -> Dict:
    """Evaluate model on test set."""
    logger.info(f"Evaluating on {len(test_examples)} test examples...")

    results = {
        "total": 0,
        "by_type": {},
    }

    # Create schema for extraction
    schema = model.create_schema()
    schema.entities(entity_types, threshold=0.25)

    for example in test_examples:
        text = example.text
        gold_entities = example.entities

        # Extract with model
        try:
            pred = model.extract(text, schema)
            pred_entities = pred.get("entities", {})
        except Exception as e:
            logger.warning(f"Extraction error: {e}")
            pred_entities = {}

        # Compare
        for entity_type in entity_types.keys():
            if entity_type not in results["by_type"]:
                results["by_type"][entity_type] = {
                    "gold_count": 0,
                    "pred_count": 0,
                    "matches": 0,
                }

            gold = set(e.lower() for e in gold_entities.get(entity_type, []))
            pred = set(e.lower() for e in pred_entities.get(entity_type, []))

            results["by_type"][entity_type]["gold_count"] += len(gold)
            results["by_type"][entity_type]["pred_count"] += len(pred)
            results["by_type"][entity_type]["matches"] += len(gold & pred)

        results["total"] += 1

    # Calculate metrics
    for entity_type, counts in results["by_type"].items():
        precision = counts["matches"] / counts["pred_count"] if counts["pred_count"] > 0 else 0
        recall = counts["matches"] / counts["gold_count"] if counts["gold_count"] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        counts["precision"] = precision
        counts["recall"] = recall
        counts["f1"] = f1

    return results


def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    for entity_type, metrics in results["by_type"].items():
        print(f"\n{entity_type}:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1:        {metrics['f1']:.2%}")
        print(f"  (Gold: {metrics['gold_count']}, Pred: {metrics['pred_count']}, Match: {metrics['matches']})")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER2 for ENEM")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--model", default="fastino/gliner2-base-v1", help="Base model")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate only")
    parser.add_argument("--adapter-path", type=str, help="Path to adapter for evaluation")

    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "training"
    output_dir = Path(__file__).parent.parent / "models" / "gliner2-enem-semantic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    train_data, val_data, test_data = load_training_data(data_dir)
    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Convert to InputExample
    train_examples = convert_to_input_examples(train_data)
    val_examples = convert_to_input_examples(val_data)
    test_examples = convert_to_input_examples(test_data)

    # Create model
    model = create_model(args.model)

    if args.eval_only:
        # Load adapter if provided
        if args.adapter_path:
            logger.info(f"Loading adapter from: {args.adapter_path}")
            model.load_adapter(args.adapter_path)

        # Evaluate
        results = evaluate_model(model, test_examples, ENTITY_TYPES)
        print_results(results)
    else:
        # Train
        trainer = train_model(
            model=model,
            train_examples=train_examples,
            val_examples=val_examples,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )

        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        results = evaluate_model(model, test_examples, ENTITY_TYPES)
        print_results(results)

        # Save results
        with open(output_dir / "eval_results.json", 'w') as f:
            # Convert non-serializable items
            serializable = {
                "total": results["total"],
                "by_type": {
                    k: {kk: float(vv) if isinstance(vv, float) else vv for kk, vv in v.items()}
                    for k, v in results["by_type"].items()
                }
            }
            json.dump(serializable, f, indent=2)

        logger.info(f"\nModel saved to: {output_dir}")
        logger.info("Adapter file: adapter_model.safetensors")


if __name__ == "__main__":
    main()
