"""
Continue training GLiNER2 ENEM semantic model for 2 more epochs.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gliner2 import GLiNER2
from gliner2.training.data import TrainingDataset, InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


def load_training_data():
    """Load training and validation data."""
    data_dir = Path(__file__).parent.parent / "data" / "training"

    train_examples = []
    with open(data_dir / "train.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            train_examples.append(InputExample(
                text=data['text'],
                entities=data['entities']
            ))

    val_examples = []
    with open(data_dir / "val.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            val_examples.append(InputExample(
                text=data['text'],
                entities=data['entities']
            ))

    return train_examples, val_examples


def main():
    print("=" * 60)
    print("Continue Training GLiNER2 ENEM Semantic Model (+2 epochs)")
    print("=" * 60)

    # Load data
    train_examples, val_examples = load_training_data()
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    # Create datasets
    train_dataset = TrainingDataset(train_examples)
    val_dataset = TrainingDataset(val_examples)

    # Load model with existing adapter
    print("\nLoading base model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    # Load existing adapter
    adapter_path = Path(__file__).parent.parent / "models" / "gliner2-enem-semantic" / "best"
    print(f"Loading existing adapter from: {adapter_path}")
    model.load_adapter(str(adapter_path))

    # Configure training for 2 more epochs
    output_dir = Path(__file__).parent.parent / "models" / "gliner2-enem-semantic-v2"

    config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name="gliner2-enem-semantic-v2",
        num_epochs=2,
        batch_size=4,
        eval_batch_size=8,
        gradient_accumulation_steps=2,
        encoder_lr=5e-6,  # Lower LR for fine-tuning
        task_lr=2.5e-4,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_best=True,
        metric_for_best="val_loss",
        use_lora=True,
        lora_r=32,
        lora_alpha=64.0,
        lora_dropout=0.1,
        save_adapter_only=True,
        early_stopping=False,  # Run all 2 epochs
        seed=42
    )

    # Train
    print("\nStarting training for 2 more epochs...")
    trainer = GLiNER2Trainer(model, config)
    trainer.train(train_data=train_dataset, eval_data=val_dataset)

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
