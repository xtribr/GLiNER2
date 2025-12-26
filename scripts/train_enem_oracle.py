#!/usr/bin/env python3
"""
Treina GLiNER2 com LoRA nos dados do ENEM para o Oráculo.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner2 import GLiNER2
from gliner2.training import TrainingConfig, GLiNER2Trainer

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "training"
OUTPUT_DIR = BASE_DIR / "models" / "enem-oracle"


def main():
    print("=" * 60)
    print("Treinamento GLiNER2 + LoRA para Oráculo ENEM")
    print("=" * 60)

    # Check data files
    train_file = DATA_DIR / "enem_train.jsonl"
    eval_file = DATA_DIR / "enem_eval.jsonl"

    if not train_file.exists():
        print(f"ERRO: Arquivo de treino não encontrado: {train_file}")
        print("Execute primeiro: python scripts/prepare_enem_training_data.py")
        return

    print(f"\nDados de treino: {train_file}")
    print(f"Dados de avaliação: {eval_file}")

    # Load model
    print("\n1. Carregando modelo GLiNER2 base...")
    try:
        model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        print("   Modelo carregado com sucesso!")
    except Exception as e:
        print(f"   ERRO ao carregar modelo: {e}")
        print("   Tentando modelo local...")
        model = GLiNER2.from_pretrained("./models/gliner2-base-v1")

    # Configure training
    print("\n2. Configurando treinamento com LoRA...")
    config = TrainingConfig(
        # Training params
        num_epochs=10,
        batch_size=2,  # Small batch for M1 Mac
        eval_batch_size=4,

        # Learning rates
        encoder_lr=1e-5,
        task_lr=5e-4,

        # Optimization
        scheduler_type="linear",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        weight_decay=0.01,

        # Mixed precision (disable for CPU/MPS)
        fp16=False,
        bf16=False,

        # Evaluation
        eval_strategy="epoch",
        save_best=True,
        save_total_limit=2,

        # LoRA config
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        lora_target_modules=["encoder", "classifier"],
        save_adapter_only=True,

        # Output
        output_dir=str(OUTPUT_DIR),
    )

    print(f"   LoRA rank: {config.lora_r}")
    print(f"   LoRA alpha: {config.lora_alpha}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Output: {OUTPUT_DIR}")

    # Create trainer
    print("\n3. Inicializando trainer...")
    trainer = GLiNER2Trainer(model, config)

    # Train
    print("\n4. Iniciando treinamento...")
    print("   (Isso pode levar alguns minutos/horas)")
    print("-" * 60)

    try:
        results = trainer.train(
            train_data=str(train_file),
            eval_data=str(eval_file) if eval_file.exists() else None
        )

        print("-" * 60)
        print("\n5. Treinamento concluído!")
        print(f"   Resultados: {results}")

    except Exception as e:
        print(f"\nERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save final model
    print("\n6. Salvando modelo...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Save adapter if using LoRA
        if config.use_lora:
            adapter_path = OUTPUT_DIR / "adapter"
            model.save_adapter(str(adapter_path))
            print(f"   Adapter LoRA salvo em: {adapter_path}")

        # Save full model
        model.save_pretrained(str(OUTPUT_DIR / "full"))
        print(f"   Modelo completo salvo em: {OUTPUT_DIR / 'full'}")

    except Exception as e:
        print(f"   ERRO ao salvar: {e}")

    print("\n" + "=" * 60)
    print("Treinamento finalizado!")
    print(f"Modelo salvo em: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
