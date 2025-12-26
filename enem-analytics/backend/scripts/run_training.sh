#!/bin/bash
# Run GLiNER fine-tuning with LoRA
# Requires Python 3.10-3.12 (onnxruntime compatibility)

set -e

echo "============================================"
echo "GLiNER2 Fine-tuning for ENEM"
echo "============================================"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" == "3.14" ]] || [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "ERROR: Python $PYTHON_VERSION is not supported by onnxruntime"
    echo "Please use Python 3.10, 3.11, or 3.12"
    echo ""
    echo "You can create a compatible environment with:"
    echo "  python3.12 -m venv venv312"
    echo "  source venv312/bin/activate"
    echo "  pip install torch gliner gliner2 transformers"
    exit 1
fi

# Install dependencies if needed
echo ""
echo "1. Checking dependencies..."
pip install -q torch gliner transformers huggingface_hub onnxruntime

# Generate training data first
echo ""
echo "2. Generating training data from matriz..."
python scripts/generate_matriz_training_data.py

# Run fine-tuning
echo ""
echo "3. Starting fine-tuning..."
python scripts/finetune_gliner_enem.py \
    --epochs 10 \
    --batch-size 4 \
    --lora-r 16 \
    --lora-alpha 32.0

echo ""
echo "============================================"
echo "Training complete!"
echo "Model saved to: models/gliner2-enem-semantic"
echo "============================================"
