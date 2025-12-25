"""
Train GLiNER2 for ENEM School Entity Extraction

Extracts entities from school names:
- tipo_instituicao: COLÉGIO, ESCOLA, INSTITUTO, CENTRO, etc.
- rede: ESTADUAL, FEDERAL, MUNICIPAL, particular
- modalidade: PRE-VESTIBULAR, APLICAÇÃO, TÉCNICO, MILITAR, NAVAL
- homenageado: Names of people schools are named after
"""

import sys
sys.path.insert(0, '/Volumes/notebook/GLiNER2')

import pandas as pd
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Training examples based on real ENEM school names
TRAINING_EXAMPLES = [
    # Private schools with pre-vestibular
    InputExample(
        text="FARIAS BRITO COLEGIO DE APLICACAO",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "modalidade": ["APLICACAO"],
            "homenageado": ["FARIAS BRITO"]
        }
    ),
    InputExample(
        text="CHRISTUS COLEGIO PRE UNIVERSITARIO",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "modalidade": ["PRE UNIVERSITARIO"]
        }
    ),
    InputExample(
        text="COLEGIO CLASSE A",
        entities={
            "tipo_instituicao": ["COLEGIO"]
        }
    ),
    InputExample(
        text="ARI DE SA CAVALCANTE COLEGIO - MAJOR FACUNDO",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "homenageado": ["ARI DE SA CAVALCANTE", "MAJOR FACUNDO"]
        }
    ),
    InputExample(
        text="FARIAS BRITO COLEGIO PRE-VESTIBULAR CENTRAL",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "modalidade": ["PRE-VESTIBULAR"],
            "homenageado": ["FARIAS BRITO"]
        }
    ),
    # Public schools - Federal
    InputExample(
        text="COL DE APLICACAO DA UFV - COLUNI",
        entities={
            "tipo_instituicao": ["COL"],
            "modalidade": ["APLICACAO"],
            "rede": ["UFV"]
        }
    ),
    InputExample(
        text="UFSM - COLEGIO POLITECNICO",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "modalidade": ["POLITECNICO"],
            "rede": ["UFSM"]
        }
    ),
    InputExample(
        text="COLEGIO NAVAL",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "modalidade": ["NAVAL"]
        }
    ),
    InputExample(
        text="COLEGIO MILITAR DE FORTALEZA",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "modalidade": ["MILITAR"]
        }
    ),
    # State schools
    InputExample(
        text="EE PROFESSOR JOSE MONTEIRO BOANOVA",
        entities={
            "tipo_instituicao": ["EE"],
            "rede": ["ESTADUAL"],
            "homenageado": ["PROFESSOR JOSE MONTEIRO BOANOVA"]
        }
    ),
    InputExample(
        text="ESCOLA ESTADUAL CORONEL ANTONIO PAIVA",
        entities={
            "tipo_instituicao": ["ESCOLA"],
            "rede": ["ESTADUAL"],
            "homenageado": ["CORONEL ANTONIO PAIVA"]
        }
    ),
    # Technical schools
    InputExample(
        text="ETEC PROF BASILIDES DE GODOY",
        entities={
            "tipo_instituicao": ["ETEC"],
            "modalidade": ["TECNICO"],
            "homenageado": ["PROF BASILIDES DE GODOY"]
        }
    ),
    InputExample(
        text="IFSP - CAMPUS SAO PAULO",
        entities={
            "tipo_instituicao": ["IFSP"],
            "rede": ["FEDERAL"],
            "modalidade": ["TECNICO"]
        }
    ),
    InputExample(
        text="CEFET-MG CAMPUS I",
        entities={
            "tipo_instituicao": ["CEFET"],
            "rede": ["FEDERAL"],
            "modalidade": ["TECNICO"]
        }
    ),
    # More examples
    InputExample(
        text="INSTITUTO FEDERAL DE EDUCACAO CIENCIA E TECNOLOGIA DO CEARA",
        entities={
            "tipo_instituicao": ["INSTITUTO FEDERAL"],
            "rede": ["FEDERAL"]
        }
    ),
    InputExample(
        text="COLEGIO OBJETIVO",
        entities={
            "tipo_instituicao": ["COLEGIO"]
        }
    ),
    InputExample(
        text="COLEGIO BERNOULLI",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "homenageado": ["BERNOULLI"]
        }
    ),
    InputExample(
        text="COLEGIO SANTO AGOSTINHO",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "homenageado": ["SANTO AGOSTINHO"]
        }
    ),
    InputExample(
        text="COLEGIO MARISTA DE NATAL",
        entities={
            "tipo_instituicao": ["COLEGIO"]
        }
    ),
    InputExample(
        text="ESCOLA PREPARATORIA DE CADETES DO AR",
        entities={
            "tipo_instituicao": ["ESCOLA"],
            "modalidade": ["PREPARATORIA", "MILITAR"]
        }
    ),
    InputExample(
        text="COLEGIO PEDRO II - CAMPUS SAO CRISTOVAO",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "rede": ["FEDERAL"],
            "homenageado": ["PEDRO II"]
        }
    ),
    InputExample(
        text="CAP - COLEGIO DE APLICACAO DA UFRJ",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "modalidade": ["APLICACAO"],
            "rede": ["UFRJ"]
        }
    ),
    InputExample(
        text="ESCOLA MUNICIPAL JOAO XXIII",
        entities={
            "tipo_instituicao": ["ESCOLA"],
            "rede": ["MUNICIPAL"],
            "homenageado": ["JOAO XXIII"]
        }
    ),
    InputExample(
        text="COLEGIO DAMAS",
        entities={
            "tipo_instituicao": ["COLEGIO"]
        }
    ),
    InputExample(
        text="COLEGIO SAO BENTO",
        entities={
            "tipo_instituicao": ["COLEGIO"],
            "homenageado": ["SAO BENTO"]
        }
    ),
]


def load_more_examples_from_csv():
    """Load additional examples from the ENEM dataset"""
    try:
        df = pd.read_csv('/Volumes/notebook/GLiNER2/enem-analytics/backend/data/enem_2018_2024_completo.csv')
        df_2024 = df[df['ano'] == 2024].head(100)

        examples = []
        for _, row in df_2024.iterrows():
            nome = row['nome_escola']
            entities = {}

            # Detect institution type
            for tipo in ['COLEGIO', 'ESCOLA', 'INSTITUTO', 'CENTRO', 'ETEC', 'IFSP', 'CEFET']:
                if tipo in nome.upper():
                    entities['tipo_instituicao'] = [tipo]
                    break

            # Detect network
            if 'ESTADUAL' in nome.upper():
                entities['rede'] = ['ESTADUAL']
            elif 'FEDERAL' in nome.upper() or 'UFRJ' in nome or 'UFMG' in nome or 'USP' in nome:
                entities['rede'] = ['FEDERAL']
            elif 'MUNICIPAL' in nome.upper():
                entities['rede'] = ['MUNICIPAL']

            # Detect modality
            for mod in ['MILITAR', 'NAVAL', 'TECNICO', 'PRE-VESTIBULAR', 'APLICACAO']:
                if mod in nome.upper():
                    entities['modalidade'] = [mod]
                    break

            if entities:
                examples.append(InputExample(text=nome, entities=entities))

        return examples
    except Exception as e:
        print(f"Warning: Could not load CSV examples: {e}")
        return []


def train():
    """Train the GLiNER2 model"""
    print("=" * 60)
    print("GLiNER2 Training for ENEM School Entities")
    print("=" * 60)

    # Combine examples
    all_examples = TRAINING_EXAMPLES + load_more_examples_from_csv()
    print(f"Total training examples: {len(all_examples)}")

    # Create dataset
    dataset = TrainingDataset(all_examples)
    print(f"Entity types: {dataset.get_entity_types()}")

    # Load model
    print("\nLoading base model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    # Configure training
    config = TrainingConfig(
        output_dir="./output/enem-gliner",
        experiment_name="enem-school-entities",
        num_epochs=10,
        batch_size=4,
        encoder_lr=1e-5,
        task_lr=5e-4,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_best=True,
        use_lora=True,  # Use LoRA for efficient fine-tuning
        lora_r=8,
        lora_alpha=16,
        validate_data=True,
        seed=42
    )

    # Train
    print("\nStarting training...")
    trainer = GLiNER2Trainer(model, config)
    trainer.train(train_data=dataset)

    print("\nTraining complete!")
    print(f"Model saved to: {config.output_dir}")

    # Test the model
    print("\n" + "=" * 60)
    print("Testing trained model...")
    print("=" * 60)

    test_texts = [
        "FARIAS BRITO COLEGIO DE APLICACAO",
        "ESCOLA ESTADUAL PROFESSOR JOAO SILVA",
        "COLEGIO MILITAR DO RIO DE JANEIRO",
        "IFSP - CAMPUS GUARULHOS",
    ]

    labels = ["tipo_instituicao", "rede", "modalidade", "homenageado"]

    for text in test_texts:
        entities = model.extract_entities(text, labels)
        print(f"\nText: {text}")
        for entity, label in entities:
            print(f"  {label}: {entity}")


if __name__ == "__main__":
    train()
