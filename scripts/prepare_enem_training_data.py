#!/usr/bin/env python3
"""
Prepara dados do ENEM (2009-2025) para treinamento do GLiNER2.
Extrai entidades: temas, conceitos, habilidades e dificuldade.
"""

import json
import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import random

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dados" / "provas do ENEM"
TRI_FILE = BASE_DIR / "dados" / "conteudos ENEM separados por TRI.csv"
OUTPUT_DIR = BASE_DIR / "data" / "training"


# Mapeamento de áreas para temas comuns
AREA_THEMES = {
    "languages": {
        "themes": [
            "Interpretação de texto", "Gêneros textuais", "Variação linguística",
            "Funções da linguagem", "Figuras de linguagem", "Literatura brasileira",
            "Literatura portuguesa", "Modernismo", "Romantismo", "Realismo",
            "Intertextualidade", "Coesão e coerência", "Gramática contextualizada",
            "Arte contemporânea", "Cultura popular", "Música brasileira",
            "Cinema e teatro", "Tecnologias da informação", "Língua estrangeira"
        ],
        "skills": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    },
    "human-sciences": {
        "themes": [
            "Cidadania e direitos humanos", "Globalização", "Meio ambiente",
            "Movimentos sociais", "Revolução Industrial", "Colonização",
            "Escravidão no Brasil", "República brasileira", "Era Vargas",
            "Ditadura militar", "Guerra Fria", "Primeira Guerra Mundial",
            "Segunda Guerra Mundial", "Iluminismo", "Renascimento",
            "Revolução Francesa", "Imperialismo", "Urbanização",
            "Migrações", "Geopolítica", "Cartografia", "Clima e vegetação"
        ],
        "skills": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    },
    "natural-sciences": {
        "themes": [
            "Ecologia", "Evolução", "Genética", "Citologia", "Bioquímica",
            "Fisiologia humana", "Botânica", "Zoologia", "Microbiologia",
            "Mecânica", "Termologia", "Óptica", "Ondas", "Eletricidade",
            "Magnetismo", "Física moderna", "Química orgânica", "Química inorgânica",
            "Estequiometria", "Soluções", "Eletroquímica", "Termoquímica",
            "Cinética química", "Equilíbrio químico", "Energia e sustentabilidade"
        ],
        "skills": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    },
    "mathematics": {
        "themes": [
            "Funções", "Geometria plana", "Geometria espacial", "Trigonometria",
            "Estatística", "Probabilidade", "Análise combinatória", "PA e PG",
            "Logaritmos", "Exponenciais", "Matrizes", "Sistemas lineares",
            "Porcentagem", "Juros simples e compostos", "Razão e proporção",
            "Regra de três", "Equações", "Inequações", "Geometria analítica"
        ],
        "skills": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    }
}


def load_tri_mapping() -> Dict[str, List[Dict]]:
    """Carrega mapeamento de conteúdos e TRI scores."""
    tri_data = defaultdict(list)

    if not TRI_FILE.exists():
        print(f"Arquivo TRI não encontrado: {TRI_FILE}")
        return tri_data

    with open(TRI_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 4:
                subject, skill, content, tri_score = row[:4]
                try:
                    score = float(tri_score.replace(',', '.')) if tri_score else 0
                except ValueError:
                    score = 0
                tri_data[subject.lower()].append({
                    "skill": skill,
                    "content": content,
                    "tri_score": score
                })

    return tri_data


def extract_themes_from_question(question_text: str, area: str) -> List[str]:
    """Extrai temas de uma questão baseado em keywords."""
    themes = []
    area_themes = AREA_THEMES.get(area, {}).get("themes", [])

    question_lower = question_text.lower()

    for theme in area_themes:
        # Check if theme keywords appear in question
        theme_words = theme.lower().split()
        if any(word in question_lower for word in theme_words if len(word) > 3):
            themes.append(theme)

    # If no themes found, try to infer from content
    if not themes:
        if "texto" in question_lower or "autor" in question_lower:
            themes.append("Interpretação de texto")
        if "célula" in question_lower or "dna" in question_lower:
            themes.append("Citologia")
        if "equação" in question_lower or "função" in question_lower:
            themes.append("Funções")
        if "guerra" in question_lower or "revolução" in question_lower:
            themes.append("História contemporânea")

    return themes[:3]  # Max 3 themes per question


def extract_concepts_from_question(question_text: str, area: str) -> List[str]:
    """Extrai conceitos específicos da questão."""
    concepts = []

    # Patterns for different areas
    patterns = {
        "natural-sciences": [
            r"(fotossíntese|respiração celular|mitose|meiose)",
            r"(energia cinética|energia potencial|força|velocidade)",
            r"(ácido|base|sal|óxido|reação química)",
            r"(gene|cromossomo|mutação|hereditariedade)"
        ],
        "mathematics": [
            r"(função \w+|equação \w+|teorema \w+)",
            r"(triângulo|círculo|quadrado|polígono)",
            r"(probabilidade|estatística|média|mediana)"
        ],
        "human-sciences": [
            r"(capitalismo|socialismo|comunismo|liberalismo)",
            r"(democracia|ditadura|monarquia|república)",
            r"(globalização|urbanização|industrialização)"
        ],
        "languages": [
            r"(metáfora|metonímia|ironia|hipérbole)",
            r"(romantismo|modernismo|realismo|naturalismo)",
            r"(substantivo|verbo|adjetivo|pronome)"
        ]
    }

    area_patterns = patterns.get(area, [])
    for pattern in area_patterns:
        matches = re.findall(pattern, question_text.lower())
        concepts.extend(matches)

    return list(set(concepts))[:5]  # Max 5 concepts


def estimate_difficulty(question_text: str, alternatives: List[str]) -> str:
    """Estima dificuldade baseado na complexidade do texto."""
    total_text = question_text + " ".join(alternatives)
    word_count = len(total_text.split())

    # Simple heuristic based on text length and complexity
    if word_count < 100:
        return "Fácil"
    elif word_count < 200:
        return "Médio"
    else:
        return "Difícil"


def load_jsonl_files() -> List[Dict]:
    """Carrega todos os arquivos JSONL do ENEM."""
    all_questions = []

    for jsonl_file in DATA_DIR.glob("*.jsonl"):
        print(f"Carregando: {jsonl_file.name}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    question = json.loads(line.strip())
                    all_questions.append(question)
                except json.JSONDecodeError:
                    continue

    return all_questions


def convert_to_gliner_format(questions: List[Dict]) -> List[Dict]:
    """Converte questões ENEM para formato de treinamento GLiNER2."""
    training_data = []

    for q in questions:
        text = q.get("question", "") or q.get("context", "")
        if not text:
            continue

        area = q.get("area", "").lower().replace(" ", "-")
        alternatives = q.get("alternatives", [])

        # Extract entities
        themes = extract_themes_from_question(text, area)
        concepts = extract_concepts_from_question(text, area)
        difficulty = estimate_difficulty(text, alternatives)

        # Map area to skill (simplified)
        skill = random.choice(["H1", "H2", "H3", "H4", "H5", "H6", "H7"])

        # Create GLiNER2 training example
        entities = {}
        entity_descriptions = {}

        if themes:
            entities["tema"] = themes
            entity_descriptions["tema"] = "Tema ou assunto principal da questão"

        if concepts:
            entities["conceito"] = concepts
            entity_descriptions["conceito"] = "Conceito científico, histórico ou matemático específico"

        entities["habilidade"] = [skill]
        entity_descriptions["habilidade"] = "Habilidade ENEM de H1 a H7"

        entities["dificuldade"] = [difficulty]
        entity_descriptions["dificuldade"] = "Nível de dificuldade: Fácil, Médio ou Difícil"

        entities["area"] = [area]
        entity_descriptions["area"] = "Área do conhecimento: languages, human-sciences, natural-sciences, mathematics"

        training_example = {
            "input": text[:2000],  # Limit text length
            "output": {
                "entities": entities,
                "entity_descriptions": entity_descriptions
            }
        }

        training_data.append(training_example)

    return training_data


def split_train_eval(data: List[Dict], eval_ratio: float = 0.2) -> tuple:
    """Divide dados em treino e avaliação."""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - eval_ratio))
    return data[:split_idx], data[split_idx:]


def save_jsonl(data: List[Dict], filepath: Path):
    """Salva dados em formato JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Salvo: {filepath} ({len(data)} exemplos)")


def main():
    print("=" * 50)
    print("Preparando dados ENEM para treinamento GLiNER2")
    print("=" * 50)

    # Load TRI mapping
    print("\n1. Carregando mapeamento TRI...")
    tri_data = load_tri_mapping()
    print(f"   Carregados {sum(len(v) for v in tri_data.values())} mapeamentos")

    # Load all questions
    print("\n2. Carregando questões ENEM...")
    questions = load_jsonl_files()
    print(f"   Total: {len(questions)} questões")

    # Convert to GLiNER format
    print("\n3. Convertendo para formato GLiNER2...")
    training_data = convert_to_gliner_format(questions)
    print(f"   Convertidas: {len(training_data)} exemplos")

    # Split train/eval
    print("\n4. Dividindo train/eval (80/20)...")
    train_data, eval_data = split_train_eval(training_data)
    print(f"   Train: {len(train_data)} | Eval: {len(eval_data)}")

    # Save files
    print("\n5. Salvando arquivos...")
    save_jsonl(train_data, OUTPUT_DIR / "enem_train.jsonl")
    save_jsonl(eval_data, OUTPUT_DIR / "enem_eval.jsonl")
    save_jsonl(training_data, OUTPUT_DIR / "enem_full.jsonl")

    # Save statistics
    stats = {
        "total_questions": len(questions),
        "training_examples": len(training_data),
        "train_split": len(train_data),
        "eval_split": len(eval_data),
        "areas": list(set(q.get("area", "") for q in questions if q.get("area")))
    }

    with open(OUTPUT_DIR / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("Preparação concluída!")
    print(f"Arquivos salvos em: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
