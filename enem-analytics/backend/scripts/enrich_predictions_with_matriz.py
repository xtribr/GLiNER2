"""
Enrich ENEM 2026 Predictions with Real Habilidades from Matriz de Referência.

This script:
1. Loads predictions_2026.json
2. Loads matriz_referencia_enem.json
3. Maps each prediction to real habilidades based on area and theme
4. Adds objetos de conhecimento (curriculum topics) related to each prediction
5. Saves enriched predictions

Usage:
    python scripts/enrich_predictions_with_matriz.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher


def load_json(filepath: Path) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, filepath: Path):
    """Save JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def get_area_code(area_name: str) -> str:
    """Convert area name to code."""
    mapping = {
        'ciências da natureza': 'CN',
        'ciencias da natureza': 'CN',
        'matemática': 'MT',
        'matematica': 'MT',
        'linguagens': 'LC',
        'ciências humanas': 'CH',
        'ciencias humanas': 'CH',
    }
    return mapping.get(area_name.lower(), 'LC')


def find_related_habilidades(
    tema: str,
    area_code: str,
    matriz: Dict,
    top_n: int = 5
) -> List[Dict]:
    """Find habilidades most related to a given theme."""
    related = []

    # Get the area from matriz
    areas = matriz['matriz_referencia_enem']['areas']
    area = next((a for a in areas if a['codigo'] == area_code), None)

    if not area:
        return related

    # Search through competencias and habilidades
    for competencia in area['competencias']:
        comp_desc = competencia['descricao']

        for hab in competencia['habilidades']:
            hab_code = hab['codigo']
            hab_desc = hab['descricao']

            # Calculate relevance score
            tema_sim = similarity(tema, hab_desc)
            comp_sim = similarity(tema, comp_desc) * 0.5

            score = tema_sim + comp_sim

            if score > 0.15:  # Threshold
                related.append({
                    'codigo': f"{area_code}-{hab_code}",
                    'habilidade': hab_code,
                    'descricao': hab_desc,
                    'competencia': competencia['numero'],
                    'competencia_descricao': comp_desc[:100] + '...',
                    'relevancia': round(score, 3)
                })

    # Sort by relevance and return top N
    related.sort(key=lambda x: x['relevancia'], reverse=True)
    return related[:top_n]


def find_related_objetos_conhecimento(
    tema: str,
    area_code: str,
    matriz: Dict,
    top_n: int = 3
) -> List[Dict]:
    """Find objetos de conhecimento related to a theme."""
    related = []

    areas = matriz['matriz_referencia_enem']['areas']
    area = next((a for a in areas if a['codigo'] == area_code), None)

    if not area:
        return related

    objetos = area.get('objetos_conhecimento', [])

    # Handle different structures
    if isinstance(objetos, dict):
        # CN has sub-areas
        for sub_area, temas in objetos.items():
            for obj in temas:
                obj_tema = obj.get('tema', '')
                conteudos = obj.get('conteudos', [])

                score = similarity(tema, obj_tema)
                for conteudo in conteudos:
                    score = max(score, similarity(tema, conteudo) * 0.8)

                if score > 0.15:
                    related.append({
                        'tema': obj_tema,
                        'sub_area': sub_area,
                        'conteudos': conteudos[:5],
                        'relevancia': round(score, 3)
                    })
    else:
        # LC, CH, MT have list structure
        for obj in objetos:
            obj_tema = obj.get('tema', '')
            descricao = obj.get('descricao', '')
            conteudos = obj.get('conteudos', [])

            score = similarity(tema, obj_tema)
            score = max(score, similarity(tema, descricao) * 0.7)

            if score > 0.15:
                related.append({
                    'tema': obj_tema,
                    'descricao': descricao[:150] + '...' if len(descricao) > 150 else descricao,
                    'conteudos': conteudos[:5] if conteudos else [],
                    'relevancia': round(score, 3)
                })

    related.sort(key=lambda x: x['relevancia'], reverse=True)
    return related[:top_n]


def find_eixos_cognitivos(tema: str, matriz: Dict) -> List[Dict]:
    """Find relevant eixos cognitivos for a theme."""
    eixos = matriz['matriz_referencia_enem']['eixos_cognitivos']
    related = []

    for eixo in eixos:
        score = similarity(tema, eixo['descricao'])
        if score > 0.1:
            related.append({
                'codigo': eixo['codigo'],
                'nome': eixo['nome'],
                'descricao': eixo['descricao'][:100] + '...',
                'relevancia': round(score, 3)
            })

    related.sort(key=lambda x: x['relevancia'], reverse=True)
    return related[:2]


def enrich_prediction(prediction: Dict, matriz: Dict) -> Dict:
    """Enrich a single prediction with matriz data."""
    tema = prediction.get('tema', '')
    area = prediction.get('area', '')
    area_code = get_area_code(area)

    # Find related elements
    habilidades = find_related_habilidades(tema, area_code, matriz)
    objetos = find_related_objetos_conhecimento(tema, area_code, matriz)
    eixos = find_eixos_cognitivos(tema, matriz)

    # Create enriched prediction
    enriched = {
        **prediction,
        'area_codigo': area_code,
        'habilidades_matriz': habilidades,
        'objetos_conhecimento': objetos,
        'eixos_cognitivos': eixos,
        # Replace generic habilidades with specific ones
        'habilidades': [h['codigo'] for h in habilidades[:3]] if habilidades else prediction.get('habilidades', []),
        # Add descriptive concepts from objetos
        'conceitos_matriz': [
            obj['tema'] for obj in objetos
        ] + [
            conteudo
            for obj in objetos
            for conteudo in obj.get('conteudos', [])[:2]
        ],
    }

    # Generate better justificativa
    if habilidades:
        top_hab = habilidades[0]
        enriched['justificativa_detalhada'] = (
            f"Este tema mobiliza a habilidade {top_hab['codigo']}: "
            f"{top_hab['descricao'][:100]}... "
            f"relacionada à competência {top_hab['competencia']} de {area}."
        )

    return enriched


def generate_study_recommendations(predictions: List[Dict], matriz: Dict) -> List[Dict]:
    """Generate study recommendations based on predictions."""
    recommendations = []

    # Group by area
    by_area = {}
    for pred in predictions[:20]:  # Top 20 predictions
        area = pred.get('area_codigo', 'LC')
        if area not in by_area:
            by_area[area] = []
        by_area[area].append(pred)

    # Generate recommendations per area
    area_names = {
        'CN': 'Ciências da Natureza',
        'CH': 'Ciências Humanas',
        'LC': 'Linguagens e Códigos',
        'MT': 'Matemática'
    }

    for area_code, area_preds in by_area.items():
        # Get unique habilidades and objetos
        habilidades = []
        objetos = []

        for pred in area_preds:
            habilidades.extend(pred.get('habilidades_matriz', [])[:2])
            objetos.extend(pred.get('objetos_conhecimento', [])[:1])

        # Deduplicate
        seen_hab = set()
        unique_habs = []
        for h in habilidades:
            if h['codigo'] not in seen_hab:
                unique_habs.append(h)
                seen_hab.add(h['codigo'])

        seen_obj = set()
        unique_objs = []
        for o in objetos:
            if o['tema'] not in seen_obj:
                unique_objs.append(o)
                seen_obj.add(o['tema'])

        recommendations.append({
            'area': area_names.get(area_code, area_code),
            'area_codigo': area_code,
            'temas_prioritarios': [p['tema'] for p in area_preds[:5]],
            'habilidades_foco': unique_habs[:5],
            'objetos_estudar': unique_objs[:3],
            'dica_estudo': f"Foque nas {len(unique_habs[:5])} habilidades prioritárias de {area_names.get(area_code, area_code)}. "
                          f"Pratique questões que exigem: {', '.join([h['descricao'][:40] + '...' for h in unique_habs[:3]])}."
        })

    return recommendations


def main():
    # Paths
    predictions_path = Path(__file__).parent.parent / "data" / "predictions_2026.json"
    matriz_path = Path("/Volumes/notebook/GLiNER2/dados/matriz_referencia_enem.json")
    output_path = Path(__file__).parent.parent / "data" / "predictions_2026_enriched.json"

    print("=" * 60)
    print("Enriching ENEM 2026 Predictions with Matriz de Referência")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    predictions = load_json(predictions_path)
    matriz = load_json(matriz_path)

    print(f"   - Predictions: {predictions.get('total_predicoes', 0)} items")
    print(f"   - Matriz areas: {len(matriz['matriz_referencia_enem']['areas'])}")

    # Enrich predictions
    print("\n2. Enriching predictions with matriz data...")
    enriched_predictions = []

    for pred in predictions.get('predicoes_temas', []):
        enriched = enrich_prediction(pred, matriz)
        enriched_predictions.append(enriched)

        # Progress indicator
        if len(enriched_predictions) % 10 == 0:
            print(f"   Processed {len(enriched_predictions)} predictions...")

    # Generate recommendations
    print("\n3. Generating study recommendations...")
    recommendations = generate_study_recommendations(enriched_predictions, matriz)

    # Build output
    output = {
        **predictions,
        'predicoes_temas': enriched_predictions,
        'recomendacoes_estudo': recommendations,
        'metadata': {
            'fonte_matriz': 'MINISTÉRIO DA EDUCAÇÃO - INEP',
            'enriquecido_em': predictions.get('gerado_em', ''),
            'total_habilidades_mapeadas': sum(
                len(p.get('habilidades_matriz', [])) for p in enriched_predictions
            ),
            'total_objetos_mapeados': sum(
                len(p.get('objetos_conhecimento', [])) for p in enriched_predictions
            ),
        }
    }

    # Save
    print("\n4. Saving enriched predictions...")
    save_json(output, output_path)

    # Also update the original file
    save_json(output, predictions_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Predictions enriched: {len(enriched_predictions)}")
    print(f"Total habilidades mapped: {output['metadata']['total_habilidades_mapeadas']}")
    print(f"Total objetos mapped: {output['metadata']['total_objetos_mapeados']}")
    print(f"Study recommendations: {len(recommendations)}")
    print(f"\nOutput saved to: {output_path}")
    print(f"Original updated: {predictions_path}")

    # Show sample
    print("\n" + "-" * 60)
    print("SAMPLE ENRICHED PREDICTION:")
    print("-" * 60)
    sample = enriched_predictions[0] if enriched_predictions else {}
    print(f"Tema: {sample.get('tema', 'N/A')}")
    print(f"Area: {sample.get('area', 'N/A')} ({sample.get('area_codigo', '')})")
    print(f"Habilidades: {sample.get('habilidades', [])}")
    print(f"Habilidades Matriz: {len(sample.get('habilidades_matriz', []))} encontradas")
    if sample.get('habilidades_matriz'):
        top = sample['habilidades_matriz'][0]
        print(f"  Top: {top['codigo']} - {top['descricao'][:60]}...")
    print(f"Objetos de Conhecimento: {len(sample.get('objetos_conhecimento', []))} encontrados")
    if sample.get('objetos_conhecimento'):
        obj = sample['objetos_conhecimento'][0]
        print(f"  Top: {obj['tema']}")


if __name__ == "__main__":
    main()
