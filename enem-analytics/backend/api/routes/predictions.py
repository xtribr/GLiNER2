"""
Prediction API endpoints for ENEM Analytics
"""

from fastapi import APIRouter, HTTPException, Query, Path as PathParam
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.prediction_model import ENEMPredictionModel

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

# Initialize model (lazy loading)
_prediction_model = None


def get_prediction_model() -> ENEMPredictionModel:
    """Get or create prediction model instance"""
    global _prediction_model
    if _prediction_model is None:
        _prediction_model = ENEMPredictionModel()
        loaded = _prediction_model.load_all_models()
        print(f"Loaded {loaded} prediction models")
    return _prediction_model


# Pydantic models for responses
class ConfidenceInterval(BaseModel):
    low: float
    high: float


class PredictionResult(BaseModel):
    codigo_inep: str
    target_year: int
    scores: Dict[str, float]
    confidence_intervals: Dict[str, ConfidenceInterval]
    model_info: Dict[str, Any]


class SinglePrediction(BaseModel):
    codigo_inep: str
    target: str
    prediction: float
    confidence_interval: ConfidenceInterval
    uncertainty: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ScenarioInput(BaseModel):
    skill_improvements: Dict[str, float]  # {"MT_H15": 0.1, "CN_H8": 0.15}


class ScenarioResult(BaseModel):
    baseline_prediction: float
    improved_prediction: float
    delta: float
    impacted_skills: List[str]


# IMPORTANT: More specific routes must come BEFORE less specific ones

@router.get("/batch/top-improvers")
async def get_top_potential_improvers(
    limit: int = Query(20, ge=1, le=100),
    uf: Optional[str] = None,
    tipo_escola: Optional[str] = None
):
    """
    Get schools with highest predicted improvement potential

    Args:
        limit: Number of schools to return
        uf: Filter by state
        tipo_escola: Filter by school type (Privada/Pública)

    Returns:
        Schools ranked by predicted improvement
    """
    model = get_prediction_model()

    if model.preprocessor is None:
        from ml.preprocessor import ENEMPreprocessor
        model.preprocessor = ENEMPreprocessor()

    # Get 2024 schools
    df_2024 = model.preprocessor.df[model.preprocessor.df['ano'] == 2024].copy()

    if uf:
        df_2024 = df_2024[df_2024['uf'] == uf]

    if tipo_escola:
        df_2024 = df_2024[df_2024['tipo_escola'] == tipo_escola]

    # Sample schools for batch prediction (limit computation)
    sample_schools = df_2024['codigo_inep'].head(500).tolist()

    results = []
    for codigo_inep in sample_schools:
        try:
            pred = model.predict(codigo_inep, 'nota_media')
            actual = df_2024[df_2024['codigo_inep'] == codigo_inep]['nota_media'].values[0]

            if actual and pred['prediction']:
                improvement = pred['prediction'] - actual
                results.append({
                    'codigo_inep': codigo_inep,
                    'nome_escola': df_2024[df_2024['codigo_inep'] == codigo_inep]['nome_escola'].values[0],
                    'nota_atual': float(actual),
                    'nota_prevista': pred['prediction'],
                    'melhoria_esperada': improvement
                })
        except:
            continue

    # Sort by improvement potential
    results.sort(key=lambda x: x['melhoria_esperada'], reverse=True)

    return {
        'total': len(results),
        'schools': results[:limit]
    }


@router.get("/comparison/{codigo_inep}")
async def get_prediction_comparison(codigo_inep: str):
    """
    Compare predicted scores with actual historical performance

    Args:
        codigo_inep: School INEP code

    Returns:
        Comparison of predictions vs historical data
    """
    model = get_prediction_model()

    try:
        predictions = model.predict_all_scores(codigo_inep)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Get historical data
    if model.preprocessor is None:
        from ml.preprocessor import ENEMPreprocessor
        model.preprocessor = ENEMPreprocessor()

    school_df = model.preprocessor.df[
        model.preprocessor.df['codigo_inep'] == codigo_inep
    ].sort_values('ano')

    if len(school_df) == 0:
        raise HTTPException(status_code=404, detail="School not found")

    # Get latest actual scores
    latest = school_df.iloc[-1]
    latest_year = int(latest['ano'])

    historical = {
        'year': latest_year,
        'scores': {
            'cn': float(latest.get('nota_cn', 0)) if latest.get('nota_cn') else None,
            'ch': float(latest.get('nota_ch', 0)) if latest.get('nota_ch') else None,
            'lc': float(latest.get('nota_lc', 0)) if latest.get('nota_lc') else None,
            'mt': float(latest.get('nota_mt', 0)) if latest.get('nota_mt') else None,
            'redacao': float(latest.get('nota_redacao', 0)) if latest.get('nota_redacao') else None,
            'media': float(latest.get('nota_media', 0)) if latest.get('nota_media') else None,
        }
    }

    # Calculate expected change
    expected_change = {}
    for key in ['cn', 'ch', 'lc', 'mt', 'redacao', 'media']:
        if key in predictions['scores'] and historical['scores'].get(key):
            expected_change[key] = predictions['scores'][key] - historical['scores'][key]

    return {
        'codigo_inep': codigo_inep,
        'historical': historical,
        'predicted': {
            'year': 2025,
            'scores': predictions['scores']
        },
        'expected_change': expected_change,
        'confidence_intervals': predictions.get('confidence_intervals', {})
    }


@router.get("/{codigo_inep}/feature-importance/{target}", response_model=List[FeatureImportance])
async def get_feature_importance(
    codigo_inep: str,
    target: str = PathParam(..., pattern="^(cn|ch|lc|mt|redacao|media)$")
):
    """
    Get feature importance for a prediction model

    Args:
        codigo_inep: School INEP code (for context, not currently used)
        target: Target score model (cn, ch, lc, mt, redacao, media)

    Returns:
        List of features with their importance scores
    """
    model = get_prediction_model()
    target_col = f"nota_{target}"

    importance = model.get_feature_importance(target_col)

    if not importance:
        raise HTTPException(status_code=404, detail="Feature importance not available")

    return [
        FeatureImportance(feature=item['feature'], importance=item['importance'])
        for item in importance
    ]


# Add pandas import for TRI analysis
import pandas as pd


@router.get("/{codigo_inep}/tri-analysis")
async def get_tri_based_prediction_analysis(codigo_inep: str):
    """
    Get TRI-based prediction analysis for a school.

    Shows which TRI content the school can handle, skill mastery levels,
    and recommendations based on TRI difficulty progression.
    """
    model = get_prediction_model()

    if model.preprocessor is None:
        from ml.preprocessor import ENEMPreprocessor
        model.preprocessor = ENEMPreprocessor()

    preprocessor = model.preprocessor

    # Get school data
    school_df = preprocessor.df[preprocessor.df['codigo_inep'] == codigo_inep]
    if len(school_df) == 0:
        raise HTTPException(status_code=404, detail="School not found")

    # Get TRI-based features
    tri_features = preprocessor.create_tri_based_features(school_df)
    skill_gap_features = preprocessor.create_skill_gap_features(school_df)

    # Get estimated TRI scores by content
    tri_estimates = preprocessor.estimate_tri_score_by_content(codigo_inep)

    # Get predictions
    try:
        predictions = model.predict_all_scores(codigo_inep)
    except:
        predictions = {'scores': {}}

    # Build area analysis
    area_analysis = []
    area_mapping = {
        'cn': ('Ciências da Natureza', 'CN', '#22c55e'),
        'ch': ('Ciências Humanas', 'CH', '#8b5cf6'),
        'lc': ('Linguagens', 'LC', '#ec4899'),
        'mt': ('Matemática', 'MT', '#f97316')
    }

    latest = school_df.sort_values('ano').iloc[-1]

    for area_key, (area_name, area_code, color) in area_mapping.items():
        current_score = latest.get(f'nota_{area_key}', 500)
        if pd.isna(current_score):
            current_score = 500

        predicted = predictions.get('scores', {}).get(area_key, current_score)

        # Get TRI content stats for this area
        accessible_content = []
        stretch_content = []
        if preprocessor.tri_content_df is not None:
            area_content = preprocessor.tri_content_df[preprocessor.tri_content_df['area_code'] == area_code]

            # Content school can handle now (TRI <= current score)
            accessible = area_content[area_content['tri_score'] <= current_score]
            if len(accessible) > 0:
                accessible_content = [
                    {
                        'skill': row['habilidade'],
                        'tri_score': row['tri_score'],
                        'description': row['descricao'][:100] + '...' if len(row['descricao']) > 100 else row['descricao']
                    }
                    for _, row in accessible.sample(min(5, len(accessible))).iterrows()
                ]

            # Stretch content (slightly above current level)
            stretch = area_content[
                (area_content['tri_score'] > current_score) &
                (area_content['tri_score'] <= current_score + 100)
            ]
            if len(stretch) > 0:
                stretch_content = [
                    {
                        'skill': row['habilidade'],
                        'tri_score': row['tri_score'],
                        'description': row['descricao'][:100] + '...' if len(row['descricao']) > 100 else row['descricao'],
                        'gap': row['tri_score'] - current_score
                    }
                    for _, row in stretch.sample(min(5, len(stretch))).iterrows()
                ]

        area_analysis.append({
            'area': area_code,
            'area_name': area_name,
            'color': color,
            'current_score': float(current_score),
            'predicted_score': float(predicted),
            'expected_change': float(predicted - current_score),
            'tri_mastery_level': tri_features.get(f'tri_mastery_level_{area_key}', 0.5),
            'tri_gap_to_median': tri_features.get(f'tri_gap_to_median_{area_key}', 0),
            'tri_potential': tri_features.get(f'tri_potential_{area_key}', 0),
            'skill_gap_national': skill_gap_features.get(f'skill_gap_national_{area_key}', 0),
            'weak_skill_count': skill_gap_features.get(f'low_skill_count_{area_key}', 0),
            'accessible_content_sample': accessible_content,
            'stretch_content_sample': stretch_content,
            'content_based_estimate': tri_estimates.get(area_key, current_score)
        })

    return {
        'codigo_inep': codigo_inep,
        'overall_tri_mastery': tri_features.get('overall_tri_mastery', 0.5),
        'total_weak_skills': skill_gap_features.get('total_weak_skills', 0),
        'area_analysis': area_analysis,
        'insights': {
            'mastery_interpretation': _get_mastery_interpretation(tri_features.get('overall_tri_mastery', 0.5)),
            'recommendation': _get_improvement_recommendation(skill_gap_features.get('total_weak_skills', 0))
        }
    }


def _get_mastery_interpretation(mastery: float) -> str:
    """Get human-readable interpretation of mastery level"""
    if mastery >= 0.8:
        return "Excelente domínio do conteúdo TRI - escola consegue resolver questões de alta dificuldade"
    elif mastery >= 0.6:
        return "Bom domínio - escola está acima da mediana e pode avançar para conteúdos mais difíceis"
    elif mastery >= 0.4:
        return "Domínio intermediário - há espaço significativo para melhoria em conteúdos de nível médio"
    else:
        return "Foco em fundamentos - priorizar conteúdos de TRI mais acessíveis antes de avançar"


def _get_improvement_recommendation(weak_skills: float) -> str:
    """Get recommendation based on weak skill count"""
    if weak_skills <= 5:
        return "Escola tem poucos gaps críticos - focar em refinamento e conteúdos avançados"
    elif weak_skills <= 15:
        return "Gaps identificados em habilidades específicas - treino focado pode gerar ganhos rápidos"
    elif weak_skills <= 30:
        return "Vários gaps identificados - recomenda-se plano estruturado de recuperação por área"
    else:
        return "Muitos gaps críticos - priorizar fundamentos e habilidades mais frequentes no ENEM"


@router.get("/{codigo_inep}/{target}", response_model=SinglePrediction)
async def predict_single_score(
    codigo_inep: str,
    target: str = PathParam(..., pattern="^(cn|ch|lc|mt|redacao|media)$")
):
    """
    Predict a single TRI score for a school

    Args:
        codigo_inep: School INEP code
        target: Score to predict (cn, ch, lc, mt, redacao, media)

    Returns:
        Single prediction with confidence interval
    """
    model = get_prediction_model()

    target_col = f"nota_{target}"

    try:
        result = model.predict(codigo_inep, target_col)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return SinglePrediction(
        codigo_inep=codigo_inep,
        target=target,
        prediction=result['prediction'],
        confidence_interval=ConfidenceInterval(
            low=result['confidence_interval']['low'],
            high=result['confidence_interval']['high']
        ),
        uncertainty=result['uncertainty']
    )


@router.get("/{codigo_inep}", response_model=PredictionResult)
async def predict_school_scores(
    codigo_inep: str,
    target_year: int = Query(2025, ge=2024, le=2030)
):
    """
    Predict all TRI scores for a school

    Args:
        codigo_inep: School INEP code
        target_year: Year to predict for (default: 2025)

    Returns:
        Predicted scores with confidence intervals
    """
    model = get_prediction_model()

    try:
        result = model.predict_all_scores(codigo_inep)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Format confidence intervals
    confidence_intervals = {}
    for key, ci in result.get('confidence_intervals', {}).items():
        confidence_intervals[key] = ConfidenceInterval(
            low=ci['low'],
            high=ci['high']
        )

    return PredictionResult(
        codigo_inep=codigo_inep,
        target_year=target_year,
        scores=result.get('scores', {}),
        confidence_intervals=confidence_intervals,
        model_info={
            "algorithm": "HistGradientBoostingRegressor",
            "features": 64,  # Updated with TRI features
            "training_samples": 15807,
            "uses_tri_analysis": True
        }
    )
