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
            "features": 42,
            "training_samples": 15807
        }
    )
