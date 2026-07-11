import json
from pathlib import Path

import pytest

from ml.prediction_model import ENEMPredictionModel
from ml.preprocessor import ENEMPreprocessor


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def preprocessor() -> ENEMPreprocessor:
    """Load only the real versioned sources committed with the backend."""
    return ENEMPreprocessor(DATA_DIR)


def test_current_sources_are_explicit_and_year_compatible(preprocessor):
    provenance = preprocessor.get_data_provenance()

    assert provenance["enem_results"]["file"] == "enem_2018_2025_completo.csv"
    assert provenance["enem_results"]["year"] == 2025
    assert provenance["national_skills"]["file"] == "habilidades_2025.csv"
    assert provenance["national_skills"]["year"] == 2025
    assert provenance["school_skills"]["file"] == "school_skills_2025.csv"
    assert provenance["school_skills"]["schema"] == "granular_by_skill"
    assert provenance["school_skills"]["used_in_features"] is False
    assert preprocessor.validate_training_sources() == []


def test_training_blocks_a_used_source_from_another_year(preprocessor):
    original_year = preprocessor.source_years["national_skills"]
    preprocessor.source_years["national_skills"] = 2024
    preprocessor._provenance_cache.clear()

    try:
        with pytest.raises(ValueError, match="Ano incompatível"):
            preprocessor.validate_training_sources()
    finally:
        preprocessor.source_years["national_skills"] = original_year
        preprocessor._provenance_cache.clear()


def test_training_mismatch_requires_explicit_override(preprocessor):
    original_year = preprocessor.source_years["national_skills"]
    preprocessor.source_years["national_skills"] = 2024
    preprocessor._provenance_cache.clear()

    try:
        problems = preprocessor.validate_training_sources(allow_year_mismatch=True)
        assert problems == [
            "Ano incompatível: resultados ENEM 2025 e habilidades nacionais 2024"
        ]
    finally:
        preprocessor.source_years["national_skills"] = original_year
        preprocessor._provenance_cache.clear()


def test_prediction_audit_report_includes_data_provenance(tmp_path):
    manifest = {
        "enem_results": {
            "file": "enem_2018_2025_completo.csv",
            "year": 2025,
            "used_in_features": True,
        }
    }
    model = ENEMPredictionModel(model_dir=tmp_path)
    model.model_artifacts = {
        "nota_media": {"metadata": {"data_provenance": manifest}}
    }

    report_path = model._write_audit_report({})
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["data_provenance"] == manifest
