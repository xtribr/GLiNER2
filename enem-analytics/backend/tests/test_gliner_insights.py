import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

auth_package = types.ModuleType("api.auth")
authorization_module = types.ModuleType("api.auth.authorization")
dependencies_module = types.ModuleType("api.auth.supabase_dependencies")


class UserProfile:
    pass


def get_authorized_school_user():
    return None


def get_current_admin():
    return None


authorization_module.get_authorized_school_user = get_authorized_school_user
dependencies_module.UserProfile = UserProfile
dependencies_module.get_current_admin = get_current_admin
sys.modules.setdefault("api.auth", auth_package)
sys.modules.setdefault("api.auth.authorization", authorization_module)
sys.modules.setdefault("api.auth.supabase_dependencies", dependencies_module)

from api.routes import gliner_insights


REQUIRED_ROW = {
    "area_code": "CN",
    "tri_score": "not-a-number",
    "habilidade": "H1",
    "conceitos_cientificos": "",
    "campos_semanticos": "",
    "campos_lexicais": "",
    "processos_fenomenos": "",
    "contextos_historicos": "",
    "habilidades_compostas": "",
}


def write_gliner_csv(data_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(data_dir / "conteudos_tri_gliner.csv", index=False)


class GlinerInsightsTest(unittest.TestCase):
    def test_parse_list_field_ignores_empty_items(self):
        self.assertEqual(
            gliner_insights.parse_list_field(" fotossintese, , cadeia alimentar "),
            ["fotossintese", "cadeia alimentar"],
        )
        self.assertEqual(gliner_insights.parse_list_field(""), [])
        self.assertEqual(gliner_insights.parse_list_field(None), [])

    def test_parse_confidence_field_coerces_invalid_values_to_zero(self):
        self.assertEqual(
            gliner_insights.parse_confidence_field("0.91, bad, 0.42"),
            [0.91, 0.0, 0.42],
        )

    def test_get_gliner_data_blocks_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            pd.DataFrame([{"area_code": "CN"}]).to_csv(
                data_dir / "conteudos_tri_gliner.csv",
                index=False,
            )

            with self.assertRaises(HTTPException) as exc:
                gliner_insights.get_gliner_data(data_dir=data_dir, use_cache=False)

        self.assertEqual(exc.exception.status_code, 500)
        self.assertIn("missing required columns", exc.exception.detail)

    def test_get_gliner_data_coerces_invalid_tri_score(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            write_gliner_csv(data_dir, [REQUIRED_ROW])

            df = gliner_insights.get_gliner_data(data_dir=data_dir, use_cache=False)

        self.assertTrue(pd.isna(df.loc[0, "tri_score"]))

    def test_trending_concepts_returns_empty_response_without_concepts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            write_gliner_csv(data_dir, [REQUIRED_ROW])

            with patch.object(gliner_insights, "DADOS_DIR", data_dir), patch.object(
                gliner_insights, "_gliner_df", None
            ):
                result = asyncio.run(
                    gliner_insights.get_trending_concepts(area=None, limit=30, _=None)
                )

        self.assertEqual(result["trending_concepts"], [])
        self.assertEqual(result["total_unique_concepts"], 0)

    def test_study_focus_covers_all_four_tri_areas_for_strong_school(self):
        """Regression: a school with no weak areas (z-scores all positive) used
        to surface only 3 TRI areas because of a hardcoded slice [:3]. The
        product expects all four TRI areas (CN, CH, LC, MT) every time."""
        rows = []
        for area in ("CN", "CH", "LC", "MT"):
            for tri in (500.0, 550.0, 600.0, 650.0):
                row = dict(REQUIRED_ROW)
                row.update({
                    "area_code": area,
                    "tri_score": tri,
                    "conceitos_cientificos": f"conceito_{area}_{int(tri)}",
                })
                rows.append(row)

        fake_predict_model = types.SimpleNamespace(
            predict_all_scores=lambda inep: {
                "scores": {"cn": 600, "ch": 600, "lc": 600, "mt": 600, "redacao": 600}
            }
        )
        fake_diagnosis_engine = types.SimpleNamespace(
            diagnose=lambda inep: {"priority_areas": []}  # strong school: no weak areas
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            write_gliner_csv(data_dir, rows)

            with patch.object(gliner_insights, "DADOS_DIR", data_dir), patch.object(
                gliner_insights, "_gliner_df", None
            ), patch.object(
                gliner_insights, "_redacao_df", None
            ), patch.object(
                gliner_insights, "_redacao_national_means", None
            ), patch(
                "ml.prediction_model.ENEMPredictionModel", lambda: fake_predict_model
            ), patch(
                "ml.diagnosis_engine.DiagnosisEngine", lambda: fake_diagnosis_engine
            ):
                result = asyncio.run(
                    gliner_insights.get_study_focus(codigo_inep="x", _=None)
                )

        areas = [a["area"] for a in result["focus_areas"]]
        self.assertEqual(sorted(areas), ["CH", "CN", "LC", "MT"])
        self.assertEqual(len(result["focus_areas"]), 4)

    def test_study_focus_appends_redacao_when_competencias_csv_present(self):
        """When competencias_redacao_por_escola_{year}.csv exists, the panel
        includes a fifth focus area for redacao with C1-C5 gaps."""
        tri_rows = []
        for area in ("CN", "CH", "LC", "MT"):
            row = dict(REQUIRED_ROW)
            row.update({"area_code": area, "tri_score": 550.0,
                        "conceitos_cientificos": f"conceito_{area}"})
            tri_rows.append(row)

        red_rows = [
            # School x: C3 is uniquely worst (gap = -40 vs national 110)
            {"codigo_inep": "x", "ano": 2024, "n_redacoes": 50,
             "media_c1": 100, "media_c2": 120, "media_c3": 70,
             "media_c4": 120, "media_c5": 140},
            # Second school broadens the national distribution.
            {"codigo_inep": "y", "ano": 2024, "n_redacoes": 50,
             "media_c1": 140, "media_c2": 180, "media_c3": 150,
             "media_c4": 160, "media_c5": 110},
        ]

        fake_predict_model = types.SimpleNamespace(
            predict_all_scores=lambda inep: {
                "scores": {"cn": 600, "ch": 600, "lc": 600, "mt": 600, "redacao": 600}
            }
        )
        fake_diagnosis_engine = types.SimpleNamespace(
            diagnose=lambda inep: {"priority_areas": []}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            write_gliner_csv(data_dir, tri_rows)
            pd.DataFrame(red_rows).to_csv(
                data_dir / "competencias_redacao_por_escola_2024.csv", index=False
            )

            with patch.object(gliner_insights, "DADOS_DIR", data_dir), patch.object(
                gliner_insights, "_gliner_df", None
            ), patch.object(
                gliner_insights, "_redacao_df", None
            ), patch.object(
                gliner_insights, "_redacao_national_means", None
            ), patch(
                "ml.prediction_model.ENEMPredictionModel", lambda: fake_predict_model
            ), patch(
                "ml.diagnosis_engine.DiagnosisEngine", lambda: fake_diagnosis_engine
            ):
                result = asyncio.run(
                    gliner_insights.get_study_focus(codigo_inep="x", _=None)
                )

        areas = [a["area"] for a in result["focus_areas"]]
        self.assertEqual(len(result["focus_areas"]), 5)
        self.assertIn("REDACAO", areas)
        red = next(a for a in result["focus_areas"] if a["area"] == "REDACAO")
        self.assertEqual(len(red["competencias"]), 5)
        # School x has worst gap on C3 (80 vs national 100 → -20)
        self.assertEqual(red["gargalo_principal"], "C3")


if __name__ == "__main__":
    unittest.main()
