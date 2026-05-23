"""Tests for ml.content_bridge — pure functions that map (area, score) to content."""

import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from ml.content_bridge import infer_content_gaps


def _df(rows):
    cols = [
        "area_code", "tri_score",
        "conceitos_cientificos", "campos_semanticos", "campos_lexicais",
        "processos_fenomenos", "contextos_historicos", "habilidades_compostas",
    ]
    return pd.DataFrame(rows, columns=cols)


class InferContentGapsTest(unittest.TestCase):
    def test_returns_top_concepts_in_band(self):
        df = _df([
            ["MT", 470, "função quadrática", None, None, None, None, None],
            ["MT", 480, "função quadrática, geometria analítica", None, None, None, None, None],
            ["MT", 520, "função quadrática", None, None, None, None, None],
            ["MT", 800, "cálculo diferencial", None, None, None, None, None],  # out of band
            ["CN", 480, "estequiometria", None, None, None, None, None],       # wrong area
        ])
        result = infer_content_gaps(area="MT", score=480, tri_content_df=df, band_width=50, top_n=3)
        concepts = {c["concept"]: c["count"] for c in result["top_concepts"]}
        self.assertEqual(concepts.get("função quadrática"), 3)
        self.assertEqual(concepts.get("geometria analítica"), 1)
        self.assertNotIn("cálculo diferencial", concepts)
        self.assertNotIn("estequiometria", concepts)
        self.assertEqual(result["sample_size"], 3)
        self.assertEqual(result["band"], [430, 530])

    def test_aggregates_across_entity_columns(self):
        df = _df([
            ["CH", 500, None, None, None, None, "revolução industrial", None],
            ["CH", 510, None, "trabalho assalariado", None, None, None, None],
            ["CH", 490, None, None, None, "urbanização", None, None],
        ])
        result = infer_content_gaps(area="CH", score=500, tri_content_df=df, band_width=30)
        labels = {c["concept"] for c in result["top_concepts"]}
        self.assertIn("revolução industrial", labels)
        self.assertIn("trabalho assalariado", labels)
        self.assertIn("urbanização", labels)

    def test_respects_top_n_limit(self):
        rows = [["MT", 500, f"conceito_{i}", None, None, None, None, None] for i in range(10)]
        df = _df(rows)
        result = infer_content_gaps(area="MT", score=500, tri_content_df=df, top_n=3)
        self.assertEqual(len(result["top_concepts"]), 3)

    def test_empty_when_no_rows_in_band(self):
        df = _df([
            ["MT", 800, "tópico avançado", None, None, None, None, None],
        ])
        result = infer_content_gaps(area="MT", score=400, tri_content_df=df, band_width=50)
        self.assertEqual(result["sample_size"], 0)
        self.assertEqual(result["top_concepts"], [])

    def test_empty_when_area_absent(self):
        df = _df([
            ["MT", 500, "função", None, None, None, None, None],
        ])
        result = infer_content_gaps(area="CN", score=500, tri_content_df=df)
        self.assertEqual(result["sample_size"], 0)
        self.assertEqual(result["top_concepts"], [])

    def test_handles_none_dataframe(self):
        result = infer_content_gaps(area="MT", score=500, tri_content_df=None)
        self.assertEqual(result["sample_size"], 0)
        self.assertEqual(result["top_concepts"], [])

    def test_ignores_blank_and_nan_cells(self):
        df = _df([
            ["MT", 500, "", "função", None, "  ", None, None],
            ["MT", 510, None, None, None, None, None, None],
        ])
        result = infer_content_gaps(area="MT", score=500, tri_content_df=df, band_width=30)
        concepts = [c["concept"] for c in result["top_concepts"]]
        self.assertEqual(concepts, ["função"])

    def test_splits_comma_separated_entities(self):
        df = _df([
            ["MT", 500, "razão e proporção, regra de três", None, None, None, None, None],
        ])
        result = infer_content_gaps(area="MT", score=500, tri_content_df=df, band_width=30)
        labels = {c["concept"] for c in result["top_concepts"]}
        self.assertEqual(labels, {"razão e proporção", "regra de três"})


if __name__ == "__main__":
    unittest.main()
