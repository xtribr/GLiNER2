"""Tests for ml.redacao_gaps — competencias C1-C5 vs national means."""

import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from ml.redacao_gaps import (
    compute_national_means,
    infer_redacao_gaps,
)

COMP_COLS = ["media_c1", "media_c2", "media_c3", "media_c4", "media_c5"]


def _df(rows):
    return pd.DataFrame(rows, columns=["codigo_inep", "ano", "n_redacoes", *COMP_COLS])


NATIONAL = {"c1": 125.0, "c2": 150.0, "c3": 120.0, "c4": 135.0, "c5": 120.0}


class ComputeNationalMeansTest(unittest.TestCase):
    def test_weights_by_n_redacoes(self):
        df = _df([
            ["A", 2024, 10, 100, 200, 100, 100, 100],
            ["B", 2024, 90, 200, 100, 200, 200, 200],
        ])
        means = compute_national_means(df)
        # A has weight 10, B weight 90 → C1 = (100*10 + 200*90)/100 = 190
        self.assertAlmostEqual(means["c1"], 190.0)
        self.assertAlmostEqual(means["c2"], 110.0)

    def test_handles_missing_values(self):
        df = _df([
            ["A", 2024, 10, 100, None, 100, 100, 100],
            ["B", 2024, 10, 200, 200, 200, 200, 200],
        ])
        means = compute_national_means(df)
        # C2 only has B's value
        self.assertAlmostEqual(means["c2"], 200.0)


class InferRedacaoGapsTest(unittest.TestCase):
    def test_returns_competencias_with_gap_and_status(self):
        df = _df([
            ["35167939", 2024, 12, 153, 177, 145, 162, 177],  # INTEGRADO COLEGIO real values
        ])
        result = infer_redacao_gaps("35167939", df, NATIONAL)
        self.assertEqual(result["codigo_inep"], "35167939")
        self.assertEqual(len(result["competencias"]), 5)

        c1 = next(c for c in result["competencias"] if c["competencia"] == "C1")
        self.assertEqual(c1["media_escola"], 153.0)
        self.assertEqual(c1["media_nacional"], 125.0)
        self.assertEqual(c1["gap"], 28.0)
        self.assertEqual(c1["status"], "verde")

    def test_gargalo_principal_is_lowest_gap(self):
        df = _df([
            ["X", 2024, 10, 100, 200, 50, 200, 200],  # C3 is worst by far
        ])
        result = infer_redacao_gaps("X", df, NATIONAL)
        self.assertEqual(result["gargalo_principal"], "C3")

    def test_status_classification(self):
        # gap < -20 → vermelho; -20 ≤ gap ≤ 0 → amarelo; gap > 0 → verde
        df = _df([
            ["X", 2024, 10, 50, 140, 130, 110, 200],
        ])
        result = infer_redacao_gaps("X", df, NATIONAL)
        statuses = {c["competencia"]: c["status"] for c in result["competencias"]}
        self.assertEqual(statuses["C1"], "vermelho")  # gap = -75
        self.assertEqual(statuses["C2"], "amarelo")   # gap = -10
        self.assertEqual(statuses["C3"], "verde")     # gap = +10
        self.assertEqual(statuses["C4"], "vermelho")  # gap = -25
        self.assertEqual(statuses["C5"], "verde")     # gap = +80

    def test_school_not_found_returns_none(self):
        df = _df([
            ["A", 2024, 10, 130, 150, 120, 135, 120],
        ])
        result = infer_redacao_gaps("NOT_THERE", df, NATIONAL)
        self.assertIsNone(result)

    def test_handles_missing_dataframe(self):
        result = infer_redacao_gaps("X", None, NATIONAL)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
