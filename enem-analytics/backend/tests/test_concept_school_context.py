"""Tests for ml.concept_school_context."""

import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from ml.concept_school_context import get_concept_school_context

ENTITY_COLS = [
    "conceitos_cientificos", "campos_semanticos", "campos_lexicais",
    "processos_fenomenos", "contextos_historicos", "habilidades_compostas",
]


def _df(rows):
    cols = ["area_code", "tri_score"] + ENTITY_COLS
    return pd.DataFrame(rows, columns=cols)


class FindConceptItemsTest(unittest.TestCase):
    def test_finds_concept_in_any_entity_column(self):
        df = _df([
            ["CN", 500.0, "Termologia", None, None, None, None, None],
            ["CN", 550.0, None, "Termologia", None, None, None, None],
            ["CN", 600.0, "Fisiologia", None, None, None, None, None],
        ])
        r = get_concept_school_context("x", "Termologia", df, {"CN": 480.0})
        self.assertEqual(r["n_items_total"], 2)
        self.assertEqual(r["dominant_area"], "CN")

    def test_returns_none_when_concept_absent(self):
        df = _df([
            ["CN", 500.0, "Termologia", None, None, None, None, None],
        ])
        r = get_concept_school_context("x", "Eletricidade", df, {"CN": 480.0})
        self.assertIsNone(r)

    def test_case_insensitive_match(self):
        df = _df([
            ["CN", 500.0, "TERMOLOGIA", None, None, None, None, None],
        ])
        r = get_concept_school_context("x", "termologia", df, {"CN": 500.0})
        self.assertIsNotNone(r)
        self.assertEqual(r["n_items_total"], 1)

    def test_does_not_match_substring_of_other_concept(self):
        # "Fisiologia" should not match a search for "Fisio".
        df = _df([
            ["CN", 500.0, "Fisiologia, Termologia", None, None, None, None, None],
        ])
        r = get_concept_school_context("x", "Fisio", df, {"CN": 500.0})
        self.assertIsNone(r)


class StatusClassificationTest(unittest.TestCase):
    BASE = ["CN", 500.0, "Termologia", None, None, None, None, None]

    def test_status_gargalo_when_school_well_below_band(self):
        df = _df([self.BASE])
        r = get_concept_school_context("x", "Termologia", df, {"CN": 400.0})
        self.assertEqual(r["status"], "gargalo")
        self.assertEqual(r["gap"], -100.0)

    def test_status_no_nivel_when_school_inside_band(self):
        df = _df([self.BASE])
        r = get_concept_school_context("x", "Termologia", df, {"CN": 490.0})
        self.assertEqual(r["status"], "no_nivel")

    def test_status_dominado_when_school_well_above_band(self):
        df = _df([self.BASE])
        r = get_concept_school_context("x", "Termologia", df, {"CN": 600.0})
        self.assertEqual(r["status"], "dominado")

    def test_status_indefinido_when_school_score_missing(self):
        df = _df([self.BASE])
        r = get_concept_school_context("x", "Termologia", df, {"CN": None})
        self.assertEqual(r["status"], "indefinido")
        self.assertIsNone(r["gap"])


class DominantAreaTest(unittest.TestCase):
    def test_picks_area_with_most_items(self):
        df = _df([
            ["CN", 500.0, "Genética", None, None, None, None, None],
            ["CN", 550.0, "Genética", None, None, None, None, None],
            ["CH", 600.0, "Genética", None, None, None, None, None],
        ])
        r = get_concept_school_context("x", "Genética", df, {"CN": 500.0, "CH": 500.0})
        self.assertEqual(r["dominant_area"], "CN")
        self.assertEqual(len(r["areas"]), 2)


if __name__ == "__main__":
    unittest.main()
