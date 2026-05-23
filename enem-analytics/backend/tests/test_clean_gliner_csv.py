"""Tests for scripts.clean_gliner_csv — concept-label cleanup."""

import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.clean_gliner_csv import (
    build_canonical_map,
    clean_cell,
    clean_dataframe,
    is_valid_token,
)

ENTITY_COLS = [
    "conceitos_cientificos",
    "campos_semanticos",
    "campos_lexicais",
    "processos_fenomenos",
    "contextos_historicos",
    "habilidades_compostas",
]


def _df(rows):
    """Build a minimal DataFrame matching the real CSV columns."""
    base = {col: None for col in ENTITY_COLS}
    return pd.DataFrame([{**base, **row} for row in rows])


class IsValidTokenTest(unittest.TestCase):
    def test_short_tokens_rejected(self):
        self.assertFalse(is_valid_token("ab"))
        self.assertFalse(is_valid_token(""))

    def test_long_tokens_rejected(self):
        self.assertFalse(is_valid_token("x" * 60))

    def test_tokens_without_letters_rejected(self):
        self.assertFalse(is_valid_token("1234"))
        self.assertFalse(is_valid_token("---"))

    def test_concept_labels_accepted(self):
        self.assertTrue(is_valid_token("Biomas Brasileiros"))
        self.assertTrue(is_valid_token("razão e proporção"))
        self.assertTrue(is_valid_token("DNA"))


class CanonicalMapTest(unittest.TestCase):
    def test_picks_most_frequent_casing(self):
        df = _df([
            {"conceitos_cientificos": "Biomas Brasileiros"},
            {"conceitos_cientificos": "Biomas Brasileiros"},
            {"conceitos_cientificos": "Biomas brasileiros"},
        ])
        canon = build_canonical_map(df)
        self.assertEqual(canon["biomas brasileiros"], "Biomas Brasileiros")

    def test_alphabetical_tiebreak(self):
        df = _df([
            {"conceitos_cientificos": "Degradação"},
            {"conceitos_cientificos": "degradação"},
        ])
        canon = build_canonical_map(df)
        # Equal counts → alphabetical: 'D' < 'd' in ASCII
        self.assertEqual(canon["degradação"], "Degradação")


class CleanCellTest(unittest.TestCase):
    def setUp(self):
        self.canon = {
            "biomas brasileiros": "Biomas Brasileiros",
            "degradação": "Degradação",
        }

    def test_dedupes_case_variants(self):
        out = clean_cell("Biomas Brasileiros, Biomas brasileiros, Degradação", self.canon)
        self.assertEqual(out, "Biomas Brasileiros, Degradação")

    def test_collapses_whitespace(self):
        out = clean_cell("Biomas  Brasileiros", self.canon)
        self.assertEqual(out, "Biomas Brasileiros")

    def test_drops_long_phrases(self):
        out = clean_cell(
            "Biomas Brasileiros, avaliar informações pessoais e comportamentais sobre o candidato",
            self.canon,
        )
        self.assertEqual(out, "Biomas Brasileiros")

    def test_returns_none_for_empty_or_unparseable(self):
        self.assertIsNone(clean_cell("", self.canon))
        self.assertIsNone(clean_cell(None, self.canon))
        # All tokens too long
        self.assertIsNone(clean_cell("x" * 60, self.canon))

    def test_preserves_order_of_first_occurrence(self):
        out = clean_cell("Degradação, Biomas Brasileiros, Degradação", self.canon)
        self.assertEqual(out, "Degradação, Biomas Brasileiros")


class CleanDataframeTest(unittest.TestCase):
    def test_idempotent_on_clean_input(self):
        df = _df([{"conceitos_cientificos": "Biomas Brasileiros, Degradação"}])
        out1, _ = clean_dataframe(df)
        out2, _ = clean_dataframe(out1)
        self.assertEqual(out1["conceitos_cientificos"].iloc[0], out2["conceitos_cientificos"].iloc[0])

    def test_reports_tokens_removed(self):
        df = _df([
            {"conceitos_cientificos": "Biomas Brasileiros, Biomas brasileiros"},
        ])
        _, stats = clean_dataframe(df)
        # 2 tokens in → 1 canonical token out
        self.assertEqual(stats["tokens_before"], 2)
        self.assertEqual(stats["tokens_after"], 1)
        self.assertEqual(stats["tokens_removed"], 1)


if __name__ == "__main__":
    unittest.main()
