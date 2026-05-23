"""Compute a school's position relative to a single concept's TRI items.

Given a concept label (e.g. "Termologia"), this module finds every TRI
item where the concept appears and reports the difficulty band of those
items. Cross-referenced with the school's score in the dominant area,
it classifies the concept as one of:

- gargalo:     concept items are above the school's score → likely gap
- no_nivel:    school is within the concept's difficulty band
- dominado:    school already scores above the concept's items
- indefinido:  no score available for the school

The result powers the bottom-sheet shown when a coordinator clicks a
node in the Mapa de Conceitos.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd

ENTITY_COLUMNS = (
    "conceitos_cientificos",
    "campos_semanticos",
    "campos_lexicais",
    "processos_fenomenos",
    "contextos_historicos",
    "habilidades_compostas",
)
AREA_TO_SCORE_COL = {
    "CN": "nota_cn",
    "CH": "nota_ch",
    "LC": "nota_lc",
    "MT": "nota_mt",
}
GAP_THRESHOLD = 30.0  # pts; defines the "no_nivel" band around items


def _find_items_with_concept(
    tri_content_df: pd.DataFrame, concept_label: str
) -> pd.DataFrame:
    """Return rows where the concept appears in any entity column (case-insensitive)."""
    pattern = re.compile(r"(?:^|,\s*)" + re.escape(concept_label) + r"(?:\s*,|$)", re.IGNORECASE)
    mask = pd.Series([False] * len(tri_content_df), index=tri_content_df.index)
    for col in ENTITY_COLUMNS:
        if col not in tri_content_df.columns:
            continue
        series = tri_content_df[col].fillna("").astype(str)
        mask |= series.str.contains(pattern, regex=True, na=False)
    return tri_content_df[mask]


def _classify_status(school_score: Optional[float], tri_mean: float) -> str:
    if school_score is None:
        return "indefinido"
    gap = school_score - tri_mean
    if gap < -GAP_THRESHOLD:
        return "gargalo"
    if gap > GAP_THRESHOLD:
        return "dominado"
    return "no_nivel"


def get_concept_school_context(
    codigo_inep: str,
    concept_label: str,
    tri_content_df: pd.DataFrame,
    school_scores: Dict[str, Optional[float]],
) -> Optional[Dict[str, Any]]:
    """Compute everything needed to render the bottom-sheet.

    Args:
        codigo_inep: school id (passed through to result).
        concept_label: the concept the user clicked.
        tri_content_df: as returned by get_gliner_data().
        school_scores: dict like {"CN": 468.8, "CH": 479.8, ...}. Areas
            with no score map to None.

    Returns:
        Result dict, or None when the concept does not appear in any item.
    """
    items = _find_items_with_concept(tri_content_df, concept_label)
    items = items[items["tri_score"].notna()]
    if len(items) == 0:
        return None

    area_breakdown = []
    for area, group in items.groupby("area_code"):
        if area not in AREA_TO_SCORE_COL:
            continue
        scores = group["tri_score"].astype(float)
        area_breakdown.append({
            "area": area,
            "n_items": int(len(group)),
            "tri_mean": round(float(scores.mean()), 1),
            "tri_min": round(float(scores.min()), 1),
            "tri_max": round(float(scores.max()), 1),
        })

    if not area_breakdown:
        return None

    # Dominant area = highest n_items, tie-break by total tri sum.
    area_breakdown.sort(key=lambda x: (-x["n_items"], -x["tri_mean"]))
    dominant = area_breakdown[0]

    school_score = school_scores.get(dominant["area"])
    school_score = float(school_score) if school_score is not None and not pd.isna(school_score) else None

    status = _classify_status(school_score, dominant["tri_mean"])
    gap = round(school_score - dominant["tri_mean"], 1) if school_score is not None else None

    return {
        "codigo_inep": codigo_inep,
        "concept": concept_label,
        "n_items_total": int(len(items)),
        "dominant_area": dominant["area"],
        "tri_range": {
            "min": dominant["tri_min"],
            "max": dominant["tri_max"],
            "mean": dominant["tri_mean"],
        },
        "school_score": school_score,
        "gap": gap,
        "status": status,
        "areas": area_breakdown,
    }
