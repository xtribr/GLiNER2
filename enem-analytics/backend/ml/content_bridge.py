"""Bridge between TRI-difficulty content (GLiNER-extracted) and per-school scores.

Single responsibility: given a school's score in an area, surface the content
concepts that dominate the TRI items in that score's neighborhood. These are
the likely content gaps for that school.

Data source: data/conteudos_tri_gliner.csv (columns: area_code, tri_score,
conceitos_cientificos, campos_semanticos, campos_lexicais, processos_fenomenos,
contextos_historicos, habilidades_compostas).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

ENTITY_COLUMNS = (
    "conceitos_cientificos",
    "campos_semanticos",
    "campos_lexicais",
    "processos_fenomenos",
    "contextos_historicos",
    "habilidades_compostas",
)


def _explode_entities(cell: Any) -> List[str]:
    if not isinstance(cell, str):
        return []
    return [token.strip() for token in cell.split(",") if token.strip()]


def infer_content_gaps(
    area: str,
    score: float,
    tri_content_df: Optional[pd.DataFrame],
    band_width: float = 50.0,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Return top content concepts among TRI items near the school's score.

    Args:
        area: area code (CN, CH, LC, MT).
        score: the school's score in that area.
        tri_content_df: DataFrame as loaded from conteudos_tri_gliner.csv.
        band_width: half-width of the TRI score band to consider (score +/- band_width).
        top_n: maximum number of concepts to return.

    Returns:
        Dict with: band (list[float]), sample_size (int), top_concepts (list of
        {concept, count}). Empty top_concepts and zero sample_size when no data.
    """
    band = [float(score) - band_width, float(score) + band_width]
    empty = {"band": band, "sample_size": 0, "top_concepts": []}

    if tri_content_df is None or len(tri_content_df) == 0:
        return empty

    mask = (
        (tri_content_df["area_code"] == area)
        & (tri_content_df["tri_score"] >= band[0])
        & (tri_content_df["tri_score"] <= band[1])
    )
    rows = tri_content_df[mask]
    if len(rows) == 0:
        return empty

    counter: Counter = Counter()
    for col in ENTITY_COLUMNS:
        if col not in rows.columns:
            continue
        for cell in rows[col].dropna():
            for entity in _explode_entities(cell):
                counter[entity] += 1

    top = [{"concept": label, "count": count} for label, count in counter.most_common(top_n)]
    return {"band": band, "sample_size": int(len(rows)), "top_concepts": top}
