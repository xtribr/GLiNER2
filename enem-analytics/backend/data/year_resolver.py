"""Utilities for resolving year-based ENEM data files and time windows."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
SCHOOL_SKILLS_PATTERNS = (
    "desempenho_habilidades_*.csv",  # resumo legado por escola
    "school_skills_*.csv",  # desempenho por escola/área/habilidade
)


def extract_years_from_name(name: str) -> list[int]:
    """Return all ENEM-like years found in a file name."""
    return [int(match) for match in YEAR_PATTERN.findall(name)]


def get_file_year(path: Path | str) -> Optional[int]:
    """Return the maximum year present in a file path, if any."""
    years = extract_years_from_name(Path(path).name)
    return max(years) if years else None


def file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest without loading the whole file in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest_yeared_file(
    data_dir: Path | str,
    pattern: str,
    *,
    ignore_suffixes: Iterable[str] = (),
) -> Optional[Path]:
    """Resolve the newest file matching a glob pattern by year in its name."""
    base_path = Path(data_dir)
    candidates: list[tuple[int, Path]] = []

    for candidate in base_path.glob(pattern):
        if not candidate.is_file():
            continue
        if any(candidate.name.endswith(suffix) for suffix in ignore_suffixes):
            continue

        year = get_file_year(candidate)
        if year is None:
            continue

        candidates.append((year, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[-1][1]


def find_latest_enem_results_file(data_dir: Path | str) -> Optional[Path]:
    """Resolve the latest consolidated ENEM results CSV."""
    return find_latest_yeared_file(data_dir, "enem_*_completo.csv")


def find_latest_skills_file(data_dir: Path | str) -> Optional[Path]:
    """Resolve the latest national skills CSV."""
    return find_latest_yeared_file(data_dir, "habilidades_*.csv")


def find_latest_school_skills_file(data_dir: Path | str) -> Optional[Path]:
    """Resolve both supported school-skill schemas, preferring the newest year.

    If both schemas exist for the same year, the granular ``school_skills``
    export wins because it contains area/skill-level observations.
    """
    candidates: list[tuple[int, int, Path]] = []
    for priority, pattern in enumerate(SCHOOL_SKILLS_PATTERNS):
        candidate = find_latest_yeared_file(data_dir, pattern)
        if candidate is None:
            continue
        year = get_file_year(candidate)
        if year is not None:
            candidates.append((year, priority, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    return candidates[-1][2]


def find_latest_oracle_predictions_file(data_dir: Path | str) -> Optional[Path]:
    """Resolve the latest base oracle predictions JSON."""
    return find_latest_yeared_file(data_dir, "predictions_*.json", ignore_suffixes=("_enriched.json",))


def get_latest_year_from_df(df: Optional[pd.DataFrame]) -> Optional[int]:
    """Return the latest ENEM year present in a dataframe."""
    if df is None or df.empty or "ano" not in df.columns:
        return None

    years = pd.to_numeric(df["ano"], errors="coerce").dropna()
    if years.empty:
        return None

    return int(years.max())


def get_previous_year_from_df(df: Optional[pd.DataFrame], latest_year: Optional[int] = None) -> Optional[int]:
    """Return the previous available ENEM year before the latest one."""
    if df is None or df.empty or "ano" not in df.columns:
        return None

    years = sorted({int(year) for year in pd.to_numeric(df["ano"], errors="coerce").dropna()})
    if not years:
        return None

    latest = latest_year if latest_year is not None else years[-1]
    previous_years = [year for year in years if year < latest]
    return previous_years[-1] if previous_years else None
