#!/usr/bin/env python3
"""Aggregate redacao competencies (C1-C5) per school from INEP microdata.

The official ingestion pipeline (update_enem_year.py) averages the five
competencies into a single value and discards the individual ones. This
script re-reads the raw microdata and persists per-competency means, so
the product can surface which competency is a school's gargalo.

Output: data/competencias_redacao_por_escola_{year}.csv with columns:
    codigo_inep, ano, media_c1..c5, n_redacoes

Usage:
    python scripts/extract_competencias_redacao.py \
        --input ../../microdados-2024/MICRODADOS_ENEM_2024.csv \
        --year 2024 \
        --output data/competencias_redacao_por_escola_2024.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

COMP_COLUMNS = ["NU_NOTA_COMP1", "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4", "NU_NOTA_COMP5"]
USECOLS = ["NU_ANO", "CO_ESCOLA", "TP_STATUS_REDACAO", *COMP_COLUMNS]
CHUNK_ROWS = 200_000


def iter_chunks(path: Path) -> Iterator[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep=";",
        usecols=USECOLS,
        dtype={"CO_ESCOLA": "string"},
        chunksize=CHUNK_ROWS,
        encoding="latin1",
        low_memory=False,
    )


def aggregate(input_path: Path, year: int) -> pd.DataFrame:
    accumulator: dict[str, dict] = {}

    for chunk in iter_chunks(input_path):
        # Keep only rows for the requested year and only fully-graded essays.
        # TP_STATUS_REDACAO == 1 means "Sem problemas" (essay accepted and scored).
        chunk = chunk[chunk["NU_ANO"] == year]
        chunk = chunk[chunk["TP_STATUS_REDACAO"] == 1]
        chunk = chunk.dropna(subset=["CO_ESCOLA"])
        if chunk.empty:
            continue

        for inep, group in chunk.groupby("CO_ESCOLA"):
            slot = accumulator.setdefault(
                inep, {"sums": [0.0] * 5, "counts": [0] * 5, "n_redacoes": 0}
            )
            slot["n_redacoes"] += len(group)
            for idx, col in enumerate(COMP_COLUMNS):
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(values) > 0:
                    slot["sums"][idx] += float(values.sum())
                    slot["counts"][idx] += int(len(values))

    rows = []
    for inep, slot in accumulator.items():
        row = {"codigo_inep": inep, "ano": year, "n_redacoes": slot["n_redacoes"]}
        for idx in range(5):
            count = slot["counts"][idx]
            row[f"media_c{idx + 1}"] = round(slot["sums"][idx] / count, 2) if count else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values("codigo_inep").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to MICRODADOS CSV")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Reading {args.input}...")
    df = aggregate(args.input, args.year)
    print(f"Aggregated {len(df)} schools for year {args.year}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
