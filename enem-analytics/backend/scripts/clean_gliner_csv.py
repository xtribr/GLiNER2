#!/usr/bin/env python3
"""Clean conteudos_tri_gliner.csv: dedupe case-insensitive, drop sentences.

The audit panel was admitting "116 duplicatas" right in the UI. Most of
those are case-only variants (e.g. "Biomas Brasileiros" vs "Biomas
brasileiros") plus a handful of long phrases that ended up tagged as
"conceitos" by the GLiNER pipeline.

This pass collapses case-only variants to the most frequent capitalization
and drops tokens that don't look like concept labels (too long, whitespace-
only, etc.). Idempotent: re-running on a clean CSV is a no-op.

Usage:
    python scripts/clean_gliner_csv.py \\
        --input  data/conteudos_tri_gliner.csv \\
        --output data/conteudos_tri_gliner.csv  # in-place
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

ENTITY_COLUMNS = (
    "conceitos_cientificos",
    "campos_semanticos",
    "campos_lexicais",
    "processos_fenomenos",
    "contextos_historicos",
    "habilidades_compostas",
)
MAX_TOKEN_LEN = 50  # tokens longer than this look like sentences
WS_RE = re.compile(r"\s+")


def split_cell(cell: object) -> list[str]:
    if not isinstance(cell, str):
        return []
    return [t.strip() for t in cell.split(",") if t.strip()]


def normalize_token(token: str) -> str:
    """Collapse whitespace; preserve casing for now (resolved later)."""
    return WS_RE.sub(" ", token).strip()


def is_valid_token(token: str) -> bool:
    if len(token) < 3:
        return False
    if len(token) > MAX_TOKEN_LEN:
        return False
    # Must contain at least one letter.
    if not re.search(r"[A-Za-zÀ-ÿ]", token):
        return False
    return True


def build_canonical_map(df: pd.DataFrame) -> Dict[str, str]:
    """For each lower-cased token, pick the most frequent original casing.

    Tie-break: alphabetical (deterministic).
    """
    counts: Dict[str, Counter] = {}
    for col in ENTITY_COLUMNS:
        if col not in df.columns:
            continue
        for cell in df[col].dropna():
            for token in split_cell(cell):
                token = normalize_token(token)
                if not is_valid_token(token):
                    continue
                key = token.lower()
                counts.setdefault(key, Counter())[token] += 1

    canonical: Dict[str, str] = {}
    for key, variants in counts.items():
        best = sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        canonical[key] = best
    return canonical


def clean_cell(cell: object, canonical: Dict[str, str]) -> str | None:
    if not isinstance(cell, str):
        return cell
    out: list[str] = []
    seen: set[str] = set()
    for token in split_cell(cell):
        token = normalize_token(token)
        if not is_valid_token(token):
            continue
        canon = canonical.get(token.lower(), token)
        if canon.lower() in seen:
            continue
        seen.add(canon.lower())
        out.append(canon)
    if not out:
        return None
    return ", ".join(out)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    canonical = build_canonical_map(df)
    before_tokens = sum(
        len(split_cell(cell))
        for col in ENTITY_COLUMNS
        if col in df.columns
        for cell in df[col].dropna()
    )

    out = df.copy()
    for col in ENTITY_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = out[col].apply(lambda c: clean_cell(c, canonical))

    after_tokens = sum(
        len(split_cell(cell))
        for col in ENTITY_COLUMNS
        if col in out.columns
        for cell in out[col].dropna()
    )

    stats = {
        "rows": len(out),
        "tokens_before": before_tokens,
        "tokens_after": after_tokens,
        "tokens_removed": before_tokens - after_tokens,
        "unique_canonical": len(canonical),
    }
    return out, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input, quoting=1)
    cleaned, stats = clean_dataframe(df)

    cleaned.to_csv(args.output, index=False, quoting=1)
    print("Cleanup done:")
    for k, v in stats.items():
        print(f"  {k:20s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
