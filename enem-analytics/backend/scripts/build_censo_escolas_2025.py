#!/usr/bin/env python3
"""Gera data/censo_escolas_2025.csv (metadados escolares por escola) a partir dos
microdados do Censo Escolar 2025.

Por que existe um script separado do 2024 (extract_censo_data.py):
- Em 2025 o INEP separou o microdado da escola em Tabela_Escola_2025 (caracteristicas)
  + Tabela_Matricula_2025 (contagens de matricula, ja agregadas por escola).
- O Novo Ensino Medio dividiu o EM em itinerarios formativos mutuamente exclusivos
  (PROP=propedeutico, IFTP_CT=tecnico/curso tecnico, IFTP_QP=tecnico/qualificacao
  profissional, NM=normal/magisterio). Os "concluintes" (3o/4o ano) sao a soma do
  3o e 4o ano EM TODOS os itinerarios. Usar apenas PROP+CT (como em 2024) subconta
  ~1.668 escolas que migraram para IFTP_QP.

Mantem a mesma definicao de porte/labels de extract_censo_data.py para coerencia
com os anos historicos.
"""
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from scripts.extract_censo_data import (  # noqa: E402
    DEPENDENCIA_LABELS,
    LOCALIZACAO_LABELS,
    PORTE_LABELS,
    calculate_porte,
)

CENSO_DIR = BACKEND.parent.parent / "microdados-2025" / "censo_2025_extract"
ESCOLA_FILE = CENSO_DIR / "Tabela_Escola_2025.csv"
MATRICULA_FILE = CENSO_DIR / "Tabela_Matricula_2025.csv"
OUTPUT_FILE = BACKEND / "data" / "censo_escolas_2025.csv"

# 3o + 4o ano somados nos quatro itinerarios formativos (mutuamente exclusivos).
CONCLUINTES_COLS = [
    "QT_MAT_MED_PROP_3", "QT_MAT_MED_PROP_4",        # propedeutico
    "QT_MAT_MED_IFTP_CT_3", "QT_MAT_MED_IFTP_CT_4",  # tecnico - curso tecnico
    "QT_MAT_MED_IFTP_QP_3", "QT_MAT_MED_IFTP_QP_4",  # tecnico - qualificacao profissional
    "QT_MAT_MED_NM_3", "QT_MAT_MED_NM_4",            # normal / magisterio
]


def main() -> int:
    escola = pd.read_csv(
        ESCOLA_FILE, sep=";", encoding="latin-1",
        usecols=["CO_ENTIDADE", "NO_ENTIDADE", "SG_UF", "TP_DEPENDENCIA",
                 "TP_LOCALIZACAO", "TP_SITUACAO_FUNCIONAMENTO"],
        dtype={"CO_ENTIDADE": str, "TP_DEPENDENCIA": "Int64",
               "TP_LOCALIZACAO": "Int64", "TP_SITUACAO_FUNCIONAMENTO": "Int64"},
    )
    escola = escola[escola["TP_SITUACAO_FUNCIONAMENTO"] == 1].copy()

    matricula = pd.read_csv(
        MATRICULA_FILE, sep=";", encoding="latin-1",
        usecols=["CO_ENTIDADE", "QT_MAT_MED", *CONCLUINTES_COLS],
        dtype={"CO_ENTIDADE": str, **{c: "Int64" for c in ["QT_MAT_MED", *CONCLUINTES_COLS]}},
    )

    df = escola.merge(matricula, on="CO_ENTIDADE", how="left")
    df["qt_concluintes"] = sum(df[c].fillna(0) for c in CONCLUINTES_COLS).astype("Int64")
    df["porte"] = df["qt_concluintes"].apply(calculate_porte)
    df["porte_label"] = df["porte"].map(PORTE_LABELS)
    df["localizacao"] = df["TP_LOCALIZACAO"].map(LOCALIZACAO_LABELS)
    df["dependencia"] = df["TP_DEPENDENCIA"].map(DEPENDENCIA_LABELS)

    df = df.rename(columns={
        "CO_ENTIDADE": "codigo_inep",
        "NO_ENTIDADE": "nome_escola_censo",
        "SG_UF": "uf",
        "QT_MAT_MED": "qt_matriculas_medio",
        "qt_concluintes": "qt_matriculas",
    })
    output = df[[
        "codigo_inep", "nome_escola_censo", "uf", "localizacao", "dependencia",
        "qt_matriculas", "qt_matriculas_medio", "porte", "porte_label",
    ]].copy()
    output.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(output)} active schools to {OUTPUT_FILE}")
    print(f"Schools with porte: {int(output['porte'].notna().sum())}")
    print(output["porte_label"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
