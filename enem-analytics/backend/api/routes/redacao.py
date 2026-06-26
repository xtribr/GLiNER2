"""Rota pública: análise ANÔNIMA da dupla correção da redação (ENEM 2025).

Serve agregados nacionais pré-computados (scripts/build_redacao_analysis.py).
Nenhum dado individual/identificável — é seguro como endpoint público.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/redacao", tags=["Redação"])


def _load_analysis():
    path = Path(__file__).parent.parent.parent / "data" / "redacao_2025_analysis.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/analysis")
async def get_redacao_analysis():
    """Análise agregada e anônima da redação (distribuição de notas + divergência entre avaliadores)."""
    data = _load_analysis()
    if not data:
        raise HTTPException(status_code=404, detail="Análise de redação não encontrada")
    return data
