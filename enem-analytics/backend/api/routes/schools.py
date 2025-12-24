"""
School endpoints for ENEM Analytics API
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import pandas as pd

router = APIRouter()


class SchoolScore(BaseModel):
    ano: int
    nota_cn: Optional[float] = None
    nota_ch: Optional[float] = None
    nota_lc: Optional[float] = None
    nota_mt: Optional[float] = None
    nota_redacao: Optional[float] = None
    nota_media: Optional[float] = None
    ranking_brasil: Optional[int] = None


class SchoolSummary(BaseModel):
    codigo_inep: str
    nome_escola: str
    uf: Optional[str] = None
    ultimo_ranking: Optional[int] = None
    ultima_nota: Optional[float] = None
    anos_participacao: int


class SchoolDetail(BaseModel):
    codigo_inep: str
    nome_escola: str
    uf: Optional[str] = None
    historico: List[SchoolScore]
    tendencia: Optional[str] = None
    melhor_ano: Optional[int] = None
    melhor_ranking: Optional[int] = None


def get_df():
    """Import here to avoid circular imports"""
    from api.main import get_dataframe
    return get_dataframe()


@router.get("/", response_model=List[SchoolSummary])
async def list_schools(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = None,
    uf: Optional[str] = None,
    ano: Optional[int] = None,
    order_by: str = Query("ranking", regex="^(ranking|nota|nome)$"),
    order: str = Query("asc", regex="^(asc|desc)$")
):
    """
    List schools with pagination and filtering
    """
    df = get_df()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # Filter by year (default to most recent)
    target_ano = ano or int(df["ano"].max())
    df_year = df[df["ano"] == target_ano].copy()

    # Apply filters
    if search:
        search_lower = search.lower()
        df_year = df_year[
            df_year["nome_escola"].str.lower().str.contains(search_lower, na=False) |
            df_year["codigo_inep"].str.contains(search, na=False)
        ]

    if uf:
        df_year = df_year[df_year["uf"] == uf.upper()]

    # Sort
    if order_by == "ranking":
        sort_col = "ranking_brasil"
        df_year = df_year.dropna(subset=["ranking_brasil"])
    elif order_by == "nota":
        sort_col = "nota_media"
    else:
        sort_col = "nome_escola"

    ascending = order == "asc"
    df_year = df_year.sort_values(sort_col, ascending=ascending)

    # Pagination
    total = len(df_year)
    start = (page - 1) * limit
    end = start + limit
    df_page = df_year.iloc[start:end]

    # Build response
    results = []
    for _, row in df_page.iterrows():
        anos_count = int(df[df["codigo_inep"] == row["codigo_inep"]]["ano"].nunique())
        results.append(SchoolSummary(
            codigo_inep=str(row["codigo_inep"]),
            nome_escola=str(row["nome_escola"]),
            uf=str(row["uf"]) if pd.notna(row.get("uf")) else None,
            ultimo_ranking=int(row["ranking_brasil"]) if pd.notna(row.get("ranking_brasil")) else None,
            ultima_nota=float(round(row["nota_media"], 2)) if pd.notna(row.get("nota_media")) else None,
            anos_participacao=anos_count
        ))

    return results


@router.get("/top")
async def get_top_schools(
    ano: Optional[int] = None,
    limit: int = Query(10, ge=1, le=100),
    uf: Optional[str] = None
):
    """
    Get top ranked schools
    """
    df = get_df()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    target_ano = ano or int(df["ano"].max())
    df_year = df[df["ano"] == target_ano].copy()

    if uf:
        df_year = df_year[df_year["uf"] == uf.upper()]

    # Sort by ranking
    df_year = df_year.dropna(subset=["ranking_brasil"])
    df_year = df_year.sort_values("ranking_brasil", ascending=True)
    df_top = df_year.head(limit)

    results = []
    for _, row in df_top.iterrows():
        results.append({
            "ranking": int(row["ranking_brasil"]),
            "codigo_inep": str(row["codigo_inep"]),
            "nome_escola": str(row["nome_escola"]),
            "uf": str(row["uf"]) if pd.notna(row.get("uf")) else None,
            "nota_media": float(round(row["nota_media"], 2)) if pd.notna(row.get("nota_media")) else None,
            "nota_cn": float(round(row["nota_cn"], 2)) if pd.notna(row.get("nota_cn")) else None,
            "nota_ch": float(round(row["nota_ch"], 2)) if pd.notna(row.get("nota_ch")) else None,
            "nota_lc": float(round(row["nota_lc"], 2)) if pd.notna(row.get("nota_lc")) else None,
            "nota_mt": float(round(row["nota_mt"], 2)) if pd.notna(row.get("nota_mt")) else None,
            "nota_redacao": float(round(row["nota_redacao"], 2)) if pd.notna(row.get("nota_redacao")) else None,
        })

    return {
        "ano": target_ano,
        "total": len(results),
        "schools": results
    }


@router.get("/search")
async def search_schools(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Quick search for schools by name or INEP code
    """
    df = get_df()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    q_lower = q.lower()

    # Get most recent year for each school
    df_latest = df.loc[df.groupby("codigo_inep")["ano"].idxmax()]

    # Search
    matches = df_latest[
        df_latest["nome_escola"].str.lower().str.contains(q_lower, na=False) |
        df_latest["codigo_inep"].str.contains(q, na=False)
    ]

    matches = matches.head(limit)

    return [
        {
            "codigo_inep": str(row["codigo_inep"]),
            "nome_escola": str(row["nome_escola"]),
            "uf": str(row["uf"]) if pd.notna(row.get("uf")) else None,
            "ultimo_ano": int(row["ano"])
        }
        for _, row in matches.iterrows()
    ]


@router.get("/{codigo_inep}", response_model=SchoolDetail)
async def get_school(codigo_inep: str):
    """
    Get detailed information for a specific school
    """
    df = get_df()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    df_school = df[df["codigo_inep"] == codigo_inep].copy()

    if df_school.empty:
        raise HTTPException(status_code=404, detail=f"School {codigo_inep} not found")

    df_school = df_school.sort_values("ano")

    # Build history
    historico = []
    for _, row in df_school.iterrows():
        historico.append(SchoolScore(
            ano=int(row["ano"]),
            nota_cn=float(round(row["nota_cn"], 2)) if pd.notna(row.get("nota_cn")) else None,
            nota_ch=float(round(row["nota_ch"], 2)) if pd.notna(row.get("nota_ch")) else None,
            nota_lc=float(round(row["nota_lc"], 2)) if pd.notna(row.get("nota_lc")) else None,
            nota_mt=float(round(row["nota_mt"], 2)) if pd.notna(row.get("nota_mt")) else None,
            nota_redacao=float(round(row["nota_redacao"], 2)) if pd.notna(row.get("nota_redacao")) else None,
            nota_media=float(round(row["nota_media"], 2)) if pd.notna(row.get("nota_media")) else None,
            ranking_brasil=int(row["ranking_brasil"]) if pd.notna(row.get("ranking_brasil")) else None
        ))

    # Calculate trend
    if len(df_school) >= 2:
        recent = df_school.tail(3)
        if len(recent) >= 2 and "nota_media" in recent.columns:
            first_nota = recent.iloc[0]["nota_media"]
            last_nota = recent.iloc[-1]["nota_media"]
            if pd.notna(first_nota) and pd.notna(last_nota):
                diff = last_nota - first_nota
                if diff > 10:
                    tendencia = "subindo"
                elif diff < -10:
                    tendencia = "descendo"
                else:
                    tendencia = "estável"
            else:
                tendencia = None
        else:
            tendencia = None
    else:
        tendencia = None

    # Best year
    df_with_ranking = df_school.dropna(subset=["ranking_brasil"])
    if not df_with_ranking.empty:
        best_idx = df_with_ranking["ranking_brasil"].idxmin()
        melhor_ano = int(df_with_ranking.loc[best_idx, "ano"])
        melhor_ranking = int(df_with_ranking.loc[best_idx, "ranking_brasil"])
    else:
        melhor_ano = None
        melhor_ranking = None

    latest = df_school.iloc[-1]

    return SchoolDetail(
        codigo_inep=str(codigo_inep),
        nome_escola=str(latest["nome_escola"]),
        uf=str(latest["uf"]) if pd.notna(latest.get("uf")) else None,
        historico=historico,
        tendencia=tendencia,
        melhor_ano=melhor_ano,
        melhor_ranking=melhor_ranking
    )


@router.get("/{codigo_inep}/history")
async def get_school_history(codigo_inep: str):
    """
    Get complete history for a school with year-over-year comparison
    """
    df = get_df()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    df_school = df[df["codigo_inep"] == codigo_inep].copy()

    if df_school.empty:
        raise HTTPException(status_code=404, detail=f"School {codigo_inep} not found")

    df_school = df_school.sort_values("ano")
    latest = df_school.iloc[-1]

    history = []
    prev_ranking = None
    prev_nota = None

    for _, row in df_school.iterrows():
        ranking = int(row["ranking_brasil"]) if pd.notna(row.get("ranking_brasil")) else None
        nota = float(round(row["nota_media"], 2)) if pd.notna(row.get("nota_media")) else None

        # Calculate year-over-year changes
        ranking_change = None
        nota_change = None

        if ranking is not None and prev_ranking is not None:
            ranking_change = prev_ranking - ranking  # Positive = improved

        if nota is not None and prev_nota is not None:
            nota_change = float(round(nota - prev_nota, 2))

        history.append({
            "ano": int(row["ano"]),
            "ranking_brasil": ranking,
            "ranking_change": ranking_change,
            "nota_media": nota,
            "nota_change": nota_change,
            "nota_cn": float(round(row["nota_cn"], 2)) if pd.notna(row.get("nota_cn")) else None,
            "nota_ch": float(round(row["nota_ch"], 2)) if pd.notna(row.get("nota_ch")) else None,
            "nota_lc": float(round(row["nota_lc"], 2)) if pd.notna(row.get("nota_lc")) else None,
            "nota_mt": float(round(row["nota_mt"], 2)) if pd.notna(row.get("nota_mt")) else None,
            "nota_redacao": float(round(row["nota_redacao"], 2)) if pd.notna(row.get("nota_redacao")) else None,
        })

        prev_ranking = ranking
        prev_nota = nota

    return {
        "codigo_inep": str(codigo_inep),
        "nome_escola": str(latest["nome_escola"]),
        "uf": str(latest["uf"]) if pd.notna(latest.get("uf")) else None,
        "anos_participacao": len(history),
        "history": history
    }


@router.get("/compare/{inep1}/{inep2}")
async def compare_schools(inep1: str, inep2: str):
    """
    Compare two schools side by side
    """
    df = get_df()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    df1 = df[df["codigo_inep"] == inep1]
    df2 = df[df["codigo_inep"] == inep2]

    if df1.empty:
        raise HTTPException(status_code=404, detail=f"School {inep1} not found")
    if df2.empty:
        raise HTTPException(status_code=404, detail=f"School {inep2} not found")

    # Get common years
    years1 = set(int(y) for y in df1["ano"].tolist())
    years2 = set(int(y) for y in df2["ano"].tolist())
    common_years = sorted(years1 & years2)

    comparison = []
    for year in common_years:
        row1 = df1[df1["ano"] == year].iloc[0]
        row2 = df2[df2["ano"] == year].iloc[0]

        comparison.append({
            "ano": int(year),
            "escola1": {
                "nota_media": float(round(row1["nota_media"], 2)) if pd.notna(row1.get("nota_media")) else None,
                "ranking": int(row1["ranking_brasil"]) if pd.notna(row1.get("ranking_brasil")) else None,
            },
            "escola2": {
                "nota_media": float(round(row2["nota_media"], 2)) if pd.notna(row2.get("nota_media")) else None,
                "ranking": int(row2["ranking_brasil"]) if pd.notna(row2.get("ranking_brasil")) else None,
            }
        })

    latest1 = df1.iloc[-1]
    latest2 = df2.iloc[-1]

    return {
        "escola1": {
            "codigo_inep": str(inep1),
            "nome_escola": str(latest1["nome_escola"]),
            "uf": str(latest1["uf"]) if pd.notna(latest1.get("uf")) else None
        },
        "escola2": {
            "codigo_inep": str(inep2),
            "nome_escola": str(latest2["nome_escola"]),
            "uf": str(latest2["uf"]) if pd.notna(latest2.get("uf")) else None
        },
        "common_years": common_years,
        "comparison": comparison
    }
