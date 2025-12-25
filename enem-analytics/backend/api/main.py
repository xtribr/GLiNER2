"""
ENEM Analytics API
FastAPI backend for ENEM school data analysis and predictions
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path

from api.routes import schools, predictions, diagnosis, clusters, recommendations, tri_lists, gliner_insights

# Global data store
data_store = {}


def load_data():
    """Load ENEM data from CSV into memory"""
    data_path = Path(__file__).parent.parent / "data" / "enem_2018_2024_completo.csv"

    df = pd.read_csv(data_path)

    # Clean and prepare data
    df["codigo_inep"] = df["codigo_inep"].astype(str)
    df["ano"] = df["ano"].astype(int)

    # Extract state code from INEP (first 2 digits)
    df["uf_code"] = df["codigo_inep"].str[:2]

    # Map UF codes to state names
    uf_map = {
        "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
        "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
        "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
        "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
        "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
        "52": "GO", "53": "DF"
    }
    df["uf"] = df["uf_code"].map(uf_map)

    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup"""
    print("Loading ENEM data...")
    data_store["df"] = load_data()
    print(f"Loaded {len(data_store['df']):,} records")
    print(f"Years: {sorted(data_store['df']['ano'].unique())}")
    print(f"Schools: {data_store['df']['codigo_inep'].nunique():,}")
    yield
    data_store.clear()


app = FastAPI(
    title="ENEM Analytics API",
    description="API para análise de dados do ENEM por escola",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(schools.router, prefix="/api/schools", tags=["Schools"])
app.include_router(predictions.router)
app.include_router(diagnosis.router)
app.include_router(clusters.router)
app.include_router(recommendations.router)
app.include_router(tri_lists.router)
app.include_router(gliner_insights.router, prefix="/api/gliner", tags=["GLiNER Insights"])


@app.get("/")
async def root():
    return {
        "name": "ENEM Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "schools": "/api/schools",
            "docs": "/docs"
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Get general statistics"""
    df = data_store["df"]

    return {
        "total_records": len(df),
        "total_schools": df["codigo_inep"].nunique(),
        "years": sorted(df["ano"].unique().tolist()),
        "states": sorted(df["uf"].dropna().unique().tolist()),
        "avg_scores": {
            "nota_cn": round(df["nota_cn"].mean(), 2),
            "nota_ch": round(df["nota_ch"].mean(), 2),
            "nota_lc": round(df["nota_lc"].mean(), 2),
            "nota_mt": round(df["nota_mt"].mean(), 2),
            "nota_redacao": round(df["nota_redacao"].mean(), 2),
        }
    }


def get_dataframe():
    """Get the loaded dataframe"""
    return data_store.get("df")
