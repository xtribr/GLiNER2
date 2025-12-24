"""
Extrator Completo de Dados ENEM - Observatório SAS Educação
Extrai rankings, redação e habilidades de todas as 22.720 escolas via PowerBI API
"""

import requests
import json
import pandas as pd
import sqlite3
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime

# =============================================================================
# CONFIGURAÇÃO API
# =============================================================================

API_URL = "https://wabi-brazil-south-b-primary-api.analysis.windows.net/public/reports/querydata?synchronous=true"
MODEL_ID = 6606248
DATASET_ID = "68135bd6-e83c-419b-90e2-64bdb5553961"
REPORT_ID = "e93cdfb7-4848-4123-b9f7-b075ec852dfc"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-BR,pt;q=0.7",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://app.powerbi.com",
    "Referer": "https://app.powerbi.com/",
    "X-PowerBI-ResourceKey": "95131326-e9de-47c4-8bf5-12c27c1e113a",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


def api_request(query: dict, retries: int = 3) -> Optional[dict]:
    """Faz requisição para a API com retry"""
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=query, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Tentativa {attempt + 1}/{retries} falhou: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# =============================================================================
# QUERY BUILDERS
# =============================================================================

def build_rankings_query(ano: int = 2024, comparacao: str = "Brasil", limit: int = 30000) -> dict:
    """Query para extrair rankings de TODAS as escolas"""
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "e", "Entity": "enem_area", "Type": 0},
                                {"Name": "c", "Entity": "cur-enem-school-tri-area-rank", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0},
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "stg-senso-escolar.inep_nome"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "c"}},
                                                "Property": "tri_area"
                                            }
                                        },
                                        "Function": 1
                                    },
                                    "Name": "Sum(cur-enem-school-tri-area-rank.tri_area)"
                                },
                                {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "c"}},
                                        "Property": "Ranking_string"
                                    },
                                    "Name": "cur-enem-school-tri-area-rank.Ranking_string"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "f"}},
                                                    "Property": "nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "c1"}},
                                                    "Property": "Coluna 1"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"'{comparacao}'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "ano"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "e"}},
                                                    "Property": "area_filtro"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'MÉDIA'"}}]]
                                        }
                                    }
                                }
                            ],
                            "OrderBy": [{
                                "Direction": 1,
                                "Expression": {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "c"}},
                                        "Property": "Ranking_string"
                                    }
                                }
                            }]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                            "DataReduction": {
                                "DataVolume": 6,
                                "Primary": {"Window": {"Count": limit}}
                            },
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": DATASET_ID,
                "Sources": [{"ReportId": REPORT_ID}]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def build_all_areas_query(ano: int = 2024, limit: int = 30000, offset: int = 0) -> dict:
    """Query para extrair notas de TODAS as áreas de uma vez (escola + área + nota)"""
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "e", "Entity": "enem_area", "Type": 0},
                                {"Name": "c", "Entity": "cur-enem-school-tri-area-rank", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0},
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "escola"
                                },
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "e"}},
                                        "Property": "descricao"
                                    },
                                    "Name": "area"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "c"}},
                                                "Property": "tri_area"
                                            }
                                        },
                                        "Function": 1
                                    },
                                    "Name": "nota_tri"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "f"}},
                                                    "Property": "nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "c1"}},
                                                    "Property": "Coluna 1"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "ano"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                }
                            ],
                            "OrderBy": [{
                                "Direction": 1,
                                "Expression": {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    }
                                }
                            }]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                            "DataReduction": {
                                "DataVolume": 6,
                                "Primary": {"Window": {"Count": limit, "First": offset}}
                            },
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": DATASET_ID,
                "Sources": [{"ReportId": REPORT_ID}]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def build_redacao_query(ano: int = 2024, limit: int = 30000, offset: int = 0) -> dict:
    """Query para extrair notas de redação por competência"""
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0},
                                {"Name": "r", "Entity": "cur-enem-school-red-competence", "Type": 0},
                                {"Name": "d", "Entity": "dim_red", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "escola"
                                },
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "d"}},
                                        "Property": "bloco"
                                    },
                                    "Name": "competencia"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "r"}},
                                                "Property": "nota_value"
                                            }
                                        },
                                        "Function": 1
                                    },
                                    "Name": "nota"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "f"}},
                                                    "Property": "nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "c1"}},
                                                    "Property": "Coluna 1"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "ano"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                }
                            ],
                            "OrderBy": [{
                                "Direction": 1,
                                "Expression": {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    }
                                }
                            }]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                            "DataReduction": {
                                "DataVolume": 6,
                                "Primary": {"Window": {"Count": limit, "First": offset}}
                            },
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": DATASET_ID,
                "Sources": [{"ReportId": REPORT_ID}]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def build_habilidades_query(ano: int = 2024, limit: int = 50000) -> dict:
    """Query para extrair desempenho por habilidade"""
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0},
                                {"Name": "h", "Entity": "cur-enem-school-skill", "Type": 0},
                                {"Name": "d", "Entity": "dim_hab", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "escola"
                                },
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "d"}},
                                        "Property": "name"
                                    },
                                    "Name": "habilidade"
                                },
                                {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "h"}},
                                        "Property": "desempenho_hab"
                                    },
                                    "Name": "desempenho"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "f"}},
                                                    "Property": "nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "c1"}},
                                                    "Property": "Coluna 1"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "ano"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                }
                            ],
                            "OrderBy": [{
                                "Direction": 1,
                                "Expression": {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    }
                                }
                            }]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                            "DataReduction": {
                                "DataVolume": 6,
                                "Primary": {"Window": {"Count": limit}}
                            },
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": DATASET_ID,
                "Sources": [{"ReportId": REPORT_ID}]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


# =============================================================================
# PARSERS
# =============================================================================

def resolve_value_dict(value, value_dicts: dict, key: str = "D0"):
    """Resolve valor do dicionário de valores do PowerBI"""
    if isinstance(value, int) and key in value_dicts:
        dict_data = value_dicts[key]
        if isinstance(dict_data, list) and value < len(dict_data):
            return dict_data[value]
        elif isinstance(dict_data, dict):
            return dict_data.get(value, value)
    return value


def parse_rankings_response(response: dict) -> List[Dict]:
    """Parseia resposta de rankings"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        for item in ds.get("PH", []):
            for row in item.get("DM0", []):
                if "C" in row:
                    values = row["C"]
                    escola = values[0] if len(values) > 0 else None
                    escola = resolve_value_dict(escola, value_dicts, "D0")

                    nota_tri = values[1] if len(values) > 1 else None
                    ranking = values[2] if len(values) > 2 else None

                    if escola:
                        records.append({
                            "inep_nome": escola,
                            "nota_tri_media": nota_tri,
                            "ranking_brasil": ranking
                        })
    except (KeyError, IndexError, TypeError) as e:
        print(f"Erro ao parsear rankings: {e}")

    return records


def parse_area_response(response: dict) -> List[Dict]:
    """Parseia resposta de notas por área"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        for item in ds.get("PH", []):
            for row in item.get("DM0", []):
                if "C" in row:
                    values = row["C"]
                    escola = values[0] if len(values) > 0 else None
                    escola = resolve_value_dict(escola, value_dicts, "D0")
                    nota = values[1] if len(values) > 1 else None

                    if escola:
                        records.append({
                            "inep_nome": escola,
                            "nota": nota
                        })
    except (KeyError, IndexError, TypeError) as e:
        print(f"Erro ao parsear área: {e}")

    return records


def parse_redacao_response(response: dict) -> List[Dict]:
    """Parseia resposta de redação"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        for item in ds.get("PH", []):
            for row in item.get("DM0", []):
                if "C" in row:
                    values = row["C"]
                    escola = values[0] if len(values) > 0 else None
                    escola = resolve_value_dict(escola, value_dicts, "D0")

                    competencia = values[1] if len(values) > 1 else None
                    competencia = resolve_value_dict(competencia, value_dicts, "D1")
                    nota = values[2] if len(values) > 2 else None

                    if escola and competencia:
                        records.append({
                            "inep_nome": escola,
                            "competencia": competencia,
                            "nota": nota
                        })
    except (KeyError, IndexError, TypeError) as e:
        print(f"Erro ao parsear redação: {e}")

    return records


def parse_habilidades_response(response: dict) -> List[Dict]:
    """Parseia resposta de habilidades"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        for item in ds.get("PH", []):
            for row in item.get("DM0", []):
                if "C" in row:
                    values = row["C"]
                    escola = values[0] if len(values) > 0 else None
                    escola = resolve_value_dict(escola, value_dicts, "D0")

                    habilidade = values[1] if len(values) > 1 else None
                    habilidade = resolve_value_dict(habilidade, value_dicts, "D1")
                    desempenho = values[2] if len(values) > 2 else None

                    if escola and habilidade:
                        records.append({
                            "inep_nome": escola,
                            "habilidade": habilidade,
                            "desempenho": desempenho
                        })
    except (KeyError, IndexError, TypeError) as e:
        print(f"Erro ao parsear habilidades: {e}")

    return records


# =============================================================================
# EXTRAÇÃO PRINCIPAL
# =============================================================================

def fetch_rankings(ano: int = 2024) -> pd.DataFrame:
    """Extrai rankings de todas as escolas"""
    print("Extraindo rankings...")
    query = build_rankings_query(ano=ano)
    response = api_request(query)

    if not response:
        print("  Erro: sem resposta da API")
        return pd.DataFrame()

    # Verificar erros
    try:
        error = response["results"][0]["result"]["data"]["dsr"]["DataShapes"][0].get("odata.error")
        if error:
            print(f"  Erro da API: {error['message']['value']}")
            return pd.DataFrame()
    except (KeyError, IndexError):
        pass

    records = parse_rankings_response(response)
    print(f"  Encontradas {len(records)} escolas")

    df = pd.DataFrame(records)
    if not df.empty:
        df["codigo_inep"] = df["inep_nome"].str.split("-").str[0]
        df["nome_escola"] = df["inep_nome"].str.split("-", n=1).str[1]

    return df


def parse_areas_page(response: dict) -> List[Dict]:
    """Parse uma página de resultados de áreas"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        last_escola = None
        last_area = None

        for item in ds.get("PH", []):
            for row in item.get("DM0", []):
                if "C" in row:
                    values = row["C"]
                    r_flag = row.get("R", 0)

                    if r_flag & 1:  # escola repeated
                        escola = last_escola
                        if r_flag & 2:  # area also repeated
                            area = last_area
                            nota = values[0] if len(values) > 0 else None
                        else:
                            area = values[0] if len(values) > 0 else None
                            area = resolve_value_dict(area, value_dicts, "D1")
                            nota = values[1] if len(values) > 1 else None
                    else:
                        escola = values[0] if len(values) > 0 else None
                        escola = resolve_value_dict(escola, value_dicts, "D0")
                        if r_flag & 2:  # area repeated
                            area = last_area
                            nota = values[1] if len(values) > 1 else None
                        else:
                            area = values[1] if len(values) > 1 else None
                            area = resolve_value_dict(area, value_dicts, "D1")
                            nota = values[2] if len(values) > 2 else None

                    last_escola = escola
                    last_area = area

                    if escola and area:
                        records.append({
                            "inep_nome": escola,
                            "area": area,
                            "nota": float(nota) if nota else None
                        })

    except (KeyError, IndexError, TypeError) as e:
        pass

    return records


def build_areas_by_state_query(ano: int = 2024, uf_code: str = "35", limit: int = 30000) -> dict:
    """Query para extrair notas por área filtrando por estado (prefixo INEP)"""
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "e", "Entity": "enem_area", "Type": 0},
                                {"Name": "c", "Entity": "cur-enem-school-tri-area-rank", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0},
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "escola"
                                },
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "e"}},
                                        "Property": "descricao"
                                    },
                                    "Name": "area"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "c"}},
                                                "Property": "tri_area"
                                            }
                                        },
                                        "Function": 1
                                    },
                                    "Name": "nota"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "f"}},
                                                    "Property": "nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "c1"}},
                                                    "Property": "Coluna 1"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "ano"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "StartsWith": {
                                            "Left": {
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "inep_nome"
                                                }
                                            },
                                            "Right": {"Literal": {"Value": f"'{uf_code}'"}}
                                        }
                                    }
                                }
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                            "DataReduction": {
                                "DataVolume": 6,
                                "Primary": {"Window": {"Count": limit}}
                            },
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": DATASET_ID,
                "Sources": [{"ReportId": REPORT_ID}]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def fetch_notas_por_area(ano: int = 2024) -> pd.DataFrame:
    """Extrai notas por área de conhecimento (query por estado para contornar limite)"""
    print("Extraindo notas por área (por estado)...")

    # Códigos UF (IBGE - primeiros 2 dígitos do INEP)
    uf_codes = [
        "11", "12", "13", "14", "15", "16", "17",  # Norte
        "21", "22", "23", "24", "25", "26", "27", "28", "29",  # Nordeste
        "31", "32", "33", "35",  # Sudeste
        "41", "42", "43",  # Sul
        "50", "51", "52", "53"  # Centro-Oeste
    ]

    all_records = []

    for uf in uf_codes:
        print(f"  UF {uf}...", end=" ", flush=True)
        query = build_areas_by_state_query(ano=ano, uf_code=uf)
        response = api_request(query)

        if response:
            records = parse_areas_page(response)
            all_records.extend(records)
            print(f"{len(records)} registros")
        else:
            print("erro")

        time.sleep(0.3)  # Rate limiting

    print(f"  Total: {len(all_records)} registros")

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    # Pivot: uma linha por escola, uma coluna por área
    df_pivot = df.pivot_table(
        index="inep_nome",
        columns="area",
        values="nota",
        aggfunc="first"
    ).reset_index()
    df_pivot.columns.name = None

    # Renomear colunas para nomes mais curtos
    area_map = {
        "CIÊNCIAS DA NATUREZA E SUAS TECNOLOGIAS": "nota_cn",
        "CIÊNCIAS HUMANAS E SUAS TECNOLOGIAS": "nota_ch",
        "LINGUAGENS, CÓDIGOS E SUAS TECNOLOGIAS": "nota_lc",
        "MATEMÁTICA E SUAS TECNOLOGIAS": "nota_mt",
        "REDAÇÃO": "nota_redacao",
        "Ciências da Natureza": "nota_cn",
        "Ciências Humanas": "nota_ch",
        "Linguagens e Códigos": "nota_lc",
        "Matemática": "nota_mt",
        "Média das 5 áreas": "nota_media_5areas"
    }
    df_pivot = df_pivot.rename(columns=area_map)

    print(f"  {len(df_pivot)} escolas com notas por área")
    return df_pivot


def parse_redacao_page(response: dict) -> List[Dict]:
    """Parse uma página de resultados de redação"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        for item in ds.get("PH", []):
            for row in item.get("DM0", []):
                if "C" in row:
                    values = row["C"]
                    escola = values[0] if len(values) > 0 else None
                    escola = resolve_value_dict(escola, value_dicts, "D0")

                    competencia = values[1] if len(values) > 1 else None
                    competencia = resolve_value_dict(competencia, value_dicts, "D1")
                    nota = values[2] if len(values) > 2 else None

                    if escola and competencia:
                        records.append({
                            "inep_nome": escola,
                            "competencia": competencia,
                            "nota": nota
                        })
    except (KeyError, IndexError, TypeError):
        pass

    return records


def build_redacao_by_state_query(ano: int = 2024, uf_code: str = "35", limit: int = 30000) -> dict:
    """Query para extrair redação filtrando por estado (prefixo INEP)"""
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0},
                                {"Name": "r", "Entity": "cur-enem-school-red-competence", "Type": 0},
                                {"Name": "d", "Entity": "dim_red", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "escola"
                                },
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "d"}},
                                        "Property": "bloco"
                                    },
                                    "Name": "competencia"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "r"}},
                                                "Property": "nota_value"
                                            }
                                        },
                                        "Function": 1
                                    },
                                    "Name": "nota"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "f"}},
                                                    "Property": "nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "c1"}},
                                                    "Property": "Coluna 1"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "ano"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "StartsWith": {
                                            "Left": {
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "inep_nome"
                                                }
                                            },
                                            "Right": {"Literal": {"Value": f"'{uf_code}'"}}
                                        }
                                    }
                                }
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                            "DataReduction": {
                                "DataVolume": 6,
                                "Primary": {"Window": {"Count": limit}}
                            },
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {
                "DatasetId": DATASET_ID,
                "Sources": [{"ReportId": REPORT_ID}]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def fetch_redacao(ano: int = 2024) -> pd.DataFrame:
    """Extrai notas de redação por competência (query por estado)"""
    print("Extraindo redação (por estado)...")

    uf_codes = [
        "11", "12", "13", "14", "15", "16", "17",
        "21", "22", "23", "24", "25", "26", "27", "28", "29",
        "31", "32", "33", "35",
        "41", "42", "43",
        "50", "51", "52", "53"
    ]

    all_records = []

    for uf in uf_codes:
        print(f"  UF {uf}...", end=" ", flush=True)
        query = build_redacao_by_state_query(ano=ano, uf_code=uf)
        response = api_request(query)

        if response:
            records = parse_redacao_page(response)
            all_records.extend(records)
            print(f"{len(records)} registros")
        else:
            print("erro")

        time.sleep(0.3)

    print(f"  Total: {len(all_records)} registros")

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    # Pivot para ter uma coluna por competência
    df_pivot = df.pivot_table(
        index="inep_nome",
        columns="competencia",
        values="nota",
        aggfunc="first"
    ).reset_index()
    df_pivot.columns.name = None

    print(f"  {len(df_pivot)} escolas com notas de redação")
    return df_pivot


def fetch_habilidades(ano: int = 2024) -> pd.DataFrame:
    """Extrai desempenho por habilidade"""
    print("Extraindo habilidades...")
    query = build_habilidades_query(ano=ano)
    response = api_request(query)

    if not response:
        print("  Erro: sem resposta da API")
        return pd.DataFrame()

    # Check for API errors
    try:
        error = response["results"][0]["result"]["data"]["dsr"]["DataShapes"][0].get("odata.error")
        if error:
            print(f"  API Error: {error['message']['value'][:100]}...")
            print("  (Habilidades query não disponível - campos podem ter sido renomeados)")
            return pd.DataFrame()
    except (KeyError, IndexError):
        pass

    records = parse_habilidades_response(response)
    print(f"  Encontrados {len(records)} registros")

    return pd.DataFrame(records)


# =============================================================================
# EXTRAÇÃO COMPLETA
# =============================================================================

def extract_all(ano: int = 2024, output_prefix: str = "enem") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extrai todos os dados e salva em CSV/SQLite"""
    print("=" * 60)
    print(f"EXTRAÇÃO COMPLETA ENEM {ano}")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Rankings principais
    df_rankings = fetch_rankings(ano)
    if df_rankings.empty:
        print("ERRO: Falha ao extrair rankings")
        return pd.DataFrame(), pd.DataFrame()

    df_main = df_rankings.copy()

    # 2. Notas por área (uma query, todas as áreas)
    df_areas = fetch_notas_por_area(ano)
    if not df_areas.empty:
        df_main = df_main.merge(df_areas, on="inep_nome", how="left")

    # 3. Redação
    df_redacao = fetch_redacao(ano)
    if not df_redacao.empty:
        df_main = df_main.merge(df_redacao, on="inep_nome", how="left")

    # 4. Habilidades (tabela separada)
    df_habilidades = fetch_habilidades(ano)

    # Salvar CSVs
    csv_main = f"{output_prefix}_{ano}_completo.csv"
    csv_hab = f"{output_prefix}_{ano}_habilidades.csv"

    df_main.to_csv(csv_main, index=False)
    print(f"\nSalvo: {csv_main} ({len(df_main)} escolas)")

    if not df_habilidades.empty:
        df_habilidades.to_csv(csv_hab, index=False)
        print(f"Salvo: {csv_hab} ({len(df_habilidades)} registros)")

    # Salvar SQLite
    db_file = f"{output_prefix}_{ano}.db"
    conn = sqlite3.connect(db_file)
    df_main.to_sql("escolas", conn, if_exists="replace", index=False)
    if not df_habilidades.empty:
        df_habilidades.to_sql("habilidades", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Salvo: {db_file}")

    print("\n" + "=" * 60)
    print(f"EXTRAÇÃO COMPLETA!")
    print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return df_main, df_habilidades


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Testar extração
    df_main, df_hab = extract_all(ano=2024, output_prefix="enem")

    if not df_main.empty:
        print("\n--- PREVIEW DADOS PRINCIPAIS ---")
        print(df_main.head(10).to_string())
        print(f"\nColunas: {list(df_main.columns)}")

    if not df_hab.empty:
        print("\n--- PREVIEW HABILIDADES ---")
        print(df_hab.head(10).to_string())
