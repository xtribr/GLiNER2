"""
Extrator completo de dados do ENEM por Escola
Observatório ENEM - SAS Educação via PowerBI API
"""

import requests
import json
import pandas as pd
from typing import List, Dict, Optional
import time

# Configuração
API_URL = "https://wabi-brazil-south-b-primary-api.analysis.windows.net/public/reports/querydata?synchronous=true"
MODEL_ID = 6606248
DATASET_ID = "68135bd6-e83c-419b-90e2-64bdb5553961"
REPORT_ID = "e93cdfb7-4848-4123-b9f7-b075ec852dfc"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://app.powerbi.com",
    "Referer": "https://app.powerbi.com/",
    "X-PowerBI-ResourceKey": "95131326-e9de-47c4-8bf5-12c27c1e113a",
}


def fetch_schools(ano: int = 2024, limit: int = 30000) -> List[str]:
    """Busca lista de todas as escolas"""
    query = {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [{"Name": "s", "Entity": "stg-senso-escolar", "Type": 0}],
                            "Select": [{
                                "Column": {
                                    "Expression": {"SourceRef": {"Source": "s"}},
                                    "Property": "inep_nome"
                                },
                                "Name": "stg-senso-escolar.inep_nome"
                            }],
                            "Where": [{
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
                            }]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0]}]},
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

    response = requests.post(API_URL, headers=HEADERS, json=query)
    response.raise_for_status()
    data = response.json()

    # Extrair nomes das escolas
    schools = []
    try:
        ds = data["results"][0]["result"]["data"]["dsr"]["DS"][0]
        for item in ds.get("PH", []):
            for row in item.get("DM0", []):
                if "G0" in row:
                    schools.append(row["G0"])
    except (KeyError, IndexError) as e:
        print(f"Erro ao extrair escolas: {e}")

    return schools


def fetch_rankings(ano: int = 2024, comparacao: str = "Brasil", limit: int = 30000) -> List[Dict]:
    """Busca rankings de todas as escolas"""
    query = {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0},
                                {"Name": "c", "Entity": "cur-enem-school-tri-area-rank", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0},
                                {"Name": "e", "Entity": "enem_area", "Type": 0}
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
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "c"}},
                                                "Property": "tri_area"
                                            }
                                        },
                                        "Function": 1  # Sum
                                    },
                                    "Name": "nota_tri"
                                },
                                {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "c"}},
                                        "Property": "_rank_area_escola"
                                    },
                                    "Name": "ranking"
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
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "c"}},
                                                "Property": "tri_area"
                                            }
                                        },
                                        "Function": 1
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

    response = requests.post(API_URL, headers=HEADERS, json=query)
    response.raise_for_status()
    return response.json()


def extract_all_data(ano: int = 2024):
    """Extrai todos os dados e salva em CSV"""
    print(f"=== Extrator ENEM {ano} ===\n")

    # 1. Buscar lista de escolas
    print("1. Buscando lista de escolas...")
    schools = fetch_schools(ano=ano, limit=30000)
    print(f"   Encontradas: {len(schools)} escolas\n")

    # Salvar lista de escolas
    df_schools = pd.DataFrame({"inep_nome": schools})
    df_schools["codigo_inep"] = df_schools["inep_nome"].str.split("-").str[0]
    df_schools["nome_escola"] = df_schools["inep_nome"].str.split("-", n=1).str[1]
    df_schools.to_csv(f"escolas_enem_{ano}.csv", index=False)
    print(f"   Salvo: escolas_enem_{ano}.csv")

    # 2. Buscar rankings
    print("\n2. Buscando rankings...")
    rankings_data = fetch_rankings(ano=ano, limit=30000)

    # Salvar JSON bruto
    with open(f"rankings_enem_{ano}_raw.json", "w") as f:
        json.dump(rankings_data, f, indent=2, ensure_ascii=False)
    print(f"   Salvo: rankings_enem_{ano}_raw.json")

    # Mostrar preview
    print("\n3. Preview dos dados:")
    print(json.dumps(rankings_data, indent=2, ensure_ascii=False)[:3000])

    return schools, rankings_data


if __name__ == "__main__":
    schools, rankings = extract_all_data(2024)
    print(f"\n\nTotal de escolas extraídas: {len(schools)}")
