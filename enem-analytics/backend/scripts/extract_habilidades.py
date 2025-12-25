"""
Extrator de Habilidades e Dados Complementares - ENEM
Extrai dados de habilidades, competências de redação e censo escolar
"""

import requests
import json
import pandas as pd
from typing import Optional, List, Dict
import time

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

# UF codes for state-based queries
UF_CODES = [
    "11", "12", "13", "14", "15", "16", "17",  # Norte
    "21", "22", "23", "24", "25", "26", "27", "28", "29",  # Nordeste
    "31", "32", "33", "35",  # Sudeste
    "41", "42", "43",  # Sul
    "50", "51", "52", "53"  # Centro-Oeste
]


def api_request(query: dict, retries: int = 3) -> Optional[dict]:
    """Make API request with retry"""
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=query, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def build_habilidades_query(ano: int, uf_code: str, area: str = "MÉDIA") -> dict:
    """
    Query para extrair habilidades por escola
    """
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
                                {"Name": "c", "Entity": "cur-enem-school-skill", "Type": 0},
                                {"Name": "h", "Entity": "dim_hab", "Type": 0},
                                {"Name": "e", "Entity": "enem_area", "Type": 0},
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
                                        "Expression": {"SourceRef": {"Source": "h"}},
                                        "Property": "co_habilidade"
                                    },
                                    "Name": "habilidade"
                                },
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "h"}},
                                        "Property": "descricao"
                                    },
                                    "Name": "descricao_hab"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "c"}},
                                                "Property": "desempenho_hab"
                                            }
                                        },
                                        "Function": 1  # Sum
                                    },
                                    "Name": "desempenho"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "ano"}}],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": "nome"}}],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "c1"}}, "Property": "Coluna 1"}}],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "StartsWith": {
                                            "Left": {"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "inep_nome"}},
                                            "Right": {"Literal": {"Value": f"'{uf_code}'"}}
                                        }
                                    }
                                }
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2, 3]}]},
                            "DataReduction": {"DataVolume": 6, "Primary": {"Window": {"Count": 50000}}},
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {"DatasetId": DATASET_ID, "Sources": [{"ReportId": REPORT_ID}]}
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def build_censo_query(ano: int, uf_code: str) -> dict:
    """
    Query para extrair dados do censo escolar (tipo de escola)
    """
    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
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
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "tipo_escolar"
                                    },
                                    "Name": "tipo"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "ano"}}],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "StartsWith": {
                                            "Left": {"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "inep_nome"}},
                                            "Right": {"Literal": {"Value": f"'{uf_code}'"}}
                                        }
                                    }
                                }
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1]}]},
                            "DataReduction": {"DataVolume": 6, "Primary": {"Window": {"Count": 30000}}},
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {"DatasetId": DATASET_ID, "Sources": [{"ReportId": REPORT_ID}]}
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def build_redacao_competencias_query(ano: int, uf_code: str) -> dict:
    """
    Query para extrair competências de redação por escola
    """
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
                                {"Name": "c", "Entity": "cur-enem-school-red-competence", "Type": 0},
                                {"Name": "d", "Entity": "dim_competencia", "Type": 0},
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
                                        "Property": "co_competencia"
                                    },
                                    "Name": "competencia"
                                },
                                {
                                    "Aggregation": {
                                        "Expression": {
                                            "Column": {
                                                "Expression": {"SourceRef": {"Source": "c"}},
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
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "ano"}}],
                                            "Values": [[{"Literal": {"Value": f"{ano}L"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": "nome"}}],
                                            "Values": [[{"Literal": {"Value": "'Todas escolas'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "c1"}}, "Property": "Coluna 1"}}],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "StartsWith": {
                                            "Left": {"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "inep_nome"}},
                                            "Right": {"Literal": {"Value": f"'{uf_code}'"}}
                                        }
                                    }
                                }
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1, 2]}]},
                            "DataReduction": {"DataVolume": 6, "Primary": {"Window": {"Count": 50000}}},
                            "Version": 1
                        },
                        "ExecutionMetricsKind": 1
                    }
                }]
            },
            "QueryId": "",
            "ApplicationContext": {"DatasetId": DATASET_ID, "Sources": [{"ReportId": REPORT_ID}]}
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def parse_response(response: dict) -> List[Dict]:
    """Parse PowerBI response to extract data"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        for ph in ds.get("PH", []):
            for row in ph.get("DM0", []):
                if "C" in row:
                    records.append({"values": row["C"], "R": row.get("R")})
    except (KeyError, IndexError) as e:
        print(f"  Parse error: {e}")

    return records


def test_query():
    """Test a single query to see what data is available"""
    print("Testing habilidades query for CE (Ceará)...")

    query = build_habilidades_query(2024, "23")
    response = api_request(query)

    if response:
        # Save raw response
        with open("/tmp/hab_test.json", "w") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        print("Response saved to /tmp/hab_test.json")

        # Check for errors
        try:
            ds = response["results"][0]["result"]["data"]["dsr"]
            if "DataShapes" in ds:
                shapes = ds["DataShapes"]
                if shapes and "odata.error" in shapes[0]:
                    print(f"API Error: {shapes[0]['odata.error']['message']['value']}")
                    return

            # Try to parse
            records = parse_response(response)
            print(f"Found {len(records)} records")
            if records:
                print(f"First record: {records[0]}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No response")


if __name__ == "__main__":
    test_query()
