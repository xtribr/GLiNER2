"""
Extrator de dados do Observatório ENEM (SAS Educação)
Usa a API pública do PowerBI para extrair rankings de escolas
"""

import requests
import json
import pandas as pd
from typing import Optional

# Configuração da API
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
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
}


def build_query_all_schools(ano: int = 2024, comparacao: str = "Brasil", tamanho: str = "Todas escolas"):
    """
    Constrói query para buscar TODAS as escolas (sem filtro de escola específica)
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
                                {"Name": "e", "Entity": "enem_area", "Type": 0},
                                {"Name": "c", "Entity": "cur-enem-school-tri-area-rank", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0},
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0}
                            ],
                            "Select": [
                                # Escola
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "stg-senso-escolar.inep_nome",
                                    "NativeReferenceName": "Escola"
                                },
                                # Área do conhecimento
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "e"}},
                                        "Property": "descricao"
                                    },
                                    "Name": "enem_area.descricao",
                                    "NativeReferenceName": "Área"
                                },
                                # Nota TRI da Escola
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
                                    "Name": "Sum(cur-enem-school-tri-area-rank.tri_area)",
                                    "NativeReferenceName": "Nota TRI"
                                },
                                # Ranking
                                {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "c"}},
                                        "Property": "Ranking_string"
                                    },
                                    "Name": "cur-enem-school-tri-area-rank.Ranking_num",
                                    "NativeReferenceName": "Ranking"
                                }
                            ],
                            "Where": [
                                # Filtro tamanho
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "f"}},
                                                    "Property": "nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"'{tamanho}'"}}]]
                                        }
                                    }
                                },
                                # Comparação (Brasil/Estado/Cidade)
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
                                # Ano
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
                                # Área = MÉDIA (para pegar média geral)
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
                                "Direction": 1,  # Ascending (melhor ranking primeiro)
                                "Expression": {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "c"}},
                                        "Property": "Ranking_string"
                                    }
                                }
                            }]
                        },
                        "Binding": {
                            "Primary": {
                                "Groupings": [{"Projections": [0, 1, 2, 3]}]
                            },
                            "DataReduction": {
                                "DataVolume": 6,  # Aumentar volume de dados
                                "Primary": {
                                    "Window": {"Count": 30000}  # Pegar até 30k escolas
                                }
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
                "Sources": [{
                    "ReportId": REPORT_ID,
                    "VisualId": "9ba3aae0da00eec4d01c"
                }]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def build_query_single_school(inep_nome: str, ano: int = 2024):
    """
    Constrói query para buscar dados de uma escola específica (todas as áreas)
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
                                {"Name": "e", "Entity": "enem_area", "Type": 0},
                                {"Name": "c", "Entity": "cur-enem-school-tri-area-rank", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c1", "Entity": "comparação", "Type": 0},
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "e"}},
                                        "Property": "descricao"
                                    },
                                    "Name": "enem_area.descricao",
                                    "NativeReferenceName": "Área do Conhecimento"
                                },
                                {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "c"}},
                                        "Property": "_tri_area_comparacao"
                                    },
                                    "Name": "cur-enem-school-tri-area-rank._tri_area_comparacao",
                                    "NativeReferenceName": "Nota TRI da Comparação"
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
                                    "Name": "Sum(cur-enem-school-tri-area-rank.tri_area)",
                                    "NativeReferenceName": "Nota TRI da Escola"
                                },
                                {
                                    "Measure": {
                                        "Expression": {"SourceRef": {"Source": "c"}},
                                        "Property": "Ranking_string"
                                    },
                                    "Name": "cur-enem-school-tri-area-rank.Ranking_num",
                                    "NativeReferenceName": "Desempenho"
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
                                        "In": {
                                            "Expressions": [{
                                                "Column": {
                                                    "Expression": {"SourceRef": {"Source": "s"}},
                                                    "Property": "inep_nome"
                                                }
                                            }],
                                            "Values": [[{"Literal": {"Value": f"'{inep_nome}'"}}]]
                                        }
                                    }
                                }
                            ],
                            "OrderBy": [{
                                "Direction": 2,
                                "Expression": {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "e"}},
                                        "Property": "descricao"
                                    }
                                }
                            }]
                        },
                        "Binding": {
                            "Primary": {
                                "Groupings": [{"Projections": [0, 1, 2, 3]}]
                            },
                            "DataReduction": {
                                "DataVolume": 3,
                                "Primary": {"Window": {"Count": 500}}
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
                "Sources": [{
                    "ReportId": REPORT_ID,
                    "VisualId": "9ba3aae0da00eec4d01c"
                }]
            }
        }],
        "cancelQueries": [],
        "modelId": MODEL_ID
    }


def parse_powerbi_response(response_json):
    """
    Parseia a resposta do PowerBI e extrai os dados
    """
    try:
        results = response_json.get("results", [])
        if not results:
            return None

        data = results[0].get("result", {}).get("data", {})
        dsr = data.get("dsr", {})
        ds = dsr.get("DS", [])

        if not ds:
            return None

        # Extrair dados
        records = []
        for dataset in ds:
            # Pegar os valores
            ph = dataset.get("PH", [])
            for item in ph:
                dm = item.get("DM0", [])
                for row in dm:
                    record = {}
                    # Os campos "C" contêm os valores
                    if "C" in row:
                        record["values"] = row["C"]
                    if "R" in row:
                        record["reference"] = row["R"]
                    records.append(record)

        return records
    except Exception as e:
        print(f"Erro ao parsear resposta: {e}")
        return None


def fetch_data(query_payload):
    """
    Faz requisição para a API do PowerBI
    """
    try:
        response = requests.post(API_URL, headers=HEADERS, json=query_payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None


def get_all_schools(ano: int = 2024):
    """
    Busca dados de todas as escolas
    """
    print(f"Buscando dados de todas as escolas - Ano {ano}...")
    query = build_query_all_schools(ano=ano)
    response = fetch_data(query)

    if response:
        print("Resposta recebida!")
        print(json.dumps(response, indent=2, ensure_ascii=False)[:2000])
        return response
    return None


def get_school_data(inep_nome: str, ano: int = 2024):
    """
    Busca dados de uma escola específica
    """
    print(f"Buscando dados de: {inep_nome} - Ano {ano}...")
    query = build_query_single_school(inep_nome=inep_nome, ano=ano)
    response = fetch_data(query)

    if response:
        print("Resposta recebida!")
        return response
    return None


if __name__ == "__main__":
    # Teste 1: Buscar escola específica
    print("=" * 60)
    print("TESTE 1: Escola específica")
    print("=" * 60)
    result = get_school_data("21199809-COLEGIO LITERATO", ano=2024)
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])

    print("\n")

    # Teste 2: Tentar buscar todas as escolas
    print("=" * 60)
    print("TESTE 2: Todas as escolas")
    print("=" * 60)
    result = get_all_schools(ano=2024)
