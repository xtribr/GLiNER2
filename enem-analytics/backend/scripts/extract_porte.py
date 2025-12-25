"""
Extract school porte (size) data from PowerBI
"""

import requests
import json
import pandas as pd

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


def query_with_porte_filter(escola_inep: str):
    """Query school ranking with 'Mesmo porte' filter to find schools of same size"""
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
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c", "Entity": "comparação", "Type": 0}
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
                                        "Property": "tipo_escola"
                                    },
                                    "Name": "tipo"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "ano"}}],
                                            "Values": [[{"Literal": {"Value": "2024L"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": "nome"}}],
                                            "Values": [[{"Literal": {"Value": "'Mesmo porte'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "c"}}, "Property": "Coluna 1"}}],
                                            "Values": [[{"Literal": {"Value": "'Brasil'"}}]]
                                        }
                                    }
                                },
                                {
                                    "Condition": {
                                        "StartsWith": {
                                            "Left": {"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "inep_nome"}},
                                            "Right": {"Literal": {"Value": f"'{escola_inep}'"}}
                                        }
                                    }
                                }
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0, 1]}]},
                            "DataReduction": {"DataVolume": 4, "Primary": {"Window": {"Count": 100}}},
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

    response = requests.post(API_URL, headers=HEADERS, json=query, timeout=30)
    response.raise_for_status()
    return response.json()


def query_porte_entity():
    """Try to find porte-related entities"""
    # Try to query porte directly as an entity
    entities_to_try = [
        "porte",
        "porte_escola",
        "tamanho",
        "faixa",
        "faixa_porte",
        "faixa_tamanho",
        "dim_porte",
        "stg-porte"
    ]

    for entity in entities_to_try:
        print(f"\nTrying entity: {entity}")
        query = {
            "version": "1.0.0",
            "queries": [{
                "Query": {
                    "Commands": [{
                        "SemanticQueryDataShapeCommand": {
                            "Query": {
                                "Version": 2,
                                "From": [
                                    {"Name": "e", "Entity": entity, "Type": 0}
                                ],
                                "Select": [
                                    {
                                        "Column": {
                                            "Expression": {"SourceRef": {"Source": "e"}},
                                            "Property": "nome"
                                        },
                                        "Name": "nome"
                                    }
                                ]
                            },
                            "Binding": {
                                "Primary": {"Groupings": [{"Projections": [0]}]},
                                "DataReduction": {"DataVolume": 3, "Primary": {"Window": {"Count": 100}}},
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

        try:
            response = requests.post(API_URL, headers=HEADERS, json=query, timeout=30)
            response.raise_for_status()
            data = response.json()

            result = data.get("results", [{}])[0].get("result", {})
            if "data" in result:
                dsr = result["data"].get("dsr", {})
                if "DataShapes" in dsr:
                    shapes = dsr["DataShapes"]
                    if shapes and "odata.error" in shapes[0]:
                        print(f"  Error: {shapes[0]['odata.error']['message']['value'][:100]}")
                        continue

                ds = dsr.get("DS", [{}])[0]
                if ds:
                    print(f"  SUCCESS! Found entity: {entity}")
                    print(f"  Data: {json.dumps(ds, indent=2)[:500]}")

        except Exception as e:
            print(f"  Error: {e}")


def query_all_school_rankings_by_porte():
    """
    Query school rankings grouped by porte.
    Try to understand how porte filtering works.
    """
    # Query with filtro_tamanho = "Mesmo porte" applied for a specific school
    # This should show us schools of the same porte

    print("\n" + "="*60)
    print("Querying schools with 'Mesmo porte' filter for school 23246847...")

    # First get data for a specific school without porte filter
    query_todas = {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "s", "Entity": "stg-senso-escolar", "Type": 0},
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0},
                                {"Name": "c", "Entity": "comparação", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "s"}},
                                        "Property": "inep_nome"
                                    },
                                    "Name": "escola"
                                }
                            ],
                            "Where": [
                                {
                                    "Condition": {
                                        "In": {
                                            "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "ano"}}],
                                            "Values": [[{"Literal": {"Value": "2024L"}}]]
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
                                }
                            ]
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": [0]}]},
                            "DataReduction": {"DataVolume": 4, "Primary": {"Window": {"Count": 10}}},
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

    try:
        response = requests.post(API_URL, headers=HEADERS, json=query_todas, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("Response with 'Todas escolas':")
        print(json.dumps(data, indent=2)[:1500])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Try to find porte entity
    query_porte_entity()

    # Query with porte filter
    query_all_school_rankings_by_porte()
