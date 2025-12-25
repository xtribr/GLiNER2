"""
Extrator de Dados Adicionais - ENEM
Extrai desempenho de habilidades e competências de redação por escola
"""

import requests
import json
import pandas as pd
from typing import Optional, List, Dict
import time
import os

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


def build_skill_query(ano: int, uf_code: str) -> dict:
    """Query para desempenho de habilidades por escola"""
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
                                {"Name": "c", "Entity": "cur-enem-school-skill", "Type": 0}
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


def build_redacao_query(ano: int, uf_code: str) -> dict:
    """Query para competências de redação por escola"""
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
                                {"Name": "c", "Entity": "cur-enem-school-red-competence", "Type": 0}
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
                                                "Property": "nota_value"
                                            }
                                        },
                                        "Function": 0  # Average
                                    },
                                    "Name": "nota_competencia"
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


def parse_response(response: dict) -> List[Dict]:
    """Parse PowerBI response to extract data"""
    records = []
    try:
        ds = response["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        for ph in ds.get("PH", []):
            for row in ph.get("DM0", []):
                if "C" in row:
                    values = row["C"]
                    escola = values[0]

                    # Handle ValueDicts lookup
                    if isinstance(escola, int) and "D0" in value_dicts:
                        escola = value_dicts["D0"][escola]

                    valor = values[1] if len(values) > 1 else None

                    if escola and valor is not None:
                        records.append({
                            "inep_nome": escola,
                            "valor": valor
                        })
    except (KeyError, IndexError) as e:
        pass

    return records


def extract_skill_data(anos: List[int]) -> pd.DataFrame:
    """Extrai desempenho de habilidades para todos os anos e estados"""
    all_records = []

    for ano in anos:
        print(f"\n=== Extracting skill data for {ano} ===")

        for uf in UF_CODES:
            print(f"  UF {uf}...", end=" ")
            query = build_skill_query(ano, uf)
            response = api_request(query)

            if response:
                records = parse_response(response)
                for r in records:
                    r["ano"] = ano
                all_records.extend(records)
                print(f"{len(records)} schools")
            else:
                print("failed")

            time.sleep(0.3)

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.rename(columns={"valor": "desempenho_habilidades"})
    return df


def extract_redacao_data(anos: List[int]) -> pd.DataFrame:
    """Extrai competências de redação para todos os anos e estados"""
    all_records = []

    for ano in anos:
        print(f"\n=== Extracting redação data for {ano} ===")

        for uf in UF_CODES:
            print(f"  UF {uf}...", end=" ")
            query = build_redacao_query(ano, uf)
            response = api_request(query)

            if response:
                records = parse_response(response)
                for r in records:
                    r["ano"] = ano
                all_records.extend(records)
                print(f"{len(records)} schools")
            else:
                print("failed")

            time.sleep(0.3)

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.rename(columns={"valor": "competencia_redacao_media"})
    return df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    anos = [2024]  # Start with just 2024 for testing

    # Extract skill data
    print("\n" + "="*50)
    print("EXTRACTING SKILL PERFORMANCE DATA")
    print("="*50)

    skill_df = extract_skill_data(anos)
    skill_path = os.path.join(data_dir, "desempenho_habilidades_2024.csv")
    skill_df.to_csv(skill_path, index=False)
    print(f"\nSaved {len(skill_df)} skill records to {skill_path}")

    # Extract redação data
    print("\n" + "="*50)
    print("EXTRACTING REDAÇÃO COMPETENCE DATA")
    print("="*50)

    redacao_df = extract_redacao_data(anos)
    redacao_path = os.path.join(data_dir, "competencia_redacao_2024.csv")
    redacao_df.to_csv(redacao_path, index=False)
    print(f"\nSaved {len(redacao_df)} redação records to {redacao_path}")

    # Merge with main dataset
    print("\n" + "="*50)
    print("MERGING WITH MAIN DATASET")
    print("="*50)

    main_csv = os.path.join(data_dir, "enem_2018_2024_completo.csv")
    main_df = pd.read_csv(main_csv)

    print(f"Main dataset: {len(main_df)} records")

    # Merge skill data
    if not skill_df.empty:
        skill_df_merge = skill_df[["inep_nome", "ano", "desempenho_habilidades"]]
        main_df = main_df.merge(skill_df_merge, on=["inep_nome", "ano"], how="left")
        print(f"Skills merged: {main_df['desempenho_habilidades'].notna().sum()} schools with data")

    # Merge redação data
    if not redacao_df.empty:
        redacao_df_merge = redacao_df[["inep_nome", "ano", "competencia_redacao_media"]]
        main_df = main_df.merge(redacao_df_merge, on=["inep_nome", "ano"], how="left")
        print(f"Redação merged: {main_df['competencia_redacao_media'].notna().sum()} schools with data")

    # Save updated main dataset
    main_df.to_csv(main_csv, index=False)
    print(f"\nUpdated main dataset: {main_csv}")
    print(f"Columns: {list(main_df.columns)}")


if __name__ == "__main__":
    main()
