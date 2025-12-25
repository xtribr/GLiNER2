"""
Discover available columns in PowerBI entities
"""

import requests
import json

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

def build_query_for_columns(entity: str, columns: list) -> dict:
    """Build query to try fetching specific columns from an entity"""
    selects = []
    for i, col in enumerate(columns):
        selects.append({
            "Column": {
                "Expression": {"SourceRef": {"Source": "s"}},
                "Property": col
            },
            "Name": f"col_{i}"
        })

    return {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "s", "Entity": entity, "Type": 0}
                            ],
                            "Select": selects
                        },
                        "Binding": {
                            "Primary": {"Groupings": [{"Projections": list(range(len(columns)))}]},
                            "DataReduction": {"DataVolume": 3, "Primary": {"Window": {"Count": 10}}},
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

def test_columns(entity: str, columns: list):
    """Test if columns exist in an entity"""
    print(f"\nTesting columns in {entity}:")
    print(f"  Columns: {columns}")

    query = build_query_for_columns(entity, columns)

    try:
        response = requests.post(API_URL, headers=HEADERS, json=query, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for errors
        result = data.get("results", [{}])[0].get("result", {})
        if "data" in result:
            dsr = result["data"].get("dsr", {})
            if "DataShapes" in dsr:
                shapes = dsr["DataShapes"]
                if shapes and "odata.error" in shapes[0]:
                    error_msg = shapes[0]["odata.error"]["message"]["value"]
                    print(f"  ERROR: {error_msg}")
                    return None

            ds = dsr.get("DS", [{}])[0]
            if "PH" in ds:
                ph = ds["PH"]
                if ph:
                    dm0 = ph[0].get("DM0", [])
                    if dm0:
                        print(f"  SUCCESS! Found {len(dm0)} rows")
                        # Show sample
                        for row in dm0[:3]:
                            print(f"    Row: {row.get('C', [])}")
                        return dm0

        print(f"  No data found")
        return None

    except Exception as e:
        print(f"  Request error: {e}")
        return None


# Test various column names that might exist
POTENTIAL_COLUMNS = {
    "stg-senso-escolar": [
        # Known to work
        ["co_entidade", "nome_escola", "uf", "ano"],
        # Location related
        ["localizacao"],
        ["tp_localizacao"],
        ["tipo_localizacao"],
        ["zona"],
        ["tp_zona"],
        # Size/student count
        ["qty_participantes"],
        ["qt_participantes"],
        ["quantidade_participantes"],
        ["qt_alunos"],
        ["qty_alunos"],
        ["num_alunos"],
        ["total_participantes"],
        # Network/dependency
        ["tp_dependencia"],
        ["rede"],
        ["tipo_rede"],
        ["dependencia"],
        # Other
        ["municipio"],
        ["cidade"],
        ["co_municipio"],
        ["regiao"],
    ],
    "enem_area": [
        ["area"],
    ],
    "filtro_tamanho": [
        ["nome"],
        ["tamanho"],
    ],
    "comparação": [
        ["Coluna 1"],
    ],
}


def test_porte_entity():
    """Test the filtro_tamanho/porte entity to see what values are available"""
    print("\n" + "="*60)
    print("Testing filtro_tamanho entity values...")

    query = {
        "version": "1.0.0",
        "queries": [{
            "Query": {
                "Commands": [{
                    "SemanticQueryDataShapeCommand": {
                        "Query": {
                            "Version": 2,
                            "From": [
                                {"Name": "f", "Entity": "filtro_tamanho", "Type": 0}
                            ],
                            "Select": [
                                {
                                    "Column": {
                                        "Expression": {"SourceRef": {"Source": "f"}},
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

        # Parse the response to extract values
        ds = data["results"][0]["result"]["data"]["dsr"]["DS"][0]
        value_dicts = ds.get("ValueDicts", {})

        print("ValueDicts:", json.dumps(value_dicts, indent=2))

        values = []
        for ph in ds.get("PH", []):
            for row in ph.get("DM0", []):
                if "C" in row:
                    val = row["C"][0]
                    if isinstance(val, int) and "D0" in value_dicts:
                        val = value_dicts["D0"][val]
                    values.append(val)

        print(f"\nfiltro_tamanho values: {values}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_porte_columns():
    """Test porte-related columns in stg-senso-escolar"""
    print("\n" + "="*60)
    print("Testing porte columns in stg-senso-escolar...")

    for col in ["porte", "porte_escola", "faixa_porte", "tamanho", "faixa_tamanho",
                "num_participantes", "participantes", "porte_inep"]:
        test_columns("stg-senso-escolar", [col])


if __name__ == "__main__":
    # Test filtro_tamanho entity
    test_porte_entity()

    # Test porte columns
    test_porte_columns()

    # First, test known columns
    print("\n" + "="*60)
    print("Testing known columns...")
    test_columns("stg-senso-escolar", ["co_entidade", "nome_escola", "uf", "ano", "tipo_escola"])

    print("\n" + "="*60)
    print("Testing potential new columns one by one...")

    working_cols = []

    # Test each potential column individually
    for col in [
        "localizacao", "tp_localizacao", "tipo_localizacao", "zona", "tp_zona",
        "qty_participantes", "qt_participantes", "quantidade_participantes",
        "qt_alunos", "qty_alunos", "num_alunos", "total_participantes",
        "tp_dependencia", "rede", "tipo_rede", "dependencia",
        "municipio", "cidade", "co_municipio", "regiao",
        "tp_dependencia_adm", "localizacao_diferenciada", "rural",
        "tp_situacao_funcionamento", "porte", "faixa_participantes"
    ]:
        result = test_columns("stg-senso-escolar", [col])
        if result:
            working_cols.append(col)

    print("\n" + "="*60)
    print("SUMMARY - Working columns:")
    for col in working_cols:
        print(f"  - {col}")

    # Test other entities that might have the data
    print("\n" + "="*60)
    print("Testing other entities...")

    entities_to_test = [
        "cur-enem-school-score",
        "cur-enem-school-skill",
        "enem_escola",
        "escola",
        "enem_participantes",
        "participantes",
        "localizacao",
        "porte_escola",
        "tamanho_escola",
        "censo_escolar",
        "stg-censo",
        "dim-escola",
        "dim_escola",
        "fato_escola",
        "fat-escola",
    ]

    for entity in entities_to_test:
        # Test with a common column
        result = test_columns(entity, ["ano"])
        if result:
            print(f"  => FOUND ENTITY: {entity}")

    # Also try cur-enem-school-score with more columns
    print("\n" + "="*60)
    print("Testing cur-enem-school-score columns...")
    for col in ["co_entidade", "nota_media", "ranking", "qty_participantes",
                "qt_participantes", "localizacao", "tp_localizacao", "porte"]:
        test_columns("cur-enem-school-score", [col])
