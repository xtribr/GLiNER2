"""
Extract school porte and location data from Censo Escolar microdata
"""

import pandas as pd
import os

# Column mappings
COLUMNS_TO_EXTRACT = [
    'CO_ENTIDADE',      # INEP code
    'NO_ENTIDADE',      # School name
    'SG_UF',            # State
    'TP_DEPENDENCIA',   # Admin dependency (1=Federal, 2=Estadual, 3=Municipal, 4=Privada)
    'TP_LOCALIZACAO',   # Location (1=Urbana, 2=Rural)
    'QT_MAT_MED',       # Total high school enrollments
    'QT_MAT_MED_PROP_3',  # 3rd year regular high school (concluintes)
    'QT_MAT_MED_CT_3',    # 3rd year technical course (concluintes)
    'QT_MAT_MED_NM_3',    # 3rd year new high school (concluintes)
    'QT_MAT_MED_PROP_4',  # 4th year regular (some courses)
    'QT_MAT_MED_CT_4',    # 4th year technical (some courses)
    'TP_SITUACAO_FUNCIONAMENTO',  # Status (1=Em atividade, 2=Paralisada, 3=Extinta)
]

# Porte categories based on concluintes (3rd year students)
def calculate_porte(qt_concluintes):
    """
    Calculate school porte based on concluintes (3rd year students)
    Categories adjusted for ENEM participants:
    - 1: Até 30 concluintes (Muito pequena)
    - 2: 31 a 100 concluintes (Pequena)
    - 3: 101 a 200 concluintes (Média)
    - 4: 201 a 400 concluintes (Grande)
    - 5: Mais de 400 concluintes (Muito grande)
    """
    if pd.isna(qt_concluintes) or qt_concluintes <= 0:
        return None
    elif qt_concluintes <= 30:
        return 1
    elif qt_concluintes <= 100:
        return 2
    elif qt_concluintes <= 200:
        return 3
    elif qt_concluintes <= 400:
        return 4
    else:
        return 5

PORTE_LABELS = {
    1: "Muito pequena (até 30)",
    2: "Pequena (31-100)",
    3: "Média (101-200)",
    4: "Grande (201-400)",
    5: "Muito grande (400+)"
}

LOCALIZACAO_LABELS = {
    1: "Urbana",
    2: "Rural"
}

DEPENDENCIA_LABELS = {
    1: "Federal",
    2: "Estadual",
    3: "Municipal",
    4: "Privada"
}


def extract_censo_data():
    """Extract relevant columns from Censo Escolar"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    censo_file = os.path.join(data_dir, "microdados_censo_escolar_2024/dados/microdados_ed_basica_2024.csv")

    print(f"Reading Censo Escolar data from: {censo_file}")
    print("This may take a moment...")

    # Read only needed columns with specific encoding
    try:
        df = pd.read_csv(
            censo_file,
            sep=';',
            encoding='latin-1',
            usecols=COLUMNS_TO_EXTRACT,
            dtype={
                'CO_ENTIDADE': str,
                'TP_DEPENDENCIA': 'Int64',
                'TP_LOCALIZACAO': 'Int64',
                'QT_MAT_MED': 'Int64',
                'QT_MAT_MED_PROP_3': 'Int64',
                'QT_MAT_MED_CT_3': 'Int64',
                'QT_MAT_MED_NM_3': 'Int64',
                'QT_MAT_MED_PROP_4': 'Int64',
                'QT_MAT_MED_CT_4': 'Int64',
                'TP_SITUACAO_FUNCIONAMENTO': 'Int64',
            }
        )
    except Exception as e:
        print(f"Error reading with latin-1, trying utf-8: {e}")
        df = pd.read_csv(
            censo_file,
            sep=';',
            encoding='utf-8',
            usecols=COLUMNS_TO_EXTRACT,
            dtype={'CO_ENTIDADE': str}
        )

    print(f"Loaded {len(df)} schools")

    # Filter only active schools
    df = df[df['TP_SITUACAO_FUNCIONAMENTO'] == 1].copy()
    print(f"Active schools: {len(df)}")

    # Calculate concluintes (3rd + 4th year students - these are ENEM eligible)
    df['qt_concluintes'] = (
        df['QT_MAT_MED_PROP_3'].fillna(0) +
        df['QT_MAT_MED_CT_3'].fillna(0) +
        df['QT_MAT_MED_NM_3'].fillna(0) +
        df['QT_MAT_MED_PROP_4'].fillna(0) +
        df['QT_MAT_MED_CT_4'].fillna(0)
    ).astype('Int64')

    print(f"Schools with concluintes: {(df['qt_concluintes'] > 0).sum()}")

    # Calculate porte based on concluintes
    df['porte'] = df['qt_concluintes'].apply(calculate_porte)
    df['porte_label'] = df['porte'].map(PORTE_LABELS)

    # Map location
    df['localizacao'] = df['TP_LOCALIZACAO'].map(LOCALIZACAO_LABELS)

    # Map dependency
    df['dependencia'] = df['TP_DEPENDENCIA'].map(DEPENDENCIA_LABELS)

    # Rename columns for clarity
    df = df.rename(columns={
        'CO_ENTIDADE': 'codigo_inep',
        'NO_ENTIDADE': 'nome_escola_censo',
        'SG_UF': 'uf',
        'QT_MAT_MED': 'qt_matriculas_medio',
        'qt_concluintes': 'qt_matriculas'  # rename to qt_matriculas for consistency
    })

    # Select final columns
    output_df = df[[
        'codigo_inep',
        'nome_escola_censo',
        'uf',
        'localizacao',
        'dependencia',
        'qt_matriculas',  # now contains concluintes count
        'qt_matriculas_medio',
        'porte',
        'porte_label'
    ]].copy()

    # Save to CSV
    output_file = os.path.join(data_dir, "censo_escolas_2024.csv")
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(output_df)} schools to {output_file}")

    # Show distribution
    print("\n" + "="*60)
    print("PORTE DISTRIBUTION")
    print("="*60)
    print(output_df['porte_label'].value_counts().sort_index())

    print("\n" + "="*60)
    print("LOCALIZACAO DISTRIBUTION")
    print("="*60)
    print(output_df['localizacao'].value_counts())

    return output_df


def merge_with_enem():
    """Merge Censo data with ENEM data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    # Load ENEM data
    enem_file = os.path.join(data_dir, "enem_2018_2024_completo.csv")
    print(f"\nLoading ENEM data from: {enem_file}")
    enem_df = pd.read_csv(enem_file, dtype={'codigo_inep': str})
    print(f"ENEM records: {len(enem_df)}")

    # Drop old porte columns if they exist
    cols_to_drop = ['localizacao', 'qt_matriculas', 'porte', 'porte_label']
    for col in cols_to_drop:
        if col in enem_df.columns:
            enem_df = enem_df.drop(columns=[col])
            print(f"Dropped old column: {col}")

    # Load Censo data
    censo_file = os.path.join(data_dir, "censo_escolas_2024.csv")
    print(f"Loading Censo data from: {censo_file}")
    censo_df = pd.read_csv(censo_file, dtype={'codigo_inep': str})
    print(f"Censo schools: {len(censo_df)}")

    # Select only needed columns from Censo
    censo_merge = censo_df[['codigo_inep', 'localizacao', 'qt_matriculas', 'porte', 'porte_label']].copy()

    # Merge
    merged_df = enem_df.merge(censo_merge, on='codigo_inep', how='left')

    # Show merge stats
    matched = merged_df['porte'].notna().sum()
    total = len(merged_df)
    print(f"\nMatched: {matched}/{total} ({100*matched/total:.1f}%)")

    # Show porte distribution in ENEM schools
    print("\n" + "="*60)
    print("PORTE DISTRIBUTION IN ENEM SCHOOLS (2024)")
    print("="*60)
    enem_2024 = merged_df[merged_df['ano'] == 2024]
    print(enem_2024['porte_label'].value_counts().sort_index())

    print("\n" + "="*60)
    print("LOCALIZACAO DISTRIBUTION IN ENEM SCHOOLS (2024)")
    print("="*60)
    print(enem_2024['localizacao'].value_counts())

    # Save merged data
    output_file = os.path.join(data_dir, "enem_2018_2024_completo.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"\nUpdated ENEM data saved to: {output_file}")

    return merged_df


if __name__ == "__main__":
    print("="*60)
    print("EXTRACTING CENSO ESCOLAR DATA")
    print("="*60)

    censo_df = extract_censo_data()

    print("\n" + "="*60)
    print("MERGING WITH ENEM DATA")
    print("="*60)

    merged_df = merge_with_enem()

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
