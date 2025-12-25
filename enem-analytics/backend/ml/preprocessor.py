"""
Feature engineering and data preprocessing for ENEM ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os

class ENEMPreprocessor:
    """Preprocessor for ENEM school data - creates features for ML models"""

    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data"
        self.data_path = Path(data_path)

        # Load main datasets
        self.df = None
        self.skills_df = None
        self.school_skills_df = None
        self._load_data()

    def _load_data(self):
        """Load all required datasets"""
        # Main ENEM data
        enem_file = self.data_path / "enem_2018_2024_completo.csv"
        self.df = pd.read_csv(enem_file, dtype={'codigo_inep': str})
        print(f"Loaded {len(self.df)} ENEM records")

        # National skills data
        skills_file = self.data_path / "habilidades_2024.csv"
        if skills_file.exists():
            self.skills_df = pd.read_csv(skills_file)
            print(f"Loaded {len(self.skills_df)} skill records")

        # School-level skills data
        school_skills_file = self.data_path / "desempenho_habilidades_2024.csv"
        if school_skills_file.exists():
            self.school_skills_df = pd.read_csv(school_skills_file)
            print(f"Loaded {len(self.school_skills_df)} school skill records")

    def create_lagged_features(self, school_df: pd.DataFrame, n_lags: int = 3) -> Dict[str, float]:
        """
        Create lagged score features for a school

        Args:
            school_df: DataFrame with school's historical data (sorted by year)
            n_lags: Number of lag years to create

        Returns:
            Dictionary of lagged features
        """
        features = {}
        score_cols = ['nota_cn', 'nota_ch', 'nota_lc', 'nota_mt', 'nota_redacao', 'nota_media', 'ranking_brasil']

        # Sort by year descending (most recent first)
        school_df = school_df.sort_values('ano', ascending=False)

        for col in score_cols:
            for lag in range(1, n_lags + 1):
                if len(school_df) >= lag:
                    features[f'{col}_lag{lag}'] = school_df.iloc[lag - 1][col]
                else:
                    features[f'{col}_lag{lag}'] = np.nan

        return features

    def create_trend_features(self, school_df: pd.DataFrame) -> Dict[str, float]:
        """
        Create trend features (slope, volatility) for a school

        Args:
            school_df: DataFrame with school's historical data

        Returns:
            Dictionary of trend features
        """
        features = {}
        score_cols = ['nota_cn', 'nota_ch', 'nota_lc', 'nota_mt', 'nota_redacao']

        # Sort by year ascending for trend calculation
        school_df = school_df.sort_values('ano')

        for col in score_cols:
            values = school_df[col].dropna().values
            col_short = col.replace('nota_', '')

            if len(values) >= 2:
                # Linear trend (slope)
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                features[f'trend_{col_short}'] = slope

                # Volatility (std dev)
                features[f'volatility_{col_short}'] = np.std(values)

                # Year-over-year change (most recent)
                features[f'yoy_{col_short}'] = values[-1] - values[-2] if len(values) >= 2 else 0
            else:
                features[f'trend_{col_short}'] = 0
                features[f'volatility_{col_short}'] = 0
                features[f'yoy_{col_short}'] = 0

        return features

    def create_school_features(self, school_df: pd.DataFrame) -> Dict[str, float]:
        """
        Create school characteristic features

        Args:
            school_df: DataFrame with school's data

        Returns:
            Dictionary of school features
        """
        features = {}

        # Get most recent record
        latest = school_df.sort_values('ano').iloc[-1]

        # Porte (1-5)
        features['porte'] = latest.get('porte', 3) or 3

        # Localizacao (one-hot)
        loc = latest.get('localizacao', 'Urbana')
        features['localizacao_rural'] = 1 if loc == 'Rural' else 0

        # Tipo escola (one-hot)
        tipo = latest.get('tipo_escola', 'Pública')
        features['tipo_privada'] = 1 if tipo == 'Privada' else 0

        # Years of participation
        features['years_of_data'] = len(school_df)

        # Average historical ranking
        features['avg_historical_ranking'] = school_df['ranking_brasil'].mean()

        # Desempenho habilidades (if available)
        features['desempenho_habilidades'] = latest.get('desempenho_habilidades', np.nan)

        return features

    def create_skill_aggregate_features(self, codigo_inep: str) -> Dict[str, float]:
        """
        Create aggregated skill features for a school

        Args:
            codigo_inep: School INEP code

        Returns:
            Dictionary of skill aggregate features
        """
        features = {}

        # Default values if no skill data
        for area in ['cn', 'ch', 'lc', 'mt']:
            features[f'avg_skill_{area}'] = np.nan
            features[f'worst_skill_{area}'] = np.nan
            features[f'best_skill_{area}'] = np.nan
        features['skill_gap_count'] = np.nan

        # If we have school skills data, we could populate this
        # For now, use the overall desempenho_habilidades as proxy

        return features

    def prepare_features_for_school(self, codigo_inep: str) -> Dict[str, float]:
        """
        Prepare all features for a single school

        Args:
            codigo_inep: School INEP code

        Returns:
            Dictionary with all features
        """
        school_df = self.df[self.df['codigo_inep'] == codigo_inep].copy()

        if len(school_df) == 0:
            return None

        features = {'codigo_inep': codigo_inep}

        # Add all feature types
        features.update(self.create_lagged_features(school_df))
        features.update(self.create_trend_features(school_df))
        features.update(self.create_school_features(school_df))
        features.update(self.create_skill_aggregate_features(codigo_inep))

        return features

    def prepare_training_data(self, target_col: str = 'nota_media', min_years: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for prediction model

        Uses 2023 as target year, trains on schools with at least min_years of data

        Args:
            target_col: Target column to predict
            min_years: Minimum years of data required

        Returns:
            X (features DataFrame), y (target Series)
        """
        # Get schools with enough data
        school_counts = self.df.groupby('codigo_inep').size()
        valid_schools = school_counts[school_counts >= min_years].index

        print(f"Schools with >= {min_years} years: {len(valid_schools)}")

        # For each school, create features using data up to 2022, target is 2023
        X_list = []
        y_list = []

        for codigo_inep in valid_schools:
            school_df = self.df[self.df['codigo_inep'] == codigo_inep]

            # Check if school has 2023 data (our target)
            if 2023 not in school_df['ano'].values:
                continue

            # Get 2023 target
            target_2023 = school_df[school_df['ano'] == 2023][target_col].values[0]
            if pd.isna(target_2023):
                continue

            # Create features from data up to 2022
            train_df = school_df[school_df['ano'] <= 2022]
            if len(train_df) < 2:
                continue

            features = {'codigo_inep': codigo_inep}
            features.update(self.create_lagged_features(train_df))
            features.update(self.create_trend_features(train_df))
            features.update(self.create_school_features(train_df))

            X_list.append(features)
            y_list.append(target_2023)

        X = pd.DataFrame(X_list)
        y = pd.Series(y_list, name=target_col)

        # Drop non-feature columns
        feature_cols = [c for c in X.columns if c != 'codigo_inep']
        X_features = X[feature_cols]

        print(f"Training samples: {len(X)}")
        print(f"Features: {len(feature_cols)}")

        return X_features, y, X['codigo_inep']

    def compute_skill_tri_correlations(self) -> pd.DataFrame:
        """
        Compute correlation between each skill and TRI scores

        Returns:
            DataFrame with skill-TRI correlations
        """
        if self.skills_df is None:
            print("No skills data available")
            return None

        # For now, return skill performance as proxy for impact
        # In a full implementation, we'd join school skills with their TRI scores
        correlations = []

        for _, skill in self.skills_df.iterrows():
            # Higher performance = easier skill = lower impact on differentiation
            # So we invert: low performance = high impact
            impact = 1 - skill['performance']

            correlations.append({
                'area': skill['area'],
                'skill_num': skill['skill_num'],
                'performance': skill['performance'],
                'impact_score': impact,
                'descricao': skill.get('descricao', '')
            })

        return pd.DataFrame(correlations)

    def get_peer_schools(self, codigo_inep: str, year: int = 2024) -> pd.DataFrame:
        """
        Get peer schools (same porte, tipo_escola) for comparison

        Args:
            codigo_inep: Target school INEP code
            year: Year for comparison

        Returns:
            DataFrame of peer schools
        """
        school_data = self.df[(self.df['codigo_inep'] == codigo_inep) & (self.df['ano'] == year)]

        if len(school_data) == 0:
            return pd.DataFrame()

        school = school_data.iloc[0]
        porte = school.get('porte')
        tipo = school.get('tipo_escola')

        # Find peers
        peers = self.df[
            (self.df['ano'] == year) &
            (self.df['porte'] == porte) &
            (self.df['tipo_escola'] == tipo) &
            (self.df['codigo_inep'] != codigo_inep)
        ]

        return peers

    def get_top_peers(self, codigo_inep: str, year: int = 2024, percentile: float = 0.8) -> pd.DataFrame:
        """
        Get top performing peer schools as benchmark

        Args:
            codigo_inep: Target school INEP code
            year: Year for comparison
            percentile: Top percentile to consider (0.8 = top 20%)

        Returns:
            DataFrame of top peer schools
        """
        peers = self.get_peer_schools(codigo_inep, year)

        if len(peers) == 0:
            return pd.DataFrame()

        # Get top percentile
        threshold = peers['nota_media'].quantile(percentile)
        top_peers = peers[peers['nota_media'] >= threshold]

        return top_peers


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = ENEMPreprocessor()

    # Test feature creation for a school
    test_school = preprocessor.df['codigo_inep'].iloc[0]
    features = preprocessor.prepare_features_for_school(test_school)
    print(f"\nFeatures for {test_school}:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    # Test training data preparation
    X, y, school_ids = preprocessor.prepare_training_data('nota_media')
    print(f"\nTraining data shape: X={X.shape}, y={y.shape}")

    # Test correlations
    corr = preprocessor.compute_skill_tri_correlations()
    if corr is not None:
        print(f"\nSkill correlations shape: {corr.shape}")
        print(corr.head())
