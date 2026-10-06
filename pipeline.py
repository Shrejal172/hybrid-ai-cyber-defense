"""
Data Pipeline for Hybrid AI Cyber Defense & Intrusion Detection System.

This module handles data ingestion, preprocessing, feature engineering,
and sequence generation for both the autoencoder and temporal classifier.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import pickle
import logging
from pathlib import Path

import config

# Configure logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Modular data pipeline for network traffic data processing.

    Handles:
    - Data loading and validation
    - Feature scaling
    - Categorical encoding
    - Time-series sliding window aggregation
    - Train/test split
    - UNSW-NB15 processed dataset loading
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data pipeline.

        Args:
            data_path: Path to the raw dataset.
                      If None, uses config.RAW_DATA_PATH.
        """
        self.data_path = data_path or config.RAW_DATA_PATH

        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        self.numeric_features = None
        self.categorical_features = None

    # ============================================================
    # ORIGINAL CSV DATA LOADING
    # ============================================================

    def load_data(self) -> pd.DataFrame:
        """
        Load network traffic data from CSV file.

        Returns:
            DataFrame containing the raw network traffic data.
        """
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            logger.warning(
                f"Data file not found at {self.data_path}. "
                "Generating synthetic data for demonstration."
            )
            return self._generate_synthetic_data()

        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    # ============================================================
    # SYNTHETIC DATA FALLBACK
    # ============================================================

    def _generate_synthetic_data(
        self,
        n_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Generate synthetic network traffic data for demonstration.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Synthetic network traffic DataFrame.
        """

        logger.info(f"Generating {n_samples} synthetic samples")

        np.random.seed(config.RANDOM_STATE)

        data = {
            'src_ip': [
                f"192.168.1.{np.random.randint(1, 255)}"
                for _ in range(n_samples)
            ],

            'dst_ip': [
                f"10.0.0.{np.random.randint(1, 255)}"
                for _ in range(n_samples)
            ],

            'src_port': np.random.randint(
                1024,
                65535,
                n_samples
            ),

            'dst_port': np.random.choice(
                [80, 443, 22, 21, 3306, 5432],
                n_samples
            ),

            'protocol': np.random.choice(
                ['TCP', 'UDP', 'ICMP'],
                n_samples
            ),

            'packet_size': np.random.exponential(
                500,
                n_samples
            ),

            'duration': np.random.exponential(
                1.0,
                n_samples
            ),

            'bytes_sent': np.random.exponential(
                10000,
                n_samples
            ),

            'bytes_received': np.random.exponential(
                15000,
                n_samples
            ),

            'inter_arrival_time': np.random.exponential(
                0.1,
                n_samples
            ),

            'flow_duration': np.random.exponential(
                2.0,
                n_samples
            ),

            'connection_count': np.random.poisson(
                5,
                n_samples
            ),

            'unique_destinations': np.random.poisson(
                3,
                n_samples
            ),

            'label': np.random.choice(
                [0, 1, 2, 3, 4],
                n_samples,
                p=[0.7, 0.1, 0.08, 0.07, 0.05]
            )
        }

        df = pd.DataFrame(data)

        logger.info(
            f"Generated synthetic data with shape: {df.shape}"
        )

        return df

    # ============================================================
    # PREPROCESSING FOR ORIGINAL CSV PIPELINE
    # ============================================================

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'label'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data.

        Handles:
        - Missing values
        - Categorical encoding
        - Numeric scaling

        Args:
            df: Raw DataFrame.
            target_col: Target column name.

        Returns:
            Tuple of feature DataFrame and target Series.
        """

        logger.info("Starting data preprocessing")

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna('Unknown')

        # Separate features and target
        self.target_column = target_col

        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            logger.warning(
                f"Target column '{target_col}' not found. "
                "Using all columns as features."
            )
            y = None
            X = df

        # Identify numeric and categorical columns
        self.numeric_features = (
            X.select_dtypes(include=[np.number])
            .columns
            .tolist()
        )

        self.categorical_features = (
            X.select_dtypes(include=['object'])
            .columns
            .tolist()
        )

        logger.info(
            f"Numeric features: {len(self.numeric_features)}"
        )

        logger.info(
            f"Categorical features: {len(self.categorical_features)}"
        )

        # Encode categorical features
        for col in self.categorical_features:

            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()

            X[col] = self.label_encoders[col].fit_transform(
                X[col].astype(str)
            )

        # Scale numeric features
        if self.numeric_features:
            X[self.numeric_features] = (
                self.scaler.fit_transform(
                    X[self.numeric_features]
                )
            )

        self.feature_columns = X.columns.tolist()

        logger.info(
            f"Preprocessed feature shape: {X.shape}"
        )

        return X, y

    # ============================================================
    # SEQUENCE CREATION
    # ============================================================

    def create_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        window_size: int = config.WINDOW_SIZE,
        stride: int = config.STRIDE
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create time-series sequences using a sliding window.

        The label of the last time step in each window is used
        as the sequence target.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            window_size: Number of time steps.
            stride: Sliding-window stride.

        Returns:
            Tuple containing sequences and optional targets.
        """

        logger.info(
            f"Creating sequences with "
            f"window_size={window_size}, stride={stride}"
        )

        X_values = X.values

        n_samples, n_features = X_values.shape

        if n_samples < window_size:
            raise ValueError(
                f"Not enough samples ({n_samples}) to create "
                f"sequences of window size {window_size}."
            )

        # Calculate number of sequences
        n_sequences = (
            (n_samples - window_size) // stride
        ) + 1

        sequences = np.zeros(
            (
                n_sequences,
                window_size,
                n_features
            ),
            dtype=np.float32
        )

        for i in range(n_sequences):

            start_idx = i * stride
            end_idx = start_idx + window_size

            sequences[i] = X_values[
                start_idx:end_idx
            ]

        logger.info(
            f"Created {n_sequences} sequences "
            f"with shape {sequences.shape}"
        )

        if y is not None:

            # Use the label of the last time step
            # in each window
            y_values = y.values

            targets = y_values[
                window_size - 1::stride
            ][:n_sequences]

            logger.info(
                f"Created targets with shape {targets.shape}"
            )

            return sequences, targets

        return sequences, None

    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================

    def split_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        test_size: float = config.TEST_SIZE,
        random_state: int = config.RANDOM_STATE
    ) -> Tuple:
        """
        Split data into training and testing sets.
        """

        logger.info(
            f"Splitting data with test_size={test_size}"
        )

        if y is not None:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )

            logger.info(
                f"Train shape: {X_train.shape}, "
                f"Test shape: {X_test.shape}"
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test
            )

        else:

            X_train, X_test = train_test_split(
                X,
                test_size=test_size,
                random_state=random_state
            )

            logger.info(
                f"Train shape: {X_train.shape}, "
                f"Test shape: {X_test.shape}"
            )

            return X_train, X_test

    # ============================================================
    # BENIGN DATA
    # ============================================================

    def get_benign_only(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Filter data to keep only benign/normal samples.

        Label 0 represents Normal in the processed UNSW-NB15 data.
        """

        benign_mask = y == 0

        X_benign = X[benign_mask]
        y_benign = y[benign_mask]

        logger.info(
            f"Benign samples: "
            f"{len(X_benign)}/{len(X)}"
        )

        return X_benign, y_benign

    # ============================================================
    # SAVE PIPELINE STATE
    # ============================================================

    def save_pipeline(
        self,
        path: Path = config.PROCESSED_DATA_PATH
    ):
        """
        Save the pipeline state to disk.
        """

        pipeline_state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'target_column': self.target_column
        }

        with open(path, 'wb') as f:
            pickle.dump(
                pipeline_state,
                f
            )

        logger.info(
            f"Pipeline state saved to {path}"
        )

    # ============================================================
    # LOAD PIPELINE STATE
    # ============================================================

    def load_pipeline(
        self,
        path: Path = config.PROCESSED_DATA_PATH
    ) -> Dict:
        """
        Load the pipeline state from disk.
        """

        with open(path, 'rb') as f:
            pipeline_state = pickle.load(f)

        self.scaler = pipeline_state['scaler']
        self.label_encoders = pipeline_state['label_encoders']
        self.feature_columns = pipeline_state['feature_columns']
        self.numeric_features = pipeline_state['numeric_features']
        self.categorical_features = pipeline_state['categorical_features']
        self.target_column = pipeline_state['target_column']

        logger.info(
            f"Pipeline state loaded from {path}"
        )

        return pipeline_state

    # ============================================================
    # FULL PIPELINE
    # ============================================================

    def prepare_full_pipeline(
        self,
        create_sequences: bool = True
    ) -> Dict:
        """
        Execute the full data pipeline.

        If the processed UNSW-NB15 dataset exists, it is loaded
        directly without reprocessing or re-splitting.

        Otherwise, the original CSV pipeline is used.
        """

        # ========================================================
        # PROCESSED UNSW-NB15 DATASET
        # ========================================================

        unsw_path = (
            config.PROJECT_ROOT
            / "data_unsw"
            / "processed_unsw_nb15.pkl"
        )

        if unsw_path.exists():

            logger.info(
                f"Loading processed UNSW-NB15 data from "
                f"{unsw_path}"
            )

            with open(unsw_path, "rb") as f:
                unsw_data = pickle.load(f)

            # ----------------------------------------------------
            # Load train/test features
            # ----------------------------------------------------

            X_train = pd.DataFrame(
                unsw_data["X_train"],
                columns=unsw_data["feature_columns"]
            )

            X_test = pd.DataFrame(
                unsw_data["X_test"],
                columns=unsw_data["feature_columns"]
            )

            # ----------------------------------------------------
            # Load labels
            # ----------------------------------------------------

            y_train = pd.Series(
                unsw_data["y_train"],
                name=unsw_data["target_column"]
            )

            y_test = pd.Series(
                unsw_data["y_test"],
                name=unsw_data["target_column"]
            )

            # ----------------------------------------------------
            # Update configuration based on actual dataset
            # ----------------------------------------------------

            config.AUTOENCODER_CONFIG["input_dim"] = (
                X_train.shape[1]
            )

            config.TEMPORAL_CONFIG["input_dim"] = (
                X_train.shape[1]
            )

            config.TEMPORAL_CONFIG["num_classes"] = (
                unsw_data["num_classes"]
            )

            # ----------------------------------------------------
            # Restore pipeline metadata
            # ----------------------------------------------------

            self.feature_columns = (
                unsw_data["feature_columns"]
            )

            self.target_column = (
                unsw_data["target_column"]
            )

            self.scaler = unsw_data["scaler"]

            # ----------------------------------------------------
            # Get normal/benign training samples
            # ----------------------------------------------------

            X_train_benign, y_train_benign = (
                self.get_benign_only(
                    X_train,
                    y_train
                )
            )

            # ----------------------------------------------------
            # Basic result dictionary
            # ----------------------------------------------------

            result = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "X_train_benign": X_train_benign,
                "y_train_benign": y_train_benign,
            }

            # ----------------------------------------------------
            # Create sequences
            # ----------------------------------------------------

            if create_sequences:

                # Temporal classifier sequences
                X_train_seq, y_train_seq = (
                    self.create_sequences(
                        X_train,
                        y_train
                    )
                )

                X_test_seq, y_test_seq = (
                    self.create_sequences(
                        X_test,
                        y_test
                    )
                )

                # Autoencoder sequences
                X_train_benign_seq, _ = (
                    self.create_sequences(
                        X_train_benign,
                        None
                    )
                )

                X_test_seq_ae, _ = (
                    self.create_sequences(
                        X_test,
                        None
                    )
                )

                result.update({
                    "X_train_seq": X_train_seq,
                    "X_test_seq": X_test_seq,
                    "y_train_seq": y_train_seq,
                    "y_test_seq": y_test_seq,
                    "X_train_benign_seq": X_train_benign_seq,
                    "X_test_seq_ae": X_test_seq_ae,
                })

            logger.info(
                "Processed UNSW-NB15 pipeline "
                "completed successfully"
            )

            return result

        # ========================================================
        # ORIGINAL CSV PIPELINE - FALLBACK
        # ========================================================

        logger.info(
            "Processed UNSW-NB15 file not found. "
            "Using original CSV pipeline."
        )

        df = self.load_data()

        X, y = self.preprocess_data(df)

        # Update configuration
        config.AUTOENCODER_CONFIG["input_dim"] = (
            X.shape[1]
        )

        config.TEMPORAL_CONFIG["input_dim"] = (
            X.shape[1]
        )

        # Split data
        X_train, X_test, y_train, y_test = (
            self.split_data(X, y)
        )

        # Get benign data
        X_train_benign, y_train_benign = (
            self.get_benign_only(
                X_train,
                y_train
            )
        )

        result = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_benign": X_train_benign,
            "y_train_benign": y_train_benign,
        }

        # Create sequences
        if create_sequences:

            X_train_seq, y_train_seq = (
                self.create_sequences(
                    X_train,
                    y_train
                )
            )

            X_test_seq, y_test_seq = (
                self.create_sequences(
                    X_test,
                    y_test
                )
            )

            X_train_benign_seq, _ = (
                self.create_sequences(
                    X_train_benign,
                    None
                )
            )

            X_test_seq_ae, _ = (
                self.create_sequences(
                    X_test,
                    None
                )
            )

            result.update({
                "X_train_seq": X_train_seq,
                "X_test_seq": X_test_seq,
                "y_train_seq": y_train_seq,
                "y_test_seq": y_test_seq,
                "X_train_benign_seq": X_train_benign_seq,
                "X_test_seq_ae": X_test_seq_ae,
            })

        # Save pipeline state for CSV pipeline
        self.save_pipeline()

        logger.info(
            "Full data pipeline completed successfully"
        )

        return result


# ================================================================
# MAIN
# ================================================================

def main():
    """
    Main function to test the data pipeline.
    """

    pipeline = DataPipeline()

    data = pipeline.prepare_full_pipeline(
        create_sequences=True
    )

    print("\n=== Data Pipeline Summary ===")

    print(
        f"Train samples: "
        f"{data['X_train'].shape[0]}"
    )

    print(
        f"Test samples: "
        f"{data['X_test'].shape[0]}"
    )

    print(
        f"Benign train samples: "
        f"{data['X_train_benign'].shape[0]}"
    )

    print(
        f"Feature dimensions: "
        f"{data['X_train'].shape[1]}"
    )

    print(
        f"Sequence shape (train): "
        f"{data['X_train_seq'].shape}"
    )

    print(
        f"Sequence shape (test): "
        f"{data['X_test_seq'].shape}"
    )

    print(
        f"Class distribution (train): "
        f"{np.bincount(data['y_train'].values)}"
    )

    print(
        f"Class distribution (test): "
        f"{np.bincount(data['y_test'].values)}"
    )


if __name__ == "__main__":
    main()