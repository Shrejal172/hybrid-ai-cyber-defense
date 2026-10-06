import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

OUTPUT_DIR = PROJECT_ROOT / "data_unsw"
OUTPUT_PATH = OUTPUT_DIR / "processed_unsw_nb15.pkl"

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sinpkt",
    "dinpkt",
    "ct_srv_src",
]

TARGET_COLUMN = "label"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    print(f"\nLoading: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print(f"Shape: {df.shape}")

    return df


# ---------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    print(f"\nValidating {name} data...")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

    # Check target
    labels = set(df[TARGET_COLUMN].dropna().unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{name} contains unexpected labels: {labels}"
        )

    # Check numeric features
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Feature '{column}' is not numeric."
            )

    print("Validation passed.")


# ---------------------------------------------------------
# Clean features
# ---------------------------------------------------------

def clean_features(df: pd.DataFrame, name: str) -> np.ndarray:
    X = df[FEATURE_COLUMNS].copy()

    # Convert infinite values to NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    missing_before = int(X.isna().sum().sum())

    if missing_before:
        print(
            f"{name}: found {missing_before} missing/infinite "
            "feature values."
        )

        raise ValueError(
            f"{name} contains missing/infinite values. "
            "We will handle this explicitly rather than silently "
            "using test-set information."
        )

    return X.to_numpy(dtype=np.float32)


# ---------------------------------------------------------
# Main preparation
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("UNSW-NB15 DATA PREPARATION")
    print("=" * 70)

    # Load
    train_df = load_csv(TRAIN_PATH)
    test_df = load_csv(TEST_PATH)

    # Validate
    validate_dataframe(train_df, "Training")
    validate_dataframe(test_df, "Testing")

    # Extract features
    X_train_raw = clean_features(train_df, "Training")
    X_test_raw = clean_features(test_df, "Testing")

    # Targets
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    print("\n" + "=" * 70)
    print("PREPARED DATA")
    print("=" * 70)

    print(f"Training feature shape : {X_train.shape}")
    print(f"Testing feature shape  : {X_test.shape}")

    print("\nTraining labels:")
    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTesting labels:")
    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFeature range after scaling:")
    print(f"Training minimum: {X_train.min():.6f}")
    print(f"Training maximum: {X_train.max():.6f}")
    print(f"Testing minimum : {X_test.min():.6f}")
    print(f"Testing maximum : {X_test.max():.6f}")

    # -----------------------------------------------------
    # Save separate processed dataset
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "num_classes": 2,
        "class_names": {
            0: "Normal",
            1: "Attack",
        },
        "dataset_name": "UNSW-NB15",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(processed_data, f)

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print(f"Saved processed dataset to:")
    print(OUTPUT_PATH)

    print("\nOriginal baseline files were NOT modified:")
    print(PROJECT_ROOT / "data" / "processed_data.pkl")
    print(PROJECT_ROOT / "checkpoints" / "autoencoder.pth")
    print(PROJECT_ROOT / "checkpoints" / "temporal_classifier.pth")


if __name__ == "__main__":
    main()