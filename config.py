"""

This module contains all hyperparameters, paths, and configuration settings
for the autoencoder anomaly detector, temporal classifier, and federated learning.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "network_traffic.csv"  # Replace with actual dataset path
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.pkl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_unsw"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Data configuration
DATASET_NAME = "UNSW-NB15"  # or "CICIDS2017"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature engineering
WINDOW_SIZE = 10  # Number of time steps for sequence models
STRIDE = 1  # Step size for sliding window

# Autoencoder configuration
AUTOENCODER_CONFIG = {
    "input_dim": 13,  # Will be updated based on actual features
    "hidden_dims": [64, 32, 16, 8],  # Encoder layers
    "latent_dim": 4,  # Bottleneck layer
    "activation": "relu",
    "dropout": 0.2,
    "batch_norm": True,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 50,
    "early_stopping_patience": 10,
    "reconstruction_loss": "mse",
    "anomaly_threshold_percentile": 95,  # Percentile for dynamic thresholding
}

# Temporal Classifier (LSTM/Transformer) configuration
TEMPORAL_CONFIG = {
    "input_dim": 13,  # Will be updated based on actual features
    "hidden_dim": 128,
    "num_layers": 2,
    "num_classes": 2,  # Benign, DoS, Reconnaissance, Exploits, Exfiltration
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 30,
    "early_stopping_patience": 8,
    "model_type": "lstm",  # "lstm" or "transformer"
    "bidirectional": True,
}

# Transformer-specific configuration
TRANSFORMER_CONFIG = {
    "num_heads": 8,
    "dim_feedforward": 256,
    "num_transformer_layers": 2,
}

# Federated Learning configuration
FEDERATED_CONFIG = {
    "num_clients": 2,
    "num_rounds": 5,
    "client_epochs": 5,
    "batch_size": 128,
    "learning_rate": 0.01,
    "local_data_fraction": 0.5,  # Fraction of data per client
    "client_selection": "random",  # "random" or "cyclic"
    "aggregation_method": "weighted_average",  # "weighted_average" or "simple_average"
    "encryption_simulation": True,  # Simulate gradient encryption
}

# Evaluation metrics
METRICS = ["precision", "recall", "f1_score", "roc_auc", "accuracy", "confusion_matrix"]

# Attack labels mapping (customize based on dataset)
ATTACK_LABELS = {
    0: "Benign",
    1: "DoS",
    2: "Reconnaissance",
    3: "Exploits",
    4: "Exfiltration",
}

# Feature categories for attribution
FEATURE_CATEGORIES = {
    "network": ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"],
    "traffic": ["packet_size", "duration", "bytes_sent", "bytes_received"],
    "timing": ["inter_arrival_time", "flow_duration"],
    "behavioral": ["connection_count", "unique_destinations"],
}

# Streamlit dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Hybrid AI Cyber Defense & IDS Dashboard",
    "page_title": "AI Cyber Defense",
    "layout": "wide",
    "theme": "dark",
    "refresh_interval": 5,  # Seconds for data refresh
    "max_display_rows": 100,
    "risk_score_threshold": 0.7,  # Threshold for high-risk alerts
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": str(LOG_DIR / "training.log"),
}

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Reproducibility
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
