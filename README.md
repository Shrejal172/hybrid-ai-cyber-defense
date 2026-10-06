# Hybrid AI Cyber Defense & Intrusion Detection System

An end-to-end AI-powered cybersecurity system featuring dual-engine detection (Autoencoder + LSTM/Transformer), federated learning for privacy, and an interactive SOC dashboard.

## 🏗️ Architecture

### Components

1. **Data Pipeline** (`pipeline.py`)
   - Modular data ingestion and preprocessing
   - Feature scaling with MinMaxScaler
   - Categorical encoding
   - Time-series sliding window aggregation
   - Support for UNSW-NB15 and CICIDS2017 datasets

2. **Dual-Engine AI Core** (`models.py`)
   - **Autoencoder**: Unsupervised spatial anomaly detector trained on benign traffic
   - **Temporal Classifier**: LSTM/Transformer-based attack classifier for multi-stage attack detection
   - Dynamic percentile thresholding for anomaly detection
   - Zero-day threat detection capability

3. **Privacy Layer** (`federated.py`)
   - Federated Averaging (FedAvg) simulation
   - 2 simulated client nodes with independent training
   - Gradient encryption simulation
   - Central server aggregation and weight redistribution

4. **Interactive SOC Dashboard** (`app.py`)
   - Real-time traffic monitoring
   - Live system risk score gauge
   - High-risk anomaly tables with feature attribution
   - Automated incident response (Block IP / Quarantine Session)
   - Multi-tab interface (Monitoring, Analytics, Incident Response)

5. **Training Script** (`train.py`)
   - Unified training pipeline for both models
   - Support for centralized and federated learning
   - Comprehensive evaluation metrics (Precision, Recall, F1-Score, ROC-AUC)
   - Model checkpointing

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Pandas, NumPy, Scikit-Learn
- Streamlit, Plotly

See `requirements.txt` for complete dependencies.

## 🚀 Installation

```bash
# Navigate to the project directory
cd "DL-CD Project - main"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # On Windows
# OR source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### 1. Train Models

Train both models with centralized learning:
```bash
python train.py --model both
```

Train only the autoencoder:
```bash
python train.py --model autoencoder
```

Train only the temporal classifier:
```bash
python train.py --model temporal
```

Train with federated learning:
```bash
python train.py --model both --federated
```

Use your own dataset:
```bash
python train.py --data-path /path/to/your/dataset.csv
```

### 2. Launch Dashboard

```bash
python -m streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### 3. Module-Specific Testing

Test data pipeline:
```bash
python pipeline.py
```

Test models:
```bash
python models.py
```

Test federated learning:
```bash
python federated.py
```

## 📊 Attack Classes

The system classifies network traffic into the following categories:

- **Benign**: Normal traffic
- **DoS**: Denial of Service attacks
- **Reconnaissance**: Network scanning and probing
- **Exploits**: Vulnerability exploitation attempts
- **Exfiltration**: Data theft and unauthorized data transfer

## 🔧 Configuration

Edit `config.py` to customize:

- Model hyperparameters (hidden dimensions, learning rates, etc.)
- Data paths and window sizes
- Federated learning settings (number of clients, rounds, etc.)
- Dashboard settings (refresh interval, risk thresholds, etc.)
- Attack label mappings

## 📈 Evaluation Metrics

The system tracks the following metrics:

- **Precision**: Percentage of correctly identified threats
- **Recall**: Percentage of actual threats detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed classification breakdown

## 🏛️ Project Structure

```
DL-CD Project - main/
├── config.py              # Configuration and hyperparameters
├── pipeline.py            # Data preprocessing and feature engineering
├── models.py              # Autoencoder and Temporal Classifier models
├── federated.py           # Federated learning simulation
├── train.py               # Training script with evaluation
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                  # Data directory
│   ├── network_traffic.csv  # Raw dataset (optional)
│   └── processed_data.pkl   # Preprocessed data
├── checkpoints/           # Saved model checkpoints
│   ├── autoencoder.pth
│   └── temporal_classifier.pth
└── logs/                  # Training logs
    └── training.log
```

## 🔒 Privacy & Security

- **Federated Learning**: Training happens locally on client nodes; only weight updates are shared
- **Gradient Encryption**: Simulated encryption of gradients before aggregation
- **Data Isolation**: Each client works with a subset of data
- **Zero-Day Detection**: Autoencoder can detect unseen attack patterns via reconstruction error

## 🎯 Key Features

### Autoencoder (Unsupervised Anomaly Detection)
- Trains exclusively on benign traffic
- Detects point anomalies and zero-day threats
- Dynamic percentile-based thresholding
- Reconstruction error as anomaly score

### Temporal Classifier (Supervised Attack Classification)
- LSTM architecture for sequence modeling
- Transformer option for attention-based classification
- Bidirectional processing for context awareness
- Multi-class attack type classification

### Federated Learning
- FedAvg algorithm for weight aggregation
- Simulated privacy-preserving gradient exchange
- Configurable number of clients and training rounds
- Weighted averaging based on client data size

### Interactive Dashboard
- Real-time traffic visualization
- Risk score gauge with color-coded alerts
- Feature attribution for anomaly interpretation
- Automated incident response controls
- Dark theme optimized for SOC environments

## 📝 Example Workflow

1. **Data Preparation**: Place your network traffic dataset in `data/network_traffic.csv`
2. **Training**: Run `python train.py --model both` to train both models
3. **Evaluation**: Check `logs/training_results.json` for performance metrics
4. **Deployment**: Launch the dashboard with `streamlit run app.py`
5. **Monitoring**: Monitor real-time traffic, analyze anomalies, and respond to threats

## 🤝 Contributing

## ⚠️ Important Notes

- **Virtual Environment**: If you renamed the project folder after creating the virtual environment, the `.venv` folder may have incorrect paths. Always use `python -m streamlit run app.py` instead of `streamlit run app.py` to avoid path issues.
- **Folder Renaming**: If you move or rename the project folder, recreate the virtual environment to ensure all paths are correct.

## 🤝 Contributing

This is a demonstration project for cybersecurity AI research. Feel free to extend it with:

- Additional attack classifiers
- Real-time data streaming integration
- Advanced federated learning algorithms
- Enhanced visualization features
- Integration with SIEM systems

## 📄 License

This project is provided as-is for educational and research purposes.

## ⚠️ Disclaimer

This system is a demonstration/prototype and should not be used as the sole security mechanism in production environments. Always implement defense-in-depth strategies and consult with security professionals for production deployments.
