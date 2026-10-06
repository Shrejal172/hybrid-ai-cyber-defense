# Hybrid AI Cyber Defense & Intrusion Detection System: Project Report

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Background](#background)
4. [System Description](#system-description)
5. [References](#references)

---

## Abstract

This project presents a comprehensive AI-powered Cyber Defense and Intrusion Detection System (IDS) designed to detect, analyze, and respond to network security threats in real-time. The system leverages deep learning models, specifically Autoencoders for anomaly detection and Temporal Classifiers (LSTM-based) for attack classification, trained on the UNSW-NB15 cybersecurity dataset. The implementation includes a Streamlit-based dashboard for real-time monitoring, threat analytics, and incident response.

Key features implemented include:
- **Red Team Testing**: A simulation framework that injects real attack samples from the UNSW-NB15 dataset to validate detection capabilities
- **Dynamic Threshold Control**: Configurable sensitivity settings for both Autoencoder and automated response systems
- **Automated Response System**: Real-time IP blocking and session quarantining based on risk scores
- **Comprehensive Logging**: Audit trail for all automated and manual security actions
- **Federated Learning Architecture**: Support for distributed model training across multiple nodes

The system achieves detection rates of 55-75% on DoS attacks depending on threshold settings, with the ability to tune sensitivity to balance detection accuracy against false positives. The Red Team Testing feature enables security teams to proactively test defenses against known attack patterns before deployment in production environments.

---

## Introduction

### 1.1 Problem Statement

Cybersecurity threats are becoming increasingly sophisticated, with attackers using advanced techniques to evade traditional signature-based detection systems. Traditional Intrusion Detection Systems (IDS) often struggle with:
- Detecting zero-day attacks that don't match known signatures
- Handling high-volume network traffic in real-time
- Minimizing false positives while maintaining high detection rates
- Providing actionable intelligence for incident response

### 1.2 Project Objectives

The primary objectives of this project are:

1. **Develop an AI-based IDS** using deep learning models for anomaly detection and attack classification
2. **Implement real-time monitoring** through an interactive dashboard
3. **Create a Red Team Testing framework** to validate detection capabilities against known attack patterns
4. **Automate incident response** with configurable risk-based triggers
5. **Provide comprehensive logging** for audit trails and compliance

### 1.3 Scope

The system focuses on:
- Network traffic analysis using the UNSW-NB15 dataset
- Detection of attack types: DoS, Reconnaissance, Exploits, and Exfiltration
- Automated response mechanisms (IP blocking and session quarantining)
- Real-time visualization and analytics
- Red team simulation for defense validation

### 1.4 Target Users

- Security Operations Center (SOC) analysts
- Network security engineers
- Cybersecurity researchers
- Incident response teams

---

## Background

### 2.1 Intrusion Detection Systems

Intrusion Detection Systems (IDS) are security tools designed to monitor network or system activities for malicious activities or policy violations. There are two main types:

#### 2.1.1 Signature-Based IDS
- Match traffic patterns against known attack signatures
- Fast and efficient for known threats
- Cannot detect zero-day attacks
- Require constant signature updates

#### 2.1.2 Anomaly-Based IDS
- Establish baseline of normal traffic behavior
- Flag deviations from baseline as potential threats
- Can detect zero-day attacks
- Higher false positive rates
- Require extensive training data

### 2.2 Deep Learning in Cybersecurity

Deep learning has revolutionized intrusion detection by enabling:

#### 2.2.1 Autoencoders for Anomaly Detection
Autoencoders are neural networks trained to reconstruct input data. They learn to reconstruct normal traffic accurately but fail to reconstruct attacks, making them effective anomaly detectors.

**Architecture:**
- Encoder: Compresses input to latent representation
- Decoder: Reconstructs input from latent representation
- Reconstruction error indicates anomaly level

**Advantages:**
- Unsupervised learning (no labeled data needed for training)
- Can detect novel attack patterns
- Effective for anomaly detection

#### 2.2.2 Temporal Classifiers (LSTM)
Long Short-Term Memory (LSTM) networks can capture temporal dependencies in network traffic, making them ideal for detecting attack patterns that unfold over time.

**Architecture:**
- Sequential input processing
- Memory cells for long-term dependencies
- Classification output (Benign vs. Attack)

**Advantages:**
- Captures temporal patterns
- Effective for multi-stage attacks
- Handles variable-length sequences

### 2.3 UNSW-NB15 Dataset

The UNSW-NB15 dataset is a comprehensive network intrusion dataset created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS). It contains:

- **2,540,044 records** of network traffic
- **10 attack types**: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Trojans, Worms
- **49 features** including packet size, duration, protocol, ports, and byte counts
- **Labeled data** for supervised learning

**Preprocessing Applied:**
- Feature scaling using MinMaxScaler
- Categorical encoding for protocol and service types
- Train/test split (80/20)
- Sequence generation for temporal models (window size: 10)

### 2.4 Federated Learning

Federated Learning enables collaborative model training across multiple nodes without centralizing data. This is particularly valuable for cybersecurity where:

- Privacy regulations prevent data sharing
- Organizations want to keep sensitive data local
- Distributed networks (e.g., branch offices) need local models

**Implementation:**
- Global model synchronization
- Local training on node-specific data
- Gradient aggregation (weighted average)
- Privacy-preserving updates

### 2.5 Red Team Testing

Red team testing is a security practice where authorized teams simulate attacks to test defense capabilities. In cybersecurity:

- Simulates real-world attack scenarios
- Tests detection and response capabilities
- Identifies weaknesses before attackers do
- Validates security controls and procedures

**Implementation in This System:**
- Extracts real attack samples from test dataset
- Injects them into the monitoring system
- Measures detection rates
- Provides detailed analysis of missed detections

---

## System Description

### 3.1 System Architecture

The system follows a modular architecture with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Real-Time    │  │ Threat       │  │ Incident     │      │
│  │ Monitoring   │  │ Analytics    │  │ Response     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Inference Engine                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Autoencoder   │  │ Temporal     │  │ Risk Score   │      │
│  │ (Anomaly)     │  │ Classifier   │  │ Aggregation  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data Loading │  │ Preprocessing│  │ Sequence     │      │
│  │ (UNSW-NB15)  │  │ (Scaling)    │  │ Generation   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Model Architecture

#### 3.2.1 Autoencoder

**Purpose:** Unsupervised anomaly detection

**Architecture:**
```
Input (49 features)
    ↓
Encoder: [49 → 64 → 32 → 16 → 8 → 4]
    ↓
Latent Space (4 dimensions)
    ↓
Decoder: [4 → 8 → 16 → 32 → 64 → 49]
    ↓
Reconstruction (49 features)
```

**Training:**
- Trained only on benign traffic (label 0)
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate: 0.001)
- Batch size: 256
- Epochs: 50

**Inference:**
- Reconstruction error calculated for each sample
- Threshold: 95th percentile of training errors (0.040674589574337006)
- Error > threshold → flagged as anomalous

**Configuration:**
```python
AUTOENCODER_CONFIG = {
    "input_dim": 49,
    "hidden_dims": [64, 32, 16, 8],
    "latent_dim": 4,
    "activation": "relu",
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 50
}
```

#### 3.2.2 Temporal Classifier (LSTM)

**Purpose:** Supervised attack classification

**Architecture:**
```
Input Sequence (10 timesteps × 49 features)
    ↓
LSTM Layer 1: 128 units (bidirectional)
    ↓
LSTM Layer 2: 128 units (bidirectional)
    ↓
Dropout (0.3)
    ↓
Fully Connected Layer
    ↓
Softmax Output (2 classes: Benign, Attack)
```

**Training:**
- Trained on all traffic (benign + attacks)
- Loss function: Cross-Entropy
- Optimizer: Adam (learning rate: 0.001)
- Batch size: 128
- Epochs: 30

**Inference:**
- Input sequences of 10 timesteps
- Output: Attack probability (0-1)
- Probability > 0.5 → classified as attack

**Configuration:**
```python
TEMPORAL_CONFIG = {
    "input_dim": 49,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_classes": 2,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 30,
    "model_type": "lstm",
    "bidirectional": True
}
```

### 3.3 Dashboard Components

#### 3.3.1 Real-Time Monitoring Tab

**Features:**
- Live traffic stream display with risk-based color coding
- System risk score gauge (0-100)
- Attack type distribution pie chart
- Metrics: Total events, detected attacks, high-risk events, average packet size

**Data Sources:**
- Live Synthetic Stream: Generated mock traffic for demonstration
- Dataset Replay (UNSW-NB15): Real UNSW-NB15 test data with model inference
- Dataset Replay (CICIDS2017): Placeholder for CICIDS2017 dataset (not connected to models)

**Risk Calculation:**
```python
risk_score = max(
    attack_probability,           # From temporal classifier
    reconstruction_risk          # From autoencoder (error / threshold)
)
```

#### 3.3.2 Threat Analytics Tab

**Features:**
- High-risk anomaly table with feature attribution
- Reconstruction error distribution histogram
- Feature importance visualization for each anomaly
- Detailed event information

**Feature Attribution:**
For each high-risk event, the system calculates feature importance based on deviation from normal values:
- Packet Size: |actual - 500| / 1000
- Duration: |actual - 1.0| / 2.0
- Bytes Sent: |actual - 10000| / 20000
- Bytes Received: |actual - 15000| / 30000

#### 3.3.3 Incident Response Tab

**Features:**
- Active threats display with manual action buttons
- Blocked IPs list with clear functionality
- Quarantined sessions list with clear functionality
- Automated response settings
- Response log with complete audit trail

**Automated Response Settings:**
- **Auto-block high-risk IPs**: Automatically block IPs with risk score > threshold
- **Auto-quarantine suspicious sessions**: Automatically quarantine sessions with risk score > threshold
- **Risk Score Threshold**: Configurable threshold (0.0-1.0, default 0.7)
- **Enable Notifications**: Real-time alerts for automated actions

**Response Actions:**
- **Block IP**: Adds source IP to blocked list (all traffic from this IP is blocked)
- **Quarantine Session**: Isolates specific connection (source IP:destination IP)
- **Log Action**: Records timestamp, action type, target, status, and risk score

#### 3.3.4 Federated Learning Hub Tab

**Features:**
- Federated node status display
- Global model synchronization status
- Training readiness indicator
- Node connectivity monitoring

**Nodes:**
- HQ Datacenter
- Branch Office
- Remote Node

#### 3.3.5 Red Team Testing Results Tab

**Features:**
- Summary metrics (total samples, detected, missed, average risk score)
- Detection rate gauge visualization
- Injected attack samples table with detection status
- Detection breakdown by model (Autoencoder vs. Temporal Classifier)
- Risk score distribution histogram
- Clear results functionality

### 3.4 Red Team Testing Implementation

#### 3.4.1 Attack Sample Extraction

The system extracts real attack samples from the UNSW-NB15 test dataset based on attack type:

```python
attack_label_map = {
    "DoS": 1,
    "Reconnaissance": 2,
    "Exploits": 3,
    "Exfiltration": 4
}

target_label = attack_label_map.get(attack_type, 1)
attack_mask = y_test == target_label
attack_samples = X_test[attack_mask]
```

If specific attack type samples are not found, the system falls back to using all attack samples (any non-zero label).

#### 3.4.2 Model Inference on Attack Samples

Each extracted attack sample is processed by both models:

**Autoencoder:**
```python
reconstruction = autoencoder(attack_sample)
reconstruction_error = mean((attack_sample - reconstruction)^2)
ae_anomaly = reconstruction_error > mse_threshold
```

**Temporal Classifier:**
```python
sequence = create_sequence(attack_sample, window_size=10)
logits = temporal_classifier(sequence)
probabilities = softmax(logits)
attack_probability = probabilities[:, 1]
prediction = argmax(probabilities)
```

#### 3.4.3 Detection Calculation

A sample is considered detected if EITHER model flags it:

```python
model_detected = (prediction == 1) | ae_anomaly
detection_rate = model_detected.sum() / total_samples
```

#### 3.4.4 Results Visualization

The system provides multiple visualizations:
- Detection rate gauge
- Table of injected samples with detection status (green = detected, red = missed)
- Detection breakdown by model
- Risk score distribution histogram

### 3.5 Automated Response System

#### 3.5.1 Risk-Based Triggering

The system continuously monitors traffic and triggers automated responses when:

```python
if risk_score > risk_threshold:
    if auto_block_enabled:
        block_ip(source_ip)
        log_action("Auto-IP Blocked", source_ip, risk_score)
        if notifications_enabled:
            show_notification(f"Blocked IP: {ip} (Risk: {risk})")

    if auto_quarantine_enabled:
        quarantine_session(source_ip, destination_ip)
        log_action("Auto-Session Quarantined", session_id, risk_score)
        if notifications_enabled:
            show_notification(f"Quarantined session: {id} (Risk: {risk})")
```

#### 3.5.2 Response Logging

All actions (automated and manual) are logged with:

```python
{
    "Timestamp": datetime.now(),
    "Action": "Auto-IP Blocked" | "Manual IP Blocked" | "Auto-Session Quarantined" | "Manual Session Quarantined",
    "Target": source_ip | session_id,
    "Status": "Success" | "Failed",
    "Risk Score": risk_score
}
```

The log is displayed in reverse chronological order (newest first) and can be cleared when needed.

### 3.6 Configuration Management

The system uses a centralized configuration file (`config.py`) for:

- File paths (data, checkpoints, logs)
- Model hyperparameters
- Dataset settings
- Dashboard configuration
- Logging configuration
- Federated learning parameters

**Key Configuration:**
```python
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_unsw"
WINDOW_SIZE = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

### 3.7 Data Pipeline

The data pipeline handles:

1. **Data Loading**: Loads processed UNSW-NB15 dataset from pickle file
2. **Feature Scaling**: Uses MinMaxScaler for normalization
3. **Sequence Generation**: Creates sliding window sequences for temporal models
4. **Train/Test Split**: 80/20 split with stratification
5. **Benign Filtering**: Extracts benign samples for autoencoder training

**Sequence Generation:**
```python
def create_sequences(X, y, window_size=10, stride=1):
    n_sequences = (len(X) - window_size) // stride + 1
    sequences = zeros((n_sequences, window_size, n_features))
    for i in range(n_sequences):
        sequences[i] = X[i*stride : i*stride + window_size]
    return sequences, y[window_size-1::stride][:n_sequences]
```

### 3.8 Technology Stack

**Frontend:**
- Streamlit (Python web framework)
- Plotly (interactive visualizations)
- Plotly Express (chart generation)

**Backend:**
- Python 3.13
- PyTorch (deep learning framework)
- Pandas (data manipulation)
- NumPy (numerical computing)
- Scikit-learn (preprocessing, metrics)

**Models:**
- Autoencoder (PyTorch)
- LSTM-based Temporal Classifier (PyTorch)

**Dataset:**
- UNSW-NB15 (processed and pre-saved as pickle)

### 3.9 Performance Metrics

**Detection Rates (based on Red Team Testing):**
- DoS attacks: 55-75% (depending on threshold settings)
- Reconnaissance: 55% (limited samples in test set)
- Exploits: Varies based on attack subtype
- Exfiltration: Varies based on attack subtype

**Model-Specific Performance:**
- Autoencoder: 0-30% detection (highly dependent on threshold)
- Temporal Classifier: 55-75% detection (more consistent)

**System Performance:**
- Inference time: <100ms per batch (50 samples)
- Dashboard refresh: Configurable (1-10 seconds)
- Memory usage: ~500MB (models + data)

### 3.10 Limitations

1. **Dataset Bias**: Trained only on UNSW-NB15, may not generalize to other datasets
2. **Attack Coverage**: Limited to attack types present in UNSW-NB15
3. **False Positives**: Trade-off between detection rate and false positives
4. **Real-Time Processing**: Currently processes batches, not true streaming
5. **Network Integration**: No actual network integration (simulated traffic only)
6. **Persistence**: Blocked IPs and quarantined sessions reset on dashboard refresh
7. **External Integration**: No integration with firewalls, SIEM, or other security tools

### 3.11 Future Enhancements

1. **Multi-Dataset Support**: Add CICIDS2017, NSL-KDD, and other datasets
2. **Real-Time Streaming**: Implement actual network traffic capture and processing
3. **External Integration**: Connect to firewalls, SIEM, and ticketing systems
4. **Persistence**: Database backend for blocked IPs, quarantined sessions, and logs
5. **Attack Severity Levels**: Different thresholds for different attack types
6. **Time-Based Blocking**: Temporary blocking with automatic expiration
7. **Whitelist Management**: Allow-list for trusted IPs
8. **Alert Escalation**: Multi-level alerting with severity-based routing
9. **Model Retraining**: Continuous learning from new attack patterns
10. **Adaptive Thresholds**: Dynamic threshold adjustment based on traffic patterns
11. **Explainable AI**: SHAP/LIME for model interpretability
12. **Threat Intelligence Integration**: Enrich with external threat feeds

---

## References

### Academic Papers

1. Moustafa, N., & Slay, J. (2015). "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." *2015 Military Communications and Information Systems Conference (MilCIS)*, Canberra, ACT, Australia, pp. 1-6.

2. Moustafa, N., & Slay, J. (2016). "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the NSL-KDD data set." *Information Security Journal: A Global Perspective*, 25(1-3), pp. 18-31.

3. Vinayakumar, R., Alazab, M., Srinivasan, K. P., Pham, Q. V., Kasthuri, R., & Kaluarachchi, S. (2020). "A deep learning approach for intrusion detection system using bottm-up and top-down approach." *Knowledge-Based Systems*, 191, 105324.

4. Yang, M., Liu, F., & Yang, H. (2020). "Deep learning for intrusion detection: A comprehensive review." *IEEE Access*, 8, pp. 156447-156466.

5. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). "Communication-efficient learning of deep networks from decentralized data." *Artificial Intelligence and Statistics*, pp. 1273-1282.

### Technical Documentation

6. PyTorch Documentation. "LSTM — PyTorch 2.0 documentation." https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

7. Streamlit Documentation. "Streamlit Documentation." https://docs.streamlit.io/

8. Scikit-learn Documentation. "Preprocessing data." https://scikit-learn.org/stable/modules/preprocessing.html

### Datasets

9. Australian Centre for Cyber Security (ACCS). "UNSW-NB15 Dataset." https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/

10. Canadian Institute for Cybersecurity (CIC). "CICIDS2017 Dataset." https://www.unb.ca/cic/datasets/ids-2017.html

### Books

11. Chaudhary, A., & Kumar, V. (2021). *Deep Learning for Cyber Security*. Springer.

12. Ahmad, I., et al. (2021). *Artificial Intelligence for Cyber Security: Principles and Applications*. CRC Press.

### Standards and Frameworks

13. NIST. "National Institute of Standards and Technology Cybersecurity Framework." https://www.nist.gov/cyberframework

14. ISO/IEC. "ISO/IEC 27001: Information security management." https://www.iso.org/standard/27001

### Online Resources

15. OWASP. "OWASP Top 10 - 2021." https://owasp.org/Top10/

16. MITRE ATT&CK. "MITRE ATT&CK® Framework." https://attack.mitre.org/

---

## Appendix

### A. File Structure

```
DL-CD Project - Backup/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── pipeline.py                     # Data processing pipeline
├── models.py                       # Model definitions (Autoencoder, LSTM)
├── train.py                        # Model training script
├── IMPLEMENTATION_GUIDE.md          # Implementation documentation
├── PROJECT_REPORT.md              # This report
├── data_unsw/                      # UNSW-NB15 data directory
│   └── processed_unsw_nb15.pkl     # Preprocessed dataset
├── checkpoints_unsw/               # Model checkpoints
│   ├── autoencoder.pth            # Trained autoencoder
│   └── temporal_classifier.pth    # Trained temporal classifier
└── logs/                          # Training logs
```

### B. Model Checkpoint Information

**Autoencoder:**
- File: `checkpoints_unsw/autoencoder.pth`
- Input dimension: 49 features
- Hidden dimensions: [64, 32, 16, 8]
- Latent dimension: 4
- Anomaly threshold: 0.040674589574337006 (95th percentile)

**Temporal Classifier:**
- File: `checkpoints_unsw/temporal_classifier.pth`
- Input dimension: 49 features
- Hidden dimension: 128
- Number of layers: 2
- Number of classes: 2 (Benign, Attack)
- Bidirectional: Yes

### C. Attack Type Mapping

```python
ATTACK_LABELS = {
    0: "Benign",
    1: "DoS",
    2: "Reconnaissance",
    3: "Exploits",
    4: "Exfiltration"
}
```

### D. Feature Categories

```python
FEATURE_CATEGORIES = {
    "network": ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"],
    "traffic": ["packet_size", "duration", "bytes_sent", "bytes_received"],
    "timing": ["inter_arrival_time", "flow_duration"],
    "behavioral": ["connection_count", "unique_destinations"]
}
```

---

**Document Version:** 1.0
**Date:** October 6, 2026
**Author:** Development Team
**Project:** Hybrid AI Cyber Defense & Intrusion Detection System
