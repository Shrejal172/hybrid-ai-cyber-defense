"""
Streamlit Dashboard for Hybrid AI Cyber Defense & Intrusion Detection System.

This module provides an interactive SOC dashboard with:

- Multi-tab interface
- Real-time monitoring
- UNSW-NB15 dataset replay
- Trained Autoencoder anomaly detection
- Trained Temporal Classifier attack detection
- Analytics
- Incident response controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime, timedelta
import time
import logging

import config
from pipeline import DataPipeline
from models import Autoencoder, TemporalClassifier, load_model

import torch


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=config.DASHBOARD_CONFIG["page_title"],
    page_icon="🛡️",
    layout=config.DASHBOARD_CONFIG["layout"],
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stMetric {
            background-color: rgba(128, 128, 128, 0.1);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }

        .alert-box {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }

        .alert-high {
            background-color: rgba(255, 75, 75, 0.1);
            border: 2px solid rgba(255, 75, 75, 0.5);
            color: #ff4b4b;
        }

        .alert-medium {
            background-color: rgba(255, 159, 75, 0.1);
            border: 2px solid rgba(255, 159, 75, 0.5);
            color: #ff9f4b;
        }

        .alert-low {
            background-color: rgba(75, 255, 107, 0.1);
            border: 2px solid rgba(75, 255, 107, 0.5);
            color: #4bff6b;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """
    Load trained models for inference.

    Returns:
        Tuple of (autoencoder, temporal_classifier, pipeline)
    """

    logger.info("Loading models...")

    # --------------------------------------------------------
    # Load UNSW-NB15 pipeline
    # --------------------------------------------------------

    pipeline = DataPipeline()

    unsw_path = (
        config.PROJECT_ROOT
        / "data_unsw"
        / "processed_unsw_nb15.pkl"
    )

    if unsw_path.exists():
        logger.info(
            "UNSW-NB15 processed dataset found: %s",
            unsw_path
        )
    else:
        logger.warning(
            "UNSW-NB15 processed dataset not found: %s",
            unsw_path
        )

    # --------------------------------------------------------
    # Load Autoencoder
    # --------------------------------------------------------

    autoencoder = None

    ae_path = (
        config.CHECKPOINT_DIR
        / "autoencoder.pth"
    )

    if ae_path.exists():

        ae_config = config.AUTOENCODER_CONFIG

        autoencoder = Autoencoder(
            input_dim=ae_config["input_dim"],
            hidden_dims=ae_config["hidden_dims"],
            latent_dim=ae_config["latent_dim"],
            activation=ae_config["activation"],
            dropout=ae_config["dropout"]
        )

        autoencoder = load_model(
            autoencoder,
            ae_path
        )

        autoencoder.eval()

        logger.info(
            "Autoencoder loaded from %s",
            ae_path
        )

    else:

        logger.warning(
            "Autoencoder checkpoint not found: %s",
            ae_path
        )

    # --------------------------------------------------------
    # Load Temporal Classifier
    # --------------------------------------------------------

    temporal_classifier = None

    tc_path = (
        config.CHECKPOINT_DIR
        / "temporal_classifier.pth"
    )

    if tc_path.exists():

        tc_config = config.TEMPORAL_CONFIG

        temporal_classifier = TemporalClassifier(
            input_dim=tc_config["input_dim"],
            hidden_dim=tc_config["hidden_dim"],
            num_layers=tc_config["num_layers"],
            num_classes=tc_config["num_classes"],
            dropout=tc_config["dropout"],
            model_type=tc_config["model_type"],
            bidirectional=tc_config["bidirectional"]
        )

        temporal_classifier = load_model(
            temporal_classifier,
            tc_path
        )

        temporal_classifier.eval()

        logger.info(
            "Temporal classifier loaded from %s",
            tc_path
        )

    else:

        logger.warning(
            "Temporal classifier checkpoint not found: %s",
            tc_path
        )

    logger.info("Models loaded successfully")

    return (
        autoencoder,
        temporal_classifier,
        pipeline
    )


# ============================================================
# REAL UNSW-NB15 INFERENCE
# ============================================================

def generate_unsw_traffic_stream(
    pipeline: DataPipeline,
    autoencoder,
    temporal_classifier,
    n_samples: int = 50
) -> pd.DataFrame:

    """
    Generate dashboard traffic from real UNSW-NB15 test data
    using the trained Autoencoder and Temporal Classifier.
    """

    # --------------------------------------------------------
    # Load processed UNSW-NB15 data
    # --------------------------------------------------------

    data = pipeline.prepare_full_pipeline(
        create_sequences=True
    )

    # --------------------------------------------------------
    # Determine available samples
    # --------------------------------------------------------

    available = min(
        n_samples,
        len(data["X_test"]),
        len(data["X_test_seq"])
    )

    if available == 0:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Test features
    # --------------------------------------------------------

    sequence_length = data["X_test_seq"].shape[1]

    X_test = (
        data["X_test"]
        .iloc[
            sequence_length - 1:
            sequence_length - 1 + available
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Autoencoder inference
    # --------------------------------------------------------

    ae_errors = np.zeros(
        available,
        dtype=np.float32
    )

    if autoencoder is not None:

        autoencoder.eval()

        x_ae = torch.tensor(
            X_test.values,
            dtype=torch.float32
        )

        with torch.no_grad():

            reconstruction = autoencoder(
                x_ae
            )

            reconstruction_errors = torch.mean(
                (x_ae - reconstruction) ** 2,
                dim=1
            )

        ae_errors = (
            reconstruction_errors
            .cpu()
            .numpy()
        )

    # Use default threshold for real-time monitoring
    anomaly_threshold = 0.040674589574337006

    ae_anomaly = (
        ae_errors > anomaly_threshold
    )

    # --------------------------------------------------------
    # Temporal classifier inference
    # --------------------------------------------------------

    predictions = np.zeros(
        available,
        dtype=int
    )

    attack_probability = np.zeros(
        available,
        dtype=np.float32
    )

    if temporal_classifier is not None:

        temporal_classifier.eval()

        X_seq = (
            data["X_test_seq"][:available]
        )

        x_temporal = torch.tensor(
            X_seq,
            dtype=torch.float32
        )

        with torch.no_grad():

            logits = temporal_classifier(
                x_temporal
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            ).cpu().numpy()

            attack_probability = (
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    # --------------------------------------------------------
    # Combine model results
    # --------------------------------------------------------

    model_attack = (
        (predictions == 1)
        |
        ae_anomaly
    )

    # --------------------------------------------------------
    # Combined risk score
    # --------------------------------------------------------

    reconstruction_risk = np.minimum(
        ae_errors / anomaly_threshold,
        1.0
    )

    risk_score = np.maximum(
        attack_probability,
        reconstruction_risk
    )

    risk_score = np.clip(
        risk_score,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Create dashboard dataframe
    # --------------------------------------------------------

        # Restore original UNSW feature values for display
    X_test_original = pd.DataFrame(
        pipeline.scaler.inverse_transform(
            X_test.values
        ),
        columns=X_test.columns
    )

    actual_labels = data["y_test_seq"][:available]

    traffic_df = pd.DataFrame({

        "timestamp": [
            datetime.now() - timedelta(seconds=i)
            for i in range(available)
        ],

        "event_id": [
            f"UNSW-{i + 1:05d}"
            for i in range(available)
        ],

        "src_ip": [
            f"UNSW-Source-{i + 1:03d}"
            for i in range(available)
        ],

        "dst_ip": [
            f"UNSW-Destination-{i + 1:03d}"
            for i in range(available)
        ],

        "src_port": ["N/A"] * available,
        "dst_port": ["N/A"] * available,
        "protocol": ["UNSW-NB15"] * available,

        "packet_size": (
            X_test_original["sbytes"].values
            + X_test_original["dbytes"].values
        ),

        "duration": X_test_original["dur"].values,

        "bytes_sent": X_test_original["sbytes"].values,

        "bytes_received": X_test_original["dbytes"].values,

        "risk_score": risk_score,

        "attack_type": [
            "Attack" if attack else "Benign"
            for attack in model_attack
        ],

        "actual_label": actual_labels,

        "model_prediction": predictions,

        "attack_probability": attack_probability,

        "reconstruction_error": ae_errors,

        "ae_anomaly": ae_anomaly
    })

    return (
        traffic_df
        .sort_values(
            "timestamp",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# RED TEAM TESTING
# ============================================================

def simulate_red_team_attack(
    pipeline: DataPipeline,
    autoencoder,
    temporal_classifier,
    attack_type: str = "DoS",
    n_samples: int = 20,
    mse_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Simulate a red team attack by injecting real attack samples
    from the UNSW-NB15 test dataset.

    Args:
        pipeline: DataPipeline instance
        autoencoder: Trained autoencoder model
        temporal_classifier: Trained temporal classifier model
        attack_type: Type of attack to simulate (DoS, Reconnaissance, Exploits, Exfiltration)
        n_samples: Number of attack samples to inject
        mse_threshold: Autoencoder MSE threshold for anomaly detection (from sidebar)

    Returns:
        DataFrame containing the injected attack samples with model predictions
    """

    # --------------------------------------------------------
    # Load processed UNSW-NB15 data
    # --------------------------------------------------------

    data = pipeline.prepare_full_pipeline(
        create_sequences=True
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    # --------------------------------------------------------
    # Map attack type to label
    # --------------------------------------------------------

    attack_label_map = {
        "DoS": 1,
        "Reconnaissance": 2,
        "Exploits": 3,
        "Exfiltration": 4
    }

    target_label = attack_label_map.get(attack_type, 1)

    # --------------------------------------------------------
    # Filter for specific attack type
    # --------------------------------------------------------

    attack_mask = y_test == target_label
    attack_samples = X_test[attack_mask]

    if len(attack_samples) == 0:
        st.warning(
            f"No {attack_type} samples found in test set. "
            f"Using all attack samples instead."
        )
        attack_mask = y_test != 0
        attack_samples = X_test[attack_mask]

    # --------------------------------------------------------
    # Limit to requested number of samples
    # --------------------------------------------------------

    available = min(n_samples, len(attack_samples))
    X_attack = attack_samples.iloc[:available].copy()

    if available == 0:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Autoencoder inference
    # --------------------------------------------------------

    ae_errors = np.zeros(
        available,
        dtype=np.float32
    )

    if autoencoder is not None:

        autoencoder.eval()

        x_ae = torch.tensor(
            X_attack.values,
            dtype=torch.float32
        )

        with torch.no_grad():

            reconstruction = autoencoder(
                x_ae
            )

            reconstruction_errors = torch.mean(
                (x_ae - reconstruction) ** 2,
                dim=1
            )

        ae_errors = (
            reconstruction_errors
            .cpu()
            .numpy()
        )

    # Use the provided MSE threshold from sidebar
    anomaly_threshold = mse_threshold

    ae_anomaly = (
        ae_errors > anomaly_threshold
    )

    # --------------------------------------------------------
    # Temporal classifier inference
    # --------------------------------------------------------

    predictions = np.zeros(
        available,
        dtype=int
    )

    attack_probability = np.zeros(
        available,
        dtype=np.float32
    )

    if temporal_classifier is not None:

        temporal_classifier.eval()

        # Create sequences for temporal classifier
        X_seq, _ = pipeline.create_sequences(
            X_attack,
            None
        )

        if len(X_seq) > 0:

            available_seq = min(available, len(X_seq))
            X_seq = X_seq[:available_seq]

            x_temporal = torch.tensor(
                X_seq,
                dtype=torch.float32
            )

            with torch.no_grad():

                logits = temporal_classifier(
                    x_temporal
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                seq_predictions = torch.argmax(
                    probabilities,
                    dim=1
                ).cpu().numpy()

                seq_attack_probability = (
                    probabilities[:, 1]
                    .cpu()
                    .numpy()
                )

            # Pad predictions to match available samples
            predictions[:available_seq] = seq_predictions
            attack_probability[:available_seq] = seq_attack_probability

    # --------------------------------------------------------
    # Calculate detection results
    # --------------------------------------------------------

    model_detected = (
        (predictions == 1)
        |
        ae_anomaly
    )

    detection_rate = (
        model_detected.sum() / available
        if available > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Restore original feature values for display
    # --------------------------------------------------------

    X_attack_original = pd.DataFrame(
        pipeline.scaler.inverse_transform(
            X_attack.values
        ),
        columns=X_attack.columns
    )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    attack_df = pd.DataFrame({

        "timestamp": [
            datetime.now() - timedelta(seconds=i)
            for i in range(available)
        ],

        "event_id": [
            f"REDTEAM-{attack_type}-{i + 1:03d}"
            for i in range(available)
        ],

        "src_ip": [
            f"ATTACKER-{i + 1:03d}"
            for i in range(available)
        ],

        "dst_ip": [
            f"TARGET-{i + 1:03d}"
            for i in range(available)
        ],

        "src_port": ["N/A"] * available,
        "dst_port": ["N/A"] * available,
        "protocol": ["REDTEAM"] * available,

        "packet_size": (
            X_attack_original["sbytes"].values
            + X_attack_original["dbytes"].values
        ),

        "duration": X_attack_original["dur"].values,

        "bytes_sent": X_attack_original["sbytes"].values,

        "bytes_received": X_attack_original["dbytes"].values,

        "risk_score": np.maximum(
            attack_probability,
            np.minimum(
                ae_errors / anomaly_threshold,
                1.0
            )
        ),

        "attack_type": [attack_type] * available,

        "actual_label": [target_label] * available,

        "model_prediction": predictions,

        "attack_probability": attack_probability,

        "reconstruction_error": ae_errors,

        "ae_anomaly": ae_anomaly,

        "model_detected": model_detected
    })

    return (
        attack_df
        .sort_values(
            "timestamp",
            ascending=False
        )
        .reset_index(drop=True)
    ), detection_rate


# ============================================================
# MOCK TRAFFIC
# ============================================================

def generate_mock_traffic_stream(
    n_samples: int = 100
) -> pd.DataFrame:

    """
    Generate mock network traffic stream for demonstration.
    """

    np.random.seed(
        int(time.time())
    )

    data = {

        "timestamp": [
            datetime.now() - timedelta(seconds=i)
            for i in range(n_samples)
        ],

        "src_ip": [
            f"192.168.1.{np.random.randint(1, 255)}"
            for _ in range(n_samples)
        ],

        "dst_ip": [
            f"10.0.0.{np.random.randint(1, 255)}"
            for _ in range(n_samples)
        ],

        "src_port": np.random.randint(
            1024,
            65535,
            n_samples
        ),

        "dst_port": np.random.choice(
            [80, 443, 22, 21, 3306, 5432],
            n_samples
        ),

        "protocol": np.random.choice(
            ["TCP", "UDP", "ICMP"],
            n_samples
        ),

        "packet_size": np.random.exponential(
            500,
            n_samples
        ),

        "duration": np.random.exponential(
            1.0,
            n_samples
        ),

        "bytes_sent": np.random.exponential(
            10000,
            n_samples
        ),

        "bytes_received": np.random.exponential(
            15000,
            n_samples
        ),

        "risk_score": np.random.beta(
            2,
            5,
            n_samples
        ),

        "attack_type": np.random.choice(
            [
                "Benign",
                "DoS",
                "Reconnaissance",
                "Exploits",
                "Exfiltration"
            ],
            n_samples,
            p=[
                0.7,
                0.1,
                0.08,
                0.07,
                0.05
            ]
        )
    }

    df = pd.DataFrame(data)

    return (
        df.sort_values(
            "timestamp",
            ascending=False
        )
    )


# ============================================================
# SYSTEM RISK SCORE
# ============================================================

def calculate_system_risk_score(
    traffic_df: pd.DataFrame
) -> float:

    """
    Calculate overall system risk score from model predictions.
    """

    if len(traffic_df) == 0:
        return 0.0

    recent_traffic = (
        traffic_df.head(100)
    )

    if "risk_score" in recent_traffic.columns:

        return float(
            np.clip(
                recent_traffic[
                    "risk_score"
                ].mean(),
                0.0,
                1.0
            )
        )

    return 0.0


# ============================================================
# RISK GAUGE
# ============================================================

def create_risk_gauge(
    risk_score: float
) -> go.Figure:

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number+delta",

            value=risk_score * 100,

            domain={
                "x": [0, 1],
                "y": [0, 1]
            },

            title={
                "text": "System Risk Score",
                "font": {
                    "size": 24,
                    "color": "#64748b"
                }
            },

            number={
                "font": {
                    "color": "#64748b"
                }
            },

            delta={
                "reference": 50,
                "font": {
                    "color": "#64748b"
                }
            },

            gauge={

                "axis": {
                    "range": [None, 100],
                    "tickwidth": 1,
                    "tickcolor":
                        "rgba(128,128,128,0.5)"
                },

                "bar": {
                    "color":
                        "rgba(59, 130, 246, 0.8)"
                },

                "bgcolor":
                    "rgba(128,128,128,0.1)",

                "borderwidth": 2,

                "bordercolor":
                    "rgba(128,128,128,0.3)",

                "steps": [

                    {
                        "range": [0, 30],
                        "color":
                            "rgba(34, 197, 94, 0.6)"
                    },

                    {
                        "range": [30, 60],
                        "color":
                            "rgba(234, 179, 8, 0.6)"
                    },

                    {
                        "range": [60, 80],
                        "color":
                            "rgba(249, 115, 22, 0.6)"
                    },

                    {
                        "range": [80, 100],
                        "color":
                            "rgba(239, 68, 68, 0.6)"
                    }
                ],

                "threshold": {

                    "line": {
                        "color":
                            "rgba(239, 68, 68, 0.8)",
                        "width": 4
                    },

                    "thickness": 0.75,

                    "value": 80
                }
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#64748b"
        },
        height=300
    )

    return fig


# ============================================================
# REAL-TIME MONITORING
# ============================================================

def render_real_time_monitoring_tab(
    autoencoder,
    temporal_classifier,
    pipeline
):

    st.header(
        "🔴 Real-Time Traffic Monitoring"
    )

    # --------------------------------------------------------
    # Select traffic source
    # --------------------------------------------------------

    if st.session_state.get(
        "telemetry_source"
    ) == "Dataset Replay (UNSW-NB15)":

        traffic_df = (
            generate_unsw_traffic_stream(
                pipeline,
                autoencoder,
                temporal_classifier,
                50
            )
        )

        st.info(
            "Telemetry Source: Real UNSW-NB15 "
            "dataset with trained AI models."
        )

    else:

        traffic_df = (
            generate_mock_traffic_stream(50)
        )

        st.info(
            "Telemetry Source: Live Synthetic Stream."
        )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_score = (
        calculate_system_risk_score(
            traffic_df
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Events",
            len(traffic_df)
        )

    with col2:

        attack_count = (
            traffic_df[
                "attack_type"
            ] != "Benign"
        ).sum()

        st.metric(
            "Detected Attacks",
            attack_count,
            delta=(
                f"{attack_count / len(traffic_df) * 100:.1f}%"
                if len(traffic_df) > 0
                else "0%"
            ),
            delta_color="inverse"
        )

    with col3:

        high_risk_count = (
            traffic_df[
                "risk_score"
            ] > 0.7
        ).sum()

        st.metric(
            "High Risk Events",
            high_risk_count
        )

    with col4:

        avg_packet_size = (
            traffic_df[
                "packet_size"
            ].mean()
        )

        st.metric(
            "Avg Packet Size",
            f"{avg_packet_size:.0f} bytes"
        )

    # --------------------------------------------------------
    # Risk gauge
    # --------------------------------------------------------

    st.subheader(
        "System Risk Score"
    )

    risk_gauge = create_risk_gauge(
        risk_score
    )

    st.plotly_chart(
        risk_gauge,
        use_container_width=True,
        key="realtime_risk_gauge"
    )

    # --------------------------------------------------------
    # Traffic stream
    # --------------------------------------------------------

    st.subheader(
        "Live Traffic Stream"
    )

    def highlight_risk(row):

        risk = row[
            "risk_score"
        ]

        if risk > 0.7:

            return [
                "background-color: rgba(255, 75, 75, 0.15)"
            ] * len(row)

        elif risk > 0.4:

            return [
                "background-color: rgba(255, 159, 75, 0.15)"
            ] * len(row)

        else:

            return [
                "background-color: rgba(75, 255, 107, 0.15)"
            ] * len(row)

    display_df = (
        traffic_df
        .head(20)
        .copy()
    )

    styled_df = (
        display_df
        .style
        .apply(
            highlight_risk,
            axis=1
        )
    )

    st.dataframe(
        styled_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Attack distribution
    # --------------------------------------------------------

    st.subheader(
        "Attack Type Distribution"
    )

    attack_counts = (
        traffic_df[
            "attack_type"
        ].value_counts()
    )

    fig = px.pie(
        values=attack_counts.values,
        names=attack_counts.index,
        title="Attack Type Distribution"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#64748b"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="realtime_attack_distribution"
    )


# ============================================================
# ANALYTICS
# ============================================================

def render_analytics_tab(
    autoencoder,
    temporal_classifier,
    pipeline
):

    st.header(
        "📊 Analytics & Anomaly Detection"
    )

    # --------------------------------------------------------
    # Select data source
    # --------------------------------------------------------

    if st.session_state.get(
        "telemetry_source"
    ) == "Dataset Replay (UNSW-NB15)":

        traffic_df = (
            generate_unsw_traffic_stream(
                pipeline,
                autoencoder,
                temporal_classifier,
                200
            )
        )

        st.info(
            "Analytics are based on real UNSW-NB15 "
            "test data and trained model predictions."
        )

    else:

        traffic_df = (
            generate_mock_traffic_stream(200)
        )

    # --------------------------------------------------------
    # High-risk anomalies
    # --------------------------------------------------------

    st.subheader(
        "High-Risk Anomalies"
    )

    high_risk_df = (
        traffic_df[
            traffic_df["risk_score"]
            >
            config.DASHBOARD_CONFIG[
                "risk_score_threshold"
            ]
        ]
        .head(20)
    )

    if len(high_risk_df) > 0:

        st.write(
            "### Feature Attribution"
        )

        for idx, row in (
            high_risk_df
            .head(5)
            .iterrows()
        ):

            with st.expander(
                f"Anomaly at {row['timestamp']} "
                f"- Risk: {row['risk_score']:.2f}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Event Details:**"
                    )

                    st.write(
                        f"- Event ID: "
                        f"{row.get('event_id', 'N/A')}"
                    )

                    st.write(
                        f"- Source: {row['src_ip']}"
                    )

                    st.write(
                        f"- Destination: "
                        f"{row['dst_ip']}"
                    )

                    st.write(
                        f"- Protocol: "
                        f"{row['protocol']}"
                    )

                    st.write(
                        f"- Attack Type: "
                        f"{row['attack_type']}"
                    )

                    st.write(
                        f"- Attack Probability: "
                        f"{row.get('attack_probability', 0):.2f}"
                    )

                    st.write(
                        f"- Reconstruction Error: "
                        f"{row.get('reconstruction_error', 0):.5f}"
                    )

                with col2:

                    st.write(
                        "**Feature Attribution:**"
                    )

                    features = {

                        "Packet Size":
                            abs(
                                row["packet_size"]
                                - 500
                            ) / 1000,

                        "Duration":
                            abs(
                                row["duration"]
                                - 1.0
                            ) / 2.0,

                        "Bytes Sent":
                            abs(
                                row["bytes_sent"]
                                - 10000
                            ) / 20000,

                        "Bytes Received":
                            abs(
                                row["bytes_received"]
                                - 15000
                            ) / 30000
                    }

                    feature_df = pd.DataFrame({

                        "Feature":
                            list(features.keys()),

                        "Importance":
                            list(features.values())
                    })

                    feature_df = (
                        feature_df
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                    )

                    fig = px.bar(
                        feature_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Feature Importance",
                        color="Importance",
                        color_continuous_scale="Reds"
                    )

                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={
                            "color": "#64748b"
                        }
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"analytics_feature_importance_{idx}"
                    )

        # ----------------------------------------------------
        # High-risk table
        # ----------------------------------------------------

        st.write(
            "### High-Risk Anomaly Table"
        )

        columns = [
            "timestamp",
            "src_ip",
            "dst_ip",
            "attack_type",
            "risk_score"
        ]

        if (
            "attack_probability"
            in high_risk_df.columns
        ):

            columns.append(
                "attack_probability"
            )

        if (
            "reconstruction_error"
            in high_risk_df.columns
        ):

            columns.append(
                "reconstruction_error"
            )

        st.dataframe(
            high_risk_df[columns],
            use_container_width=True
        )

    else:

        st.info(
            "No high-risk anomalies detected "
            "in the current time window."
        )

    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    if autoencoder is not None:

        st.subheader(
            "Reconstruction Error Distribution"
        )

        if (
            "reconstruction_error"
            in traffic_df.columns
        ):

            errors = (
                traffic_df[
                    "reconstruction_error"
                ].values
            )

        else:

            errors = np.random.exponential(
                0.01,
                100
            )

        fig = px.histogram(
            x=errors,
            nbins=30,
            title="Reconstruction Error Distribution",
            labels={
                "x": "Reconstruction Error",
                "y": "Count"
            }
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={
                "color": "#64748b"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="analytics_reconstruction_error"
        )


# ============================================================
# INCIDENT RESPONSE
# ============================================================

def render_incident_response_tab(
    autoencoder,
    temporal_classifier,
    pipeline
):

    st.header(
        "🚨 Incident Response"
    )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if "blocked_ips" not in st.session_state:

        st.session_state.blocked_ips = set()

    if "quarantined_sessions" not in st.session_state:

        st.session_state.quarantined_sessions = set()

    if "response_log" not in st.session_state:

        st.session_state.response_log = []

    # --------------------------------------------------------
    # Current threats
    # --------------------------------------------------------

    if st.session_state.get(
        "telemetry_source"
    ) == "Dataset Replay (UNSW-NB15)":

        traffic_df = (
            generate_unsw_traffic_stream(
                pipeline,
                autoencoder,
                temporal_classifier,
                30
            )
        )

    else:

        traffic_df = (
            generate_mock_traffic_stream(30)
        )

    threats_df = (
        traffic_df[
            traffic_df[
                "attack_type"
            ] != "Benign"
        ]
        .head(10)
    )

    if len(threats_df) > 0:

        st.subheader(
            "Active Threats"
        )

        for idx, row in (
            threats_df.iterrows()
        ):

            with st.container():

                col1, col2, col3 = (
                    st.columns([3, 1, 1])
                )

                with col1:

                    st.write(
                        f"**{row['attack_type']}** "
                        f"from {row['src_ip']} "
                        f"to {row['dst_ip']}"
                    )

                    st.write(
                        f"Risk Score: "
                        f"{row['risk_score']:.2f}"
                    )

                    if (
                        "attack_probability"
                        in row
                    ):

                        st.write(
                            f"Attack Probability: "
                            f"{row['attack_probability']:.2f}"
                        )

                with col2:

                    if st.button(
                        "Block IP",
                        key=f"block_{idx}"
                    ):

                        st.session_state.blocked_ips.add(
                            row["src_ip"]
                        )

                        # Log manual action
                        st.session_state.response_log.append({
                            "Timestamp": datetime.now(),
                            "Action": "Manual IP Blocked",
                            "Target": row["src_ip"],
                            "Status": "Success",
                            "Risk Score": row["risk_score"]
                        })

                        st.success(
                            f"Blocked {row['src_ip']}"
                        )

                with col3:

                    if st.button(
                        "Quarantine",
                        key=f"quarantine_{idx}"
                    ):

                        session_id = (
                            f"{row['src_ip']}:"
                            f"{row['dst_ip']}"
                        )

                        st.session_state.quarantined_sessions.add(
                            session_id
                        )

                        # Log manual action
                        st.session_state.response_log.append({
                            "Timestamp": datetime.now(),
                            "Action": "Manual Session Quarantined",
                            "Target": session_id,
                            "Status": "Success",
                            "Risk Score": row["risk_score"]
                        })

                        st.success(
                            f"Quarantined session "
                            f"{session_id}"
                        )

                st.divider()

    else:

        st.info(
            "No active threats detected."
        )

    # --------------------------------------------------------
    # Blocked IPs
    # --------------------------------------------------------

    st.subheader(
        "Blocked IPs"
    )

    if st.session_state.blocked_ips:

        blocked_df = pd.DataFrame({

            "IP Address":
                list(
                    st.session_state.blocked_ips
                ),

            "Blocked At":
                [
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ]
                *
                len(
                    st.session_state.blocked_ips
                )
        })

        st.dataframe(
            blocked_df,
            use_container_width=True
        )

        if st.button(
            "Clear All Blocked IPs"
        ):

            st.session_state.blocked_ips.clear()

            st.success(
                "All blocked IPs cleared"
            )

    else:

        st.info(
            "No IPs are currently blocked."
        )

    # --------------------------------------------------------
    # Quarantined sessions
    # --------------------------------------------------------

    st.subheader(
        "Quarantined Sessions"
    )

    if st.session_state.quarantined_sessions:

        quarantined_df = pd.DataFrame({

            "Session ID":
                list(
                    st.session_state.quarantined_sessions
                ),

            "Quarantined At":
                [
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ]
                *
                len(
                    st.session_state.quarantined_sessions
                )
        })

        st.dataframe(
            quarantined_df,
            use_container_width=True
        )

        if st.button(
            "Clear All Quarantined Sessions"
        ):

            st.session_state.quarantined_sessions.clear()

            st.success(
                "All quarantined sessions cleared"
            )

    else:

        st.info(
            "No sessions are currently quarantined."
        )

    # --------------------------------------------------------
    # Automated response settings
    # --------------------------------------------------------

    st.subheader(
        "Automated Response Settings"
    )

    col1, col2 = st.columns(2)

    with col1:

        auto_block = st.checkbox(
            "Auto-block high-risk IPs",
            value=False,
            help=(
                "Automatically block IPs "
                f"with risk score > threshold"
            )
        )

        auto_quarantine = st.checkbox(
            "Auto-quarantine suspicious sessions",
            value=False,
            help=(
                "Automatically quarantine "
                f"sessions with risk score > threshold"
            )
        )

    with col2:

        risk_threshold = st.slider(
            "Risk Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum risk score to trigger automated response"
        )

        notification_enabled = st.checkbox(
            "Enable Notifications",
            value=True,
            help=(
                "Send notifications "
                "for detected threats"
            )
        )

    # --------------------------------------------------------
    # Apply automated responses
    # --------------------------------------------------------

    if auto_block or auto_quarantine:

        high_risk_df = (
            traffic_df[
                traffic_df["risk_score"] > risk_threshold
            ]
        )

        if len(high_risk_df) > 0:

            new_actions = []

            for idx, row in high_risk_df.iterrows():

                src_ip = row["src_ip"]
                session_id = f"{row['src_ip']}:{row['dst_ip']}"

                # Auto-block IPs
                if auto_block and src_ip not in st.session_state.blocked_ips:

                    st.session_state.blocked_ips.add(src_ip)

                    new_actions.append({
                        "Timestamp": datetime.now(),
                        "Action": "Auto-IP Blocked",
                        "Target": src_ip,
                        "Status": "Success",
                        "Risk Score": row["risk_score"]
                    })

                    if notification_enabled:

                        st.warning(
                            f"🚨 Auto-blocked IP: {src_ip} "
                            f"(Risk: {row['risk_score']:.2f})"
                        )

                # Auto-quarantine sessions
                if auto_quarantine and session_id not in st.session_state.quarantined_sessions:

                    st.session_state.quarantined_sessions.add(session_id)

                    new_actions.append({
                        "Timestamp": datetime.now(),
                        "Action": "Auto-Session Quarantined",
                        "Target": session_id,
                        "Status": "Success",
                        "Risk Score": row["risk_score"]
                    })

                    if notification_enabled:

                        st.warning(
                            f"🚨 Auto-quarantined session: {session_id} "
                            f"(Risk: {row['risk_score']:.2f})"
                        )

            # Add new actions to response log
            if new_actions:

                st.session_state.response_log.extend(new_actions)

                st.success(
                    f"Automated response triggered: "
                    f"{len(new_actions)} action(s) taken"
                )

    # --------------------------------------------------------
    # Response log
    # --------------------------------------------------------

    st.subheader(
        "Response Log"
    )

    if st.session_state.response_log:

        log_df = pd.DataFrame(
            st.session_state.response_log
        )

        # Format timestamps
        log_df["Timestamp"] = log_df["Timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Show most recent actions first
        log_df = log_df.iloc[::-1]

        st.dataframe(
            log_df,
            use_container_width=True
        )

        if st.button(
            "Clear Response Log",
            key="clear_response_log"
        ):

            st.session_state.response_log.clear()

            st.success(
                "Response log cleared"
            )

    else:

        st.info(
            "No response actions logged yet. "
            "Enable automated responses or manually block/quarantine threats."
        )


# ============================================================
# FEDERATED LEARNING HUB
# ============================================================

def render_federated_learning_hub():

    st.header(
        "🌐 Federated Learning Hub"
    )

    st.info(
        "Federated learning monitoring interface."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Federated Nodes",
            "3"
        )

    with col2:

        st.metric(
            "Global Model",
            "Synchronized"
        )

    with col3:

        st.metric(
            "Training Status",
            "Ready"
        )

    st.subheader(
        "Federated Node Status"
    )

    node_df = pd.DataFrame({

        "Node": [
            "HQ Datacenter",
            "Branch Office",
            "Remote Node"
        ],

        "Status": [
            "Online",
            "Online",
            "Online"
        ],

        "Model Status": [
            "Synchronized",
            "Synchronized",
            "Synchronized"
        ]
    })

    st.dataframe(
        node_df,
        use_container_width=True
    )


# ============================================================
# RED TEAM TESTING RESULTS
# ============================================================

def render_red_team_results_tab():

    st.header(
        "⚡ Red Team Testing Results"
    )

    st.info(
        "View results from simulated threat vector injections. "
        "Use the sidebar to simulate attacks from the UNSW-NB15 dataset."
    )

    # --------------------------------------------------------
    # Check if results exist
    # --------------------------------------------------------

    if "red_team_results" not in st.session_state:

        st.warning(
            "No red team test results available. "
            "Use the sidebar to simulate a threat vector."
        )
        return

    attack_df = st.session_state.red_team_results
    detection_rate = st.session_state.red_team_detection_rate

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Samples",
            len(attack_df)
        )

    with col2:

        detected_count = attack_df["model_detected"].sum()
        st.metric(
            "Detected",
            detected_count,
            delta=f"{detection_rate:.1%}",
            delta_color="normal"
        )

    with col3:

        missed_count = len(attack_df) - detected_count
        st.metric(
            "Missed",
            missed_count,
            delta=f"{1 - detection_rate:.1%}",
            delta_color="inverse"
        )

    with col4:

        avg_risk = attack_df["risk_score"].mean()
        st.metric(
            "Avg Risk Score",
            f"{avg_risk:.2f}"
        )

    # --------------------------------------------------------
    # Detection rate gauge
    # --------------------------------------------------------

    st.subheader(
        "Detection Rate"
    )

    detection_gauge = create_risk_gauge(
        detection_rate
    )

    st.plotly_chart(
        detection_gauge,
        use_container_width=True,
        key="redteam_detection_gauge"
    )

    # --------------------------------------------------------
    # Attack samples table
    # --------------------------------------------------------

    st.subheader(
        "Injected Attack Samples"
    )

    def highlight_detection(row):

        if row["model_detected"]:

            return [
                "background-color: rgba(75, 255, 107, 0.15)"
            ] * len(row)

        else:

            return [
                "background-color: rgba(255, 75, 75, 0.15)"
            ] * len(row)

    display_df = (
        attack_df[
            [
                "timestamp",
                "event_id",
                "attack_type",
                "risk_score",
                "attack_probability",
                "reconstruction_error",
                "model_detected"
            ]
        ]
        .copy()
    )

    styled_df = (
        display_df
        .style
        .apply(
            highlight_detection,
            axis=1
        )
    )

    st.dataframe(
        styled_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Detection breakdown
    # --------------------------------------------------------

    st.subheader(
        "Detection Breakdown"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**By Autoencoder (Anomaly Detection):**"
        )

        ae_detected = attack_df["ae_anomaly"].sum()
        ae_missed = len(attack_df) - ae_detected

        st.metric(
            "Detected",
            ae_detected,
            delta=f"{ae_detected / len(attack_df):.1%}"
        )

    with col2:

        st.write(
            "**By Temporal Classifier (Attack Classification):**"
        )

        tc_detected = (attack_df["model_prediction"] == 1).sum()
        tc_missed = len(attack_df) - tc_detected

        st.metric(
            "Detected",
            tc_detected,
            delta=f"{tc_detected / len(attack_df):.1%}"
        )

    # --------------------------------------------------------
    # Risk score distribution
    # --------------------------------------------------------

    st.subheader(
        "Risk Score Distribution"
    )

    fig = px.histogram(
        attack_df,
        x="risk_score",
        nbins=20,
        title="Risk Score Distribution of Injected Attacks",
        color="model_detected",
        color_discrete_map={
            True: "rgba(75, 255, 107, 0.8)",
            False: "rgba(255, 75, 75, 0.8)"
        },
        labels={
            "risk_score": "Risk Score",
            "model_detected": "Detected",
            "count": "Count"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#64748b"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="redteam_risk_distribution"
    )

    # --------------------------------------------------------
    # Clear results button
    # --------------------------------------------------------

    if st.button(
        "Clear Results",
        key="clear_redteam_results"
    ):

        del st.session_state.red_team_results
        del st.session_state.red_team_detection_rate

        st.success(
            "Red team results cleared."
        )
        st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    with st.spinner(
        "Loading trained models..."
    ):

        (
            autoencoder,
            temporal_classifier,
            pipeline
        ) = load_models()

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    st.sidebar.markdown(
        "### 🛡️ Hybrid AI Cyber Defense & IDS Dashboard"
    )

    st.sidebar.markdown("---")

    # --------------------------------------------------------
    # Node scope
    # --------------------------------------------------------

    node_scope = st.sidebar.selectbox(
        "Node Scope",
        [
            "HQ Datacenter",
            "Branch Office",
            "All Federated Nodes"
        ]
    )

    st.sidebar.markdown("---")

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    st.sidebar.markdown(
        "### 🧭 Navigation"
    )

    page = st.sidebar.radio(
        "Select View",
        [
            "Real-Time Monitoring",
            "Threat Analytics",
            "Federated Learning Hub",
            "Incident Response & Logs",
            "Red Team Testing Results"
        ]
    )

    st.sidebar.markdown("---")

    # --------------------------------------------------------
    # Stream ingestion
    # --------------------------------------------------------

    st.sidebar.markdown(
        "### 📡 Stream Ingestion Engine"
    )

    telemetry_source = st.sidebar.selectbox(
        "Telemetry Source",
        [
            "Live Synthetic Stream",
            "Dataset Replay (UNSW-NB15)",
            "Dataset Replay (CICIDS2017)"
        ]
    )

    st.session_state.telemetry_source = (
        telemetry_source
    )

    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=1,
        max_value=10,
        value=config.DASHBOARD_CONFIG[
            "refresh_interval"
        ]
    )

    st.sidebar.markdown("---")

    # --------------------------------------------------------
    # AI sensitivity
    # --------------------------------------------------------

    st.sidebar.markdown(
        "### 🧠 AI Sensitivity & Health"
    )

    mse_threshold = st.sidebar.slider(
        "Autoencoder MSE Threshold",
        min_value=0.01,
        max_value=0.50,
        value=0.05,
        step=0.01
    )

    st.sidebar.markdown(
        "**Model Status:**"
    )

    if autoencoder is not None:

        st.sidebar.markdown(
            "🟢 Autoencoder"
        )

    else:

        st.sidebar.markdown(
            "🔴 Autoencoder"
        )

    if temporal_classifier is not None:

        st.sidebar.markdown(
            "🟢 Temporal Classifier"
        )

    else:

        st.sidebar.markdown(
            "🔴 Temporal Classifier"
        )

    st.sidebar.markdown(
        "🟢 Federated Sync"
    )

    st.sidebar.markdown("---")

    # --------------------------------------------------------
    # Red Team Testing
    # --------------------------------------------------------

    st.sidebar.markdown(
        "### ⚡ Red Team Testing"
    )

    attack_type = st.sidebar.selectbox(
        "Attack Type",
        [
            "DoS",
            "Reconnaissance",
            "Exploits",
            "Exfiltration"
        ],
        help="Select the type of attack to simulate from UNSW-NB15 dataset"
    )

    n_attack_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of attack samples to inject"
    )

    simulate_threat = st.sidebar.button(
        "Simulate Threat Vector"
    )

    if simulate_threat:

        if (
            telemetry_source
            == "Dataset Replay (UNSW-NB15)"
            and pipeline is not None
        ):

            with st.spinner(
                f"Injecting {attack_type} attack samples..."
            ):

                attack_df, detection_rate = (
                    simulate_red_team_attack(
                        pipeline,
                        autoencoder,
                        temporal_classifier,
                        attack_type,
                        n_attack_samples,
                        mse_threshold
                    )
                )

                if len(attack_df) > 0:

                    st.session_state.red_team_results = attack_df
                    st.session_state.red_team_detection_rate = detection_rate

                    st.sidebar.success(
                        f"{attack_type} attack injected! "
                        f"Detection rate: {detection_rate:.1%}"
                    )

                else:

                    st.sidebar.error(
                        "No attack samples available for simulation."
                    )

        else:

            st.sidebar.warning(
                "Red Team Testing requires UNSW-NB15 dataset replay mode."
            )

    st.sidebar.markdown("---")

    # --------------------------------------------------------
    # Current data source indicator
    # --------------------------------------------------------

    if (
        telemetry_source
        == "Dataset Replay (UNSW-NB15)"
    ):

        st.sidebar.success(
            "REAL UNSW-NB15 MODEL INFERENCE"
        )

    elif (
        telemetry_source
        == "Live Synthetic Stream"
    ):

        st.sidebar.info(
            "Synthetic traffic mode"
        )

    else:

        st.sidebar.warning(
            "CICIDS2017 replay is not connected "
            "to the trained UNSW models."
        )

    # --------------------------------------------------------
    # Render selected page
    # --------------------------------------------------------

    if page == "Real-Time Monitoring":

        render_real_time_monitoring_tab(
            autoencoder,
            temporal_classifier,
            pipeline
        )

    elif page == "Threat Analytics":

        render_analytics_tab(
            autoencoder,
            temporal_classifier,
            pipeline
        )

    elif page == "Federated Learning Hub":

        render_federated_learning_hub()

    elif page == "Incident Response & Logs":

        render_incident_response_tab(
            autoencoder,
            temporal_classifier,
            pipeline
        )

    elif page == "Red Team Testing Results":

        render_red_team_results_tab()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()