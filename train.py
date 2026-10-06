"""
Training script for Hybrid AI Cyber Defense & Intrusion Detection System.

This script orchestrates the complete training pipeline:
1. Data loading and preprocessing
2. Autoencoder training (unsupervised anomaly detection)
3. Temporal classifier training (supervised attack classification)
4. Federated learning simulation (optional)
5. Model evaluation and metrics calculation
6. Model checkpointing
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, Optional

import config
from pipeline import DataPipeline
from models import (
    Autoencoder,
    TemporalClassifier,
    train_autoencoder,
    train_temporal_classifier,
    calculate_metrics,
    calculate_anomaly_threshold,
    detect_anomalies,
    save_model
)
from federated import run_federated_autoencoder, run_federated_temporal_classifier

# Configure logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def train_autoencoder_pipeline(data: Dict, use_federated: bool = False) -> Dict:
    """
    Train the autoencoder for anomaly detection.
    
    Args:
        data: Dictionary containing preprocessed data
        use_federated: Whether to use federated learning
        
    Returns:
        Dictionary containing training results and metrics
    """
    logger.info("=" * 80)
    logger.info("Starting Autoencoder Training Pipeline")
    logger.info("=" * 80)
    
    ae_config = config.AUTOENCODER_CONFIG
    
    # Initialize autoencoder
    autoencoder = Autoencoder(
        input_dim=ae_config['input_dim'],
        hidden_dims=ae_config['hidden_dims'],
        latent_dim=ae_config['latent_dim'],
        activation=ae_config['activation'],
        dropout=ae_config['dropout'],
        batch_norm=ae_config['batch_norm']
    ).to(config.DEVICE)
    
    # Training data (benign only for autoencoder)
    X_train_benign = data['X_train_benign'].values
    X_test = data['X_test'].values
    
    # Split train into train/val
    val_split = int(0.8 * len(X_train_benign))
    X_train = X_train_benign[:val_split]
    X_val = X_train_benign[val_split:]
    
    if use_federated:
        logger.info("Using Federated Learning for Autoencoder")
        trained_model, round_history = run_federated_autoencoder(
            X_train_benign,
            X_test,
            config=config.FEDERATED_CONFIG
        )
        train_losses = [round['client_metrics'][0]['loss'] for round in round_history]
        val_losses = train_losses  # Simplified for federated
    else:
        logger.info("Using Centralized Training for Autoencoder")
        trained_model, train_losses, val_losses = train_autoencoder(
            autoencoder,
            X_train,
            X_val,
            ae_config
        )
    
    # Calculate anomaly threshold
    threshold = calculate_anomaly_threshold(
        trained_model,
        X_train_benign,
        percentile=ae_config['anomaly_threshold_percentile']
    )
    
    # Detect anomalies on test set
    anomalies, errors = detect_anomalies(trained_model, X_test, threshold)
    
    # Calculate metrics (compare with true labels)
    y_test = data['y_test'].values
    anomaly_labels = (y_test != 0).astype(int)  # Non-benign = anomaly
    
    metrics = calculate_metrics(anomaly_labels, anomalies.astype(int))
    
    # Save model
    model_path = config.CHECKPOINT_DIR / "autoencoder.pth"
    save_model(trained_model, model_path)
    
    results = {
        'model_type': 'autoencoder',
        'train_losses': train_losses,
        'val_losses': val_losses,
        'anomaly_threshold': threshold,
        'num_anomalies_detected': int(anomalies.sum()),
        'metrics': metrics,
        'model_path': str(model_path)
    }
    
    if use_federated:
        results['round_history'] = round_history
    
    logger.info(f"Autoencoder training completed. Metrics: {metrics}")
    
    return results


def train_temporal_classifier_pipeline(data: Dict, use_federated: bool = False) -> Dict:
    """
    Train the temporal classifier for attack classification.
    
    Args:
        data: Dictionary containing preprocessed data
        use_federated: Whether to use federated learning
        
    Returns:
        Dictionary containing training results and metrics
    """
    logger.info("=" * 80)
    logger.info("Starting Temporal Classifier Training Pipeline")
    logger.info("=" * 80)
    
    tc_config = config.TEMPORAL_CONFIG
    
    # Initialize temporal classifier
    temporal_classifier = TemporalClassifier(
        input_dim=tc_config['input_dim'],
        hidden_dim=tc_config['hidden_dim'],
        num_layers=tc_config['num_layers'],
        num_classes=tc_config['num_classes'],
        dropout=tc_config['dropout'],
        model_type=tc_config['model_type'],
        bidirectional=tc_config['bidirectional']
    ).to(config.DEVICE)
    
    # Training data
    X_train_seq = data['X_train_seq']
    y_train_seq = data['y_train_seq']
    X_test_seq = data['X_test_seq']
    y_test_seq = data['y_test_seq']
    
    # Split train into train/val
    val_split = int(0.8 * len(X_train_seq))
    X_train = X_train_seq[:val_split]
    y_train = y_train_seq[:val_split]
    X_val = X_train_seq[val_split:]
    y_val = y_train_seq[val_split:]
    
    if use_federated:
        logger.info("Using Federated Learning for Temporal Classifier")
        trained_model, round_history = run_federated_temporal_classifier(
            X_train,
            y_train,
            X_test_seq,
            y_test_seq,
            config=config.FEDERATED_CONFIG
        )
        train_losses = [round['client_metrics'][0]['loss'] for round in round_history]
        val_losses = train_losses  # Simplified for federated
    else:
        logger.info("Using Centralized Training for Temporal Classifier")
        trained_model, train_losses, val_losses = train_temporal_classifier(
            temporal_classifier,
            X_train,
            y_train,
            X_val,
            y_val,
            tc_config
        )
    
    # Evaluate on test set
    trained_model.eval()
    import torch
    X_test_tensor = torch.FloatTensor(X_test_seq).to(config.DEVICE)
    
    with torch.no_grad():
        logits = trained_model(X_test_tensor)
        _, predictions = torch.max(logits, 1)
        predictions = predictions.cpu().numpy()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_seq, predictions, probabilities)
    
    # Save model
    model_path = config.CHECKPOINT_DIR / "temporal_classifier.pth"
    save_model(trained_model, model_path)
    
    results = {
        'model_type': 'temporal_classifier',
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics,
        'model_path': str(model_path)
    }
    
    if use_federated:
        results['round_history'] = round_history
    
    logger.info(f"Temporal classifier training completed. Metrics: {metrics}")
    
    return results


def save_results(results: Dict, output_path: Path):
    """
    Save training results to JSON file.
    
    Args:
        results: Dictionary of training results
        output_path: Path to save results
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_serializable = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def print_summary(results: Dict):
    """
    Print a summary of training results.
    
    Args:
        results: Dictionary of training results
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    
    if 'autoencoder' in results:
        ae_results = results['autoencoder']
        logger.info(f"\nAutoencoder Results:")
        logger.info(f"  - Model saved to: {ae_results['model_path']}")
        logger.info(f"  - Anomaly threshold: {ae_results['anomaly_threshold']:.6f}")
        logger.info(f"  - Anomalies detected: {ae_results['num_anomalies_detected']}")
        logger.info(f"  - Precision: {ae_results['metrics']['precision']:.4f}")
        logger.info(f"  - Recall: {ae_results['metrics']['recall']:.4f}")
        logger.info(f"  - F1-Score: {ae_results['metrics']['f1_score']:.4f}")
        logger.info(f"  - Accuracy: {ae_results['metrics']['accuracy']:.4f}")
    
    if 'temporal_classifier' in results:
        tc_results = results['temporal_classifier']
        logger.info(f"\nTemporal Classifier Results:")
        logger.info(f"  - Model saved to: {tc_results['model_path']}")
        logger.info(f"  - Precision: {tc_results['metrics']['precision']:.4f}")
        logger.info(f"  - Recall: {tc_results['metrics']['recall']:.4f}")
        logger.info(f"  - F1-Score: {tc_results['metrics']['f1_score']:.4f}")
        logger.info(f"  - Accuracy: {tc_results['metrics']['accuracy']:.4f}")
        if 'roc_auc' in tc_results['metrics']:
            logger.info(f"  - ROC-AUC: {tc_results['metrics']['roc_auc']:.4f}")
    
    logger.info("\n" + "=" * 80)


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train Hybrid AI Cyber Defense & IDS')
    parser.add_argument(
        '--model',
        type=str,
        choices=['autoencoder', 'temporal', 'both'],
        default='both',
        help='Which model to train'
    )
    parser.add_argument(
        '--federated',
        action='store_true',
        help='Use federated learning'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to dataset (uses synthetic if not provided)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=config.LOG_DIR / "training_results.json",
        help='Path to save training results'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting training pipeline")
    logger.info(f"Model: {args.model}")
    logger.info(f"Federated Learning: {args.federated}")
    logger.info(f"Device: {config.DEVICE}")
    
    # Load and preprocess data
    pipeline = DataPipeline(data_path=Path(args.data_path) if args.data_path else None)
    data = pipeline.prepare_full_pipeline(create_sequences=True)
    
    results = {}
    
    # Train autoencoder
    if args.model in ['autoencoder', 'both']:
        ae_results = train_autoencoder_pipeline(data, use_federated=args.federated)
        results['autoencoder'] = ae_results
    
    # Train temporal classifier
    if args.model in ['temporal', 'both']:
        tc_results = train_temporal_classifier_pipeline(data, use_federated=args.federated)
        results['temporal_classifier'] = tc_results
    
    # Save results
    output_path = Path(args.output)
    save_results(results, output_path)
    
    # Print summary
    print_summary(results)
    
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
