"""
PyTorch Models for Hybrid AI Cyber Defense & Intrusion Detection System.

This module contains:
- Autoencoder: Unsupervised spatial anomaly detector
- TemporalClassifier: LSTM/Transformer-based attack classifier
- Training utilities and evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
from pathlib import Path

import config

# Configure logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    Deep Autoencoder for unsupervised anomaly detection.
    
    Trains exclusively on benign traffic to learn normal patterns.
    Reconstruction error (MSE) is used to detect anomalies and zero-day threats.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, 
                 activation: str = "relu", dropout: float = 0.2, batch_norm: bool = True):
        """
        Initialize the Autoencoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions for encoder
            latent_dim: Dimension of bottleneck layer
            activation: Activation function ('relu', 'leaky_relu', 'tanh')
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(self._get_activation(activation))
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Bottleneck layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(self._get_activation(activation))
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"Autoencoder initialized with input_dim={input_dim}, latent_dim={latent_dim}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        return self.encoder(x)
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error (MSE) for each sample.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstruction error per sample
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error


class TemporalClassifier(nn.Module):
    """
    Temporal Attack Classifier using LSTM or Transformer architecture.
    
    Processes time-series sequences of log events to classify multi-stage attack types:
    - DoS (Denial of Service)
    - Reconnaissance
    - Exploits
    - Exfiltration
    - Benign
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        model_type: str = "lstm",
        bidirectional: bool = True,
        **kwargs
    ):
        """
        Initialize the Temporal Classifier.
        
        Args:
            input_dim: Dimension of input features per time step
            hidden_dim: Hidden dimension of LSTM/Transformer
            num_layers: Number of recurrent/transformer layers
            num_classes: Number of attack classes
            dropout: Dropout rate
            model_type: 'lstm' or 'transformer'
            bidirectional: Whether to use bidirectional LSTM
            **kwargs: Additional arguments for Transformer
        """
        super(TemporalClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.model_type = model_type
        self.bidirectional = bidirectional
        
        if model_type == "lstm":
            self.model = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=kwargs.get('num_heads', 8),
                dim_feedforward=kwargs.get('dim_feedforward', 256),
                dropout=dropout,
                batch_first=True
            )
            self.model = nn.TransformerEncoder(
                encoder_layer,
                num_layers=kwargs.get('num_transformer_layers', 2)
            )
            lstm_output_dim = input_dim
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        logger.info(f"TemporalClassifier initialized: {model_type}, hidden_dim={hidden_dim}, num_classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal classifier.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        if self.model_type == "lstm":
            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.model(x)
            
            # Use the last hidden state
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                last_hidden = h_n[-1]
            
            out = self.classifier(last_hidden)
        else:
            # Transformer forward pass
            transformer_out = self.model(x)
            # Use mean pooling over sequence
            pooled = transformer_out.mean(dim=1)
            out = self.classifier(pooled)
        
        return out


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def train_autoencoder(
    model: Autoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: Dict,
    device: torch.device = config.DEVICE
) -> Tuple[Autoencoder, List[float], List[float]]:
    """
    Train the autoencoder on benign traffic data.
    
    Args:
        model: Autoencoder model
        X_train: Training data
        X_val: Validation data
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Tuple of (trained model, train_losses, val_losses)
    """
    logger.info("Starting autoencoder training")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_dataset = TensorDataset(X_val_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, train_losses, val_losses


def train_temporal_classifier(
    model: TemporalClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    device: torch.device = config.DEVICE
) -> Tuple[TemporalClassifier, List[float], List[float]]:
    """
    Train the temporal classifier.
    
    Args:
        model: TemporalClassifier model
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Tuple of (trained model, train_losses, val_losses)
    """
    logger.info("Starting temporal classifier training")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, train_losses, val_losses


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'accuracy': (y_true == y_pred).mean()
    }
    
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def save_model(model: nn.Module, path: Path):
    """
    Save model to disk.
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.__dict__
    }, path)
    logger.info(f"Model saved to {path}")


def load_model(model: nn.Module, path: Path, device: torch.device = config.DEVICE) -> nn.Module:
    """
    Load model from disk.
    
    Args:
        model: Model instance
        path: Path to load the model from
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logger.info(f"Model loaded from {path}")
    return model


def calculate_anomaly_threshold(model: Autoencoder, X_benign: np.ndarray, percentile: float = 95) -> float:
    """
    Calculate dynamic threshold for anomaly detection based on benign data.
    
    Args:
        model: Trained autoencoder
        X_benign: Benign data
        percentile: Percentile for threshold
        
    Returns:
        Threshold value
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_benign).to(config.DEVICE)
    
    with torch.no_grad():
        errors = model.reconstruction_error(X_tensor).cpu().numpy()
    
    threshold = np.percentile(errors, percentile)
    logger.info(f"Anomaly threshold (percentile {percentile}): {threshold:.6f}")
    
    return threshold


def detect_anomalies(model: Autoencoder, X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Detect anomalies using reconstruction error threshold.
    
    Args:
        model: Trained autoencoder
        X: Data to evaluate
        threshold: Anomaly threshold
        
    Returns:
        Boolean array indicating anomalies
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(config.DEVICE)
    
    with torch.no_grad():
        errors = model.reconstruction_error(X_tensor).cpu().numpy()
    
    anomalies = errors > threshold
    logger.info(f"Detected {anomalies.sum()} anomalies out of {len(anomalies)} samples")
    
    return anomalies, errors


def main():
    """
    Main function to test the models.
    """
    # Test Autoencoder
    logger.info("Testing Autoencoder")
    ae_config = config.AUTOENCODER_CONFIG
    autoencoder = Autoencoder(
        input_dim=ae_config['input_dim'],
        hidden_dims=ae_config['hidden_dims'],
        latent_dim=ae_config['latent_dim'],
        activation=ae_config['activation'],
        dropout=ae_config['dropout']
    ).to(config.DEVICE)
    
    # Test Temporal Classifier
    logger.info("Testing Temporal Classifier")
    tc_config = config.TEMPORAL_CONFIG
    temporal_classifier = TemporalClassifier(
        input_dim=tc_config['input_dim'],
        hidden_dim=tc_config['hidden_dim'],
        num_layers=tc_config['num_layers'],
        num_classes=tc_config['num_classes'],
        dropout=tc_config['dropout'],
        model_type=tc_config['model_type'],
        bidirectional=tc_config['bidirectional']
    ).to(config.DEVICE)
    
    # Test forward pass
    batch_size = 32
    seq_len = config.WINDOW_SIZE
    input_dim = ae_config['input_dim']
    
    # Autoencoder test
    x_ae = torch.randn(batch_size, input_dim).to(config.DEVICE)
    reconstructed = autoencoder(x_ae)
    logger.info(f"Autoencoder input shape: {x_ae.shape}, output shape: {reconstructed.shape}")
    
    # Temporal Classifier test
    x_tc = torch.randn(batch_size, seq_len, input_dim).to(config.DEVICE)
    logits = temporal_classifier(x_tc)
    logger.info(f"TemporalClassifier input shape: {x_tc.shape}, output shape: {logits.shape}")
    
    logger.info("Model tests completed successfully")


if __name__ == "__main__":
    main()
