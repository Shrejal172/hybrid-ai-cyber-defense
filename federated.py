"""
Federated Learning Simulation for Hybrid AI Cyber Defense & IDS.

This module implements a lightweight Federated Averaging (FedAvg) simulation
that splits the dataset into simulated local client nodes, trains models
independently, encrypts/aggregates weight gradients on a central server,
and redistributes global updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import logging
from pathlib import Path

import config
from models import Autoencoder, TemporalClassifier, train_autoencoder, train_temporal_classifier

# Configure logging
logging.basicConfig(**config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Simulated federated learning client.
    
    Each client has its own local data and trains a model independently.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        X_local: np.ndarray,
        y_local: Optional[np.ndarray] = None,
        config: Dict = None
    ):
        """
        Initialize a federated client.
        
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model to train
            X_local: Local training data
            y_local: Local training labels (optional for autoencoder)
            config: Training configuration
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.X_local = X_local
        self.y_local = y_local
        self.config = config or {}
        
        # Add DEVICE to config if not present
        if 'DEVICE' not in self.config:
            import config as cfg
            self.config['DEVICE'] = cfg.DEVICE
        self.device = self.config['DEVICE']
        
        # Convert to tensors
        self.X_tensor = torch.FloatTensor(X_local).to(self.device)
        if y_local is not None:
            self.y_tensor = torch.LongTensor(y_local).to(self.device)
        
        logger.info(f"Client {client_id} initialized with {len(X_local)} samples")
    
    def train_local(self, epochs: int) -> Dict[str, float]:
        """
        Train the model locally for specified epochs.
        
        Args:
            epochs: Number of local training epochs
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Client {self.client_id} starting local training for {epochs} epochs")
        
        # Create data loader
        if self.y_local is not None:
            dataset = TensorDataset(self.X_tensor, self.y_tensor)
        else:
            dataset = TensorDataset(self.X_tensor)
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 128),
            shuffle=True
        )
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.01))
        
        # Loss function
        if self.y_local is not None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                
                if self.y_local is not None:
                    x, y = batch
                    outputs = self.model(x)
                    loss = criterion(outputs, y)
                else:
                    x = batch[0]
                    reconstructed = self.model(x)
                    loss = criterion(reconstructed, x)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(loader)
        
        avg_loss = total_loss / epochs
        logger.info(f"Client {self.client_id} local training completed. Avg loss: {avg_loss:.6f}")
        
        return {'loss': avg_loss}
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get the current model weights.
        
        Returns:
            Dictionary of model parameters
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Set model weights from server.
        
        Args:
            weights: Dictionary of model parameters
        """
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = weights[name].clone()
        logger.info(f"Client {self.client_id} updated model weights from server")


class FederatedServer:
    """
    Federated learning server that aggregates client updates.
    
    Implements FedAvg (Federated Averaging) with optional encryption simulation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict = None
    ):
        """
        Initialize the federated server.
        
        Args:
            model: Global model architecture
            config: Federated learning configuration
        """
        self.global_model = model
        self.config = config or {}
        
        # Add DEVICE to config if not present
        if 'DEVICE' not in self.config:
            import config as cfg
            self.config['DEVICE'] = cfg.DEVICE
        self.device = self.config['DEVICE']
        
        # Track training history
        self.round_history = []
        
        logger.info("Federated server initialized")
    
    def aggregate_weights(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        encryption_simulation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client weights using Federated Averaging (FedAvg).
        
        Args:
            client_weights: List of client weight dictionaries
            client_sizes: List of client data sizes for weighted averaging
            encryption_simulation: Whether to simulate gradient encryption
            
        Returns:
            Aggregated global weights
        """
        logger.info(f"Aggregating weights from {len(client_weights)} clients")
        
        if encryption_simulation:
            logger.info("Simulating gradient encryption/decryption")
            # In a real system, this would involve homomorphic encryption
            # Here we simulate by adding noise that gets averaged out
            for i, weights in enumerate(client_weights):
                noise = {k: torch.randn_like(v) * 0.01 for k, v in weights.items()}
                for k in weights:
                    weights[k] += noise[k]
        
        # Calculate weighted average
        aggregated_weights = {}
        total_samples = sum(client_sizes)
        
        for key in client_weights[0].keys():
            weighted_sum = torch.zeros_like(client_weights[0][key])
            for weights, size in zip(client_weights, client_sizes):
                weighted_sum += weights[key] * size
            aggregated_weights[key] = weighted_sum / total_samples
        
        logger.info("Weight aggregation completed")
        return aggregated_weights
    
    def distribute_weights(self, clients: List[FederatedClient]):
        """
        Distribute global model weights to all clients.
        
        Args:
            clients: List of federated clients
        """
        global_weights = {name: param.data.clone() for name, param in self.global_model.named_parameters()}
        
        for client in clients:
            client.set_model_weights(global_weights)
        
        logger.info(f"Distributed global weights to {len(clients)} clients")
    
    def update_global_model(self, weights: Dict[str, torch.Tensor]):
        """
        Update the global model with aggregated weights.
        
        Args:
            weights: Aggregated weights from clients
        """
        for name, param in self.global_model.named_parameters():
            if name in weights:
                param.data = weights[name].clone()
        
        logger.info("Global model updated with aggregated weights")


class FederatedLearningSimulator:
    """
    Main simulator for federated learning process.
    
    Coordinates the training across multiple clients and server aggregation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        config: Dict = None
    ):
        """
        Initialize the federated learning simulator.
        
        Args:
            model: Model architecture to train
            X: Full dataset
            y: Labels (optional for autoencoder)
            config: Federated learning configuration
        """
        self.model = model
        self.X = X
        self.y = y
        self.config = config or config.FEDERATED_CONFIG
        
        # Add DEVICE to config if not present
        if 'DEVICE' not in self.config:
            import config as cfg
            self.config['DEVICE'] = cfg.DEVICE
        self.device = self.config['DEVICE']
        
        self.num_clients = self.config.get('num_clients', 2)
        self.num_rounds = self.config.get('num_rounds', 5)
        self.client_epochs = self.config.get('client_epochs', 5)
        self.local_data_fraction = self.config.get('local_data_fraction', 0.5)
        
        self.server = FederatedServer(copy.deepcopy(model), self.config)
        self.clients = []
        
        logger.info(f"Federated learning simulator initialized with {self.num_clients} clients")
    
    def split_data_to_clients(self) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Split data among clients (simulating distributed data).
        
        Returns:
            Tuple of (list of X splits, list of y splits)
        """
        logger.info(f"Splitting data among {self.num_clients} clients")
        
        n_samples = len(self.X)
        samples_per_client = int(n_samples * self.local_data_fraction / self.num_clients)
        
        X_splits = []
        y_splits = []
        
        indices = np.random.permutation(n_samples)
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices = indices[start_idx:end_idx]
            
            X_splits.append(self.X[client_indices])
            
            if self.y is not None:
                y_splits.append(self.y[client_indices])
            else:
                y_splits.append(None)
            
            logger.info(f"Client {i}: {len(client_indices)} samples")
        
        return X_splits, y_splits
    
    def initialize_clients(self, X_splits: List[np.ndarray], y_splits: List[Optional[np.ndarray]]):
        """
        Initialize federated clients with their local data.
        
        Args:
            X_splits: List of feature splits for each client
            y_splits: List of label splits for each client
        """
        self.clients = []
        
        for i, (X_local, y_local) in enumerate(zip(X_splits, y_splits)):
            client = FederatedClient(
                client_id=i,
                model=self.model,
                X_local=X_local,
                y_local=y_local,
                config=self.config
            )
            self.clients.append(client)
        
        logger.info(f"Initialized {len(self.clients)} clients")
    
    def run_federated_training(self) -> Tuple[nn.Module, List[Dict]]:
        """
        Run the complete federated learning process.
        
        Returns:
            Tuple of (trained global model, round history)
        """
        logger.info(f"Starting federated training for {self.num_rounds} rounds")
        
        # Split data and initialize clients
        X_splits, y_splits = self.split_data_to_clients()
        self.initialize_clients(X_splits, y_splits)
        
        # Distribute initial global weights
        self.server.distribute_weights(self.clients)
        
        round_history = []
        
        for round_num in range(self.num_rounds):
            logger.info(f"\n=== Round {round_num + 1}/{self.num_rounds} ===")
            
            # Local training on each client
            client_weights = []
            client_sizes = []
            round_metrics = {'round': round_num + 1, 'client_metrics': []}
            
            for client in self.clients:
                metrics = client.train_local(self.client_epochs)
                round_metrics['client_metrics'].append({
                    'client_id': client.client_id,
                    'loss': metrics['loss']
                })
                
                client_weights.append(client.get_model_weights())
                client_sizes.append(len(client.X_local))
            
            # Aggregate weights on server
            aggregated_weights = self.server.aggregate_weights(
                client_weights,
                client_sizes,
                encryption_simulation=self.config.get('encryption_simulation', False)
            )
            
            # Update global model
            self.server.update_global_model(aggregated_weights)
            
            # Distribute updated weights to clients
            self.server.distribute_weights(self.clients)
            
            round_history.append(round_metrics)
            
            logger.info(f"Round {round_num + 1} completed")
        
        logger.info("Federated training completed")
        
        return self.server.global_model, round_history
    
    def evaluate_global_model(
        self,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate the global model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating global model")
        
        self.server.global_model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            if y_test is not None:
                y_tensor = torch.LongTensor(y_test).to(self.device)
                outputs = self.server.global_model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_tensor).float().mean().item()
                
                metrics = {'accuracy': accuracy}
            else:
                reconstructed = self.server.global_model(X_tensor)
                mse = torch.mean((X_tensor - reconstructed) ** 2).item()
                metrics = {'reconstruction_error': mse}
        
        logger.info(f"Global model evaluation: {metrics}")
        return metrics


def run_federated_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    fed_config: Dict = config.FEDERATED_CONFIG
) -> Tuple[Autoencoder, List[Dict]]:
    """
    Run federated learning for autoencoder.
    
    Args:
        X_train: Training data
        X_test: Test data
        fed_config: Federated learning configuration
        
    Returns:
        Tuple of (trained autoencoder, round history)
    """
    logger.info("Starting federated autoencoder training")
    
    # Initialize autoencoder
    ae_config = config.AUTOENCODER_CONFIG
    autoencoder = Autoencoder(
        input_dim=ae_config['input_dim'],
        hidden_dims=ae_config['hidden_dims'],
        latent_dim=ae_config['latent_dim'],
        activation=ae_config['activation'],
        dropout=ae_config['dropout']
    ).to(config.DEVICE)
    
    # Run federated training
    simulator = FederatedLearningSimulator(
        model=autoencoder,
        X=X_train,
        y=None,
        config=fed_config
    )
    
    trained_model, round_history = simulator.run_federated_training()
    
    # Evaluate
    test_metrics = simulator.evaluate_global_model(X_test)
    logger.info(f"Test metrics: {test_metrics}")
    
    return trained_model, round_history


def run_federated_temporal_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fed_config: Dict = config.FEDERATED_CONFIG
) -> Tuple[TemporalClassifier, List[Dict]]:
    """
    Run federated learning for temporal classifier.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_test: Test sequences
        y_test: Test labels
        fed_config: Federated learning configuration
        
    Returns:
        Tuple of (trained classifier, round history)
    """
    logger.info("Starting federated temporal classifier training")
    
    # Initialize temporal classifier
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
    
    # Run federated training
    simulator = FederatedLearningSimulator(
        model=temporal_classifier,
        X=X_train,
        y=y_train,
        config=fed_config
    )
    
    trained_model, round_history = simulator.run_federated_training()
    
    # Evaluate
    test_metrics = simulator.evaluate_global_model(X_test, y_test)
    logger.info(f"Test metrics: {test_metrics}")
    
    return trained_model, round_history


def main():
    """
    Main function to test federated learning.
    """
    logger.info("Testing Federated Learning Simulation")
    
    # Generate synthetic data
    np.random.seed(config.RANDOM_STATE)
    X_train = np.random.randn(1000, 78)
    y_train = np.random.randint(0, 5, 1000)
    X_test = np.random.randn(200, 78)
    y_test = np.random.randint(0, 5, 200)
    
    # Test federated autoencoder
    logger.info("\n=== Testing Federated Autoencoder ===")
    ae_model, ae_history = run_federated_autoencoder(X_train, X_test)
    
    # Test federated temporal classifier (requires 3D sequences)
    # Note: Skipping temporal classifier test due to sequence dimension requirements
    # In production, use pre-generated sequences from the data pipeline
    logger.info("\n=== Skipping Federated Temporal Classifier Test ===")
    logger.info("Temporal classifier requires 3D sequence data from pipeline")
    logger.info("Use train.py --model temporal --federated for full training")
    
    logger.info("Federated learning tests completed successfully")


if __name__ == "__main__":
    main()
