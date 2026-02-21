"""
TDCNN + ECA Classifier

Time-Delay Convolutional Neural Network with Efficient Channel Attention.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Optional, Union
from .base_model import IModel, ModelState

class ECA(nn.Module):
    """
    Efficient Channel Attention Module.
    Ref: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    """
    def __init__(self, channels: int = None, k_size: int = 3, gamma: int = 2, b: int = 1):
        super().__init__()
        if channels is not None:
            t = int(abs((math.log2(channels) + b) / gamma))
            k_size = t if t % 2 else t + 1
            
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        y = self.avg_pool(x)  # (B, C, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # (B, C, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Chomp1d(nn.Module):
    """
    Removes the last elements of a time series.
    Used to ensure causal convolutions by removing the 'future' padding.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TDCNN(nn.Module):
    """
    Time-Delay CNN with ECA blocks (Strictly Causal Implementation).
    """
    def __init__(self, input_channels: int, num_classes: int, 
                 hidden_channels: List[int] = [64, 128], 
                 kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        layers = []
        in_c = input_channels
        dilation = 1
        
        for out_c in hidden_channels:
            # Causal Padding: (k-1) * d
            padding = (kernel_size - 1) * dilation
            
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, 
                                  padding=padding, dilation=dilation))
            layers.append(Chomp1d(padding)) # Remove future padding
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.LeakyReLU(0.1)) # LeakyReLU is often better for EMG
            layers.append(ECA(channels=out_c))
            layers.append(nn.Dropout(0.1)) # Light dropout in blocks
            
            in_c = out_c
            dilation *= 2 # Exponential dilation
            
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with stronger dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_c, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

class TDCNNClassifier(IModel):
    """
    TDCNN + ECA Classifier wrapper.
    """
    def __init__(self, input_channels: int = 8, num_classes: int = 7,
                 hidden_channels: Tuple[int, ...] = (64, 128),
                 kernel_size: int = 3, dropout: float = 0.5,
                 learning_rate: float = 0.001, batch_size: int = 32,
                 epochs: int = 10, device: str = 'auto', verbose: bool = True):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            device=device
        )
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_channels = list(hidden_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        
        self.device = self._select_device(device)
        
        self.model = TDCNN(input_channels, num_classes, self.hidden_channels, 
                           kernel_size, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        if self.verbose:
            print(f"  → TDCNN using device: {self.device}")
            if self.device.type == 'cuda':
                print(f"    GPU: {torch.cuda.get_device_name(0)}")
    
    def _select_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available. Use 'auto' or 'cpu'.")
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> DataLoader:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (samples, time, channels), got {X.shape}")
            
        if X.shape[2] == self.input_channels:
            X = X.transpose(0, 2, 1)

        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            y_tensor = torch.LongTensor(y).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            dataset = TensorDataset(X_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'TDCNNClassifier':
        """
        Trains the model. Optionally accepts validation data to track performance per epoch.
        """
        train_loader = self._prepare_data(X, y)
        val_loader = self._prepare_data(X_val, y_val) if (X_val is not None and y_val is not None) else None
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            
            if val_loader:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total
                
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            else:
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {avg_train_loss:.4f}")
            
        self._is_trained = True
        self._n_samples_seen += len(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_trained()
        self.model.eval()
        dataloader = self._prepare_data(X)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch_X = batch[0] 
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_trained()
        self.model.eval()
        dataloader = self._prepare_data(X)
        
        probabilities = []
        with torch.no_grad():
            for batch in dataloader:
                batch_X = batch[0]
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
                
        return np.array(probabilities)

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> 'TDCNNClassifier':
        self._check_is_trained()
        self.fit(X_new, y_new)
        self._n_updates += 1
        return self
    
    def save(self, filepath: str) -> None:
        from pathlib import Path
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            '_is_trained': self._is_trained,
            '_n_samples_seen': self._n_samples_seen,
            '_n_updates': self._n_updates
        }, filepath)
        print(f"✓ Saved TDCNN model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TDCNNClassifier':
        from pathlib import Path
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        model = cls(
            input_channels=checkpoint['input_channels'],
            num_classes=checkpoint['num_classes'],
            hidden_channels=checkpoint['hidden_channels'],
            kernel_size=checkpoint['kernel_size'],
            dropout=checkpoint['dropout'],
            learning_rate=checkpoint['learning_rate'],
            epochs=checkpoint['epochs'],
            batch_size=checkpoint['batch_size']
        )
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model._is_trained = checkpoint['_is_trained']
        model._n_samples_seen = checkpoint['_n_samples_seen']
        model._n_updates = checkpoint['_n_updates']
        
        print(f"✓ Loaded TDCNN model from {filepath}")
        return model