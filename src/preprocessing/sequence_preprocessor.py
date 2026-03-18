"""
Preprocessing utilities for robotics data sequences.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

class SequencePreprocessor:
    def __init__(self):
        self.scalers = {}

    def normalize_data(self, data: np.ndarray, feature_name: str):
        """
        Normalize data using StandardScaler.
        """
        if feature_name not in self.scalers:
            self.scalers[feature_name] = StandardScaler()

        scaler = self.scalers[feature_name]
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        normalized = scaler.fit_transform(data_flat)
        return normalized.reshape(original_shape)

    def create_sequences(self, data: np.ndarray, seq_length: int, step: int = 1):
        """
        Create sequences from time series data.
        """
        sequences = []
        for i in range(0, len(data) - seq_length + 1, step):
            seq = data[i:i + seq_length]
            sequences.append(seq)
        return np.array(sequences)

    def split_train_val_test(self, sequences: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Split sequences into train, validation, test sets.
        """
        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_data = sequences[:n_train]
        val_data = sequences[n_train:n_train + n_val]
        test_data = sequences[n_train + n_val:]

        return train_data, val_data, test_data