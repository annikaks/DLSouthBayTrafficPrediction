import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """
    Multi-output linear regression:
    flatten window + sensors + features -> predict speed at all sensors.
    """
    def __init__(self, window_size: int, num_nodes: int, num_features: int):
        super().__init__()
        in_dim = window_size * num_nodes * num_features
        out_dim = num_nodes
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, N, F)
        B, W, N, F = x.shape
        x_flat = x.view(B, W * N * F)
        return self.linear(x_flat)  # (B, N)


class LogisticRegressionModel(nn.Module):
    """
    Binary logistic regression per node (congested vs not).
    """
    def __init__(self, window_size: int, num_nodes: int, num_features: int):
        super().__init__()
        in_dim = window_size * num_nodes * num_features
        out_dim = num_nodes
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, N, F)
        B, W, N, F = x.shape
        x_flat = x.view(B, W * N * F)
        return self.linear(x_flat)  # compute w.tx + b


class MLPRegressor(nn.Module):
    """
    Simple deep MLP model (DL approach baseline).
    """
    def __init__(
        self,
        window_size: int,
        num_nodes: int,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = window_size * num_nodes * num_features
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_nodes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, N, F)
        B, W, N, F = x.shape
        x_flat = x.view(B, W * N * F)
        return self.net(x_flat)  # (B, N)


class LSTMRegressor(nn.Module):
    """
    LSTM-based temporal model:
    Uses past W timesteps to predict future speed at all sensors.
    """
    def __init__(
        self,
        window_size: int,
        num_nodes: int,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.window_size = window_size
        self.num_nodes = num_nodes
        self.num_features = num_features

        # Each timestep: flatten all sensors and features
        self.input_dim = num_nodes * num_features

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,   # x: (B, W, D)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map final hidden state -> per-sensor speeds
        self.fc = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, N, F)
        returns: (B, N)
        """
        B, W, N, F = x.shape

        # (B, W, N*F) sequence over time
        x_seq = x.view(B, W, N * F)

        # LSTM over time dimension
        lstm_out, (h_n, c_n) = self.lstm(x_seq)

        # Last layer's hidden state at final timestep: (B, hidden_dim)
        h_last = h_n[-1]

        # Predict speed for each sensor
        out = self.fc(h_last)  # (B, N)
        return out
