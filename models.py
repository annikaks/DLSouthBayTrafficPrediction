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
