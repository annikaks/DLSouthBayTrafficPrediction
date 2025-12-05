import torch
from torch import nn
from torch_geometric.nn import GCNConv



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



#gnn : gcn -> gcn -> lstm -> linear 
class GNN_LSTM_Regressor(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_features,
        hidden_dim=64,
        temporal_channels=32,
        edge_index=None,
    ):
        super().__init__()

        # Save edge_index for use in forward()
        self.edge_index = edge_index

        # GNN layers
        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Temporal CNN (fast, instead of LSTM)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, temporal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(temporal_channels, temporal_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Final readout per node
        self.fc = nn.Linear(temporal_channels, 1)

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index=None):
        # Use stored edge_index if user didn't pass one
        if edge_index is None:
            edge_index = self.edge_index

        # x shape: (B, W, N, F)
        B, W, N, F = x.shape

        outputs = []

        for t in range(W):
            # (B, N, F) → (B*N, F)
            xt = x[:, t].reshape(B * N, F)

            # GNN
            h = self.gcn1(xt, edge_index)
            h = torch.relu(h)
            h = self.gcn2(h, edge_index)
            h = torch.relu(h)

            # Back to (B, N, H)
            h = h.reshape(B, N, self.hidden_dim)
            outputs.append(h)

        # Stack temporal dimension
        H = torch.stack(outputs, dim=1)    # (B, W, N, H)

        # Temporal CNN wants channel-first:
        # (B, W, N, H) → (B*N, H, W)
        H = H.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, W)

        # Apply temporal CNN
        H = self.temporal_cnn(H)  # (B*N, C, W)

        # Take last timestep
        H_last = H[:, :, -1]  # (B*N, C)

        # Final prediction
        out = self.fc(H_last).reshape(B, N)

        return out
