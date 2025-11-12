import torch
from train_utils import regression_metrics


def evaluate_naive_last_value(X, y):
    """
    Super reedy baseline:
    Next speed = last observed speed in the window.

    X: (S, W, N, F) normalized
    y: (S, N) normalized
    """
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).float()

    last_speed = X_torch[:, -1, :, 0] 
    rmse, mae = regression_metrics(y_torch, last_speed)
    print(f"[Naive last-value] RMSE={rmse:.4f}, MAE={mae:.4f}")
    return rmse, mae
