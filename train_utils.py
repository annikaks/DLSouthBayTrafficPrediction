import torch
from torch import nn
from config import DEVICE, LR, EPOCHS


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    y_true, y_pred: (S, N)
    Returns RMSE, MAE.
    """
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = (y_true - y_pred).abs().mean()
    return rmse.item(), mae.item()


def classification_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5):
    """
    y_true: (S, N) in {0,1}
    y_prob: (S, N) in [0,1]
    Returns accuracy.
    """
    y_pred = (y_prob >= threshold).float()
    correct = (y_pred == y_true).float().mean()
    return correct.item()


def train_epoch(model, dataloader, optimizer, criterion, task: str):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, task: str):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item() * X_batch.size(0)

        if task == "regression":
            all_true.append(y_batch.cpu())
            all_pred.append(outputs.cpu())
        else:
            probs = torch.sigmoid(outputs)
            all_true.append(y_batch.cpu())
            all_pred.append(probs.cpu())

    all_true = torch.cat(all_true, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    avg_loss = total_loss / len(dataloader.dataset)

    if task == "regression":
        rmse, mae = regression_metrics(all_true, all_pred)
        return avg_loss, rmse, mae
    else:
        acc = classification_metrics(all_true, all_pred)
        return avg_loss, acc


def train_regression_model(
    model,
    train_loader,
    val_loader,
    epochs: int = EPOCHS,
    lr: float = LR,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, task="regression")
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, task="regression")
        print(
            f"[Reg][Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"RMSE={val_rmse:.4f} MAE={val_mae:.4f}"
        )

    return model


def train_classification_model(
    model,
    train_loader,
    val_loader,
    epochs: int = EPOCHS,
    lr: float = LR,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, task="classification")
        val_loss, val_acc = evaluate(model, val_loader, criterion, task="classification")
        print(
            f"[Cls][Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"acc={val_acc:.4f}"
        )

    return model
