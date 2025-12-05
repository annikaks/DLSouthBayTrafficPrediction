import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn



import config  # to read ranges / switches

from config import (
    DATA_PATH,
    WINDOW_SIZE,
    HORIZON,
    BATCH_SIZE,
)
from data import (
    load_raw_data,
    train_val_test_split,
    Normalizer,
    create_sliding_windows,
    TrafficDataset,
    add_context_features,
    add_spatial_avg_feature,
)
from models import (
    LinearRegressionModel,
    LogisticRegressionModel,
    MLPRegressor,
    LSTMRegressor,
    GNN_TemporalCNN_Regressor,
    GAT_Temporal_Regressor,
)
from train_utils import (
    train_regression_model,
    train_classification_model,
    evaluate,  # to get val RMSE for MLP trials
)
from baselines import evaluate_naive_last_value
from routes import ROUTES, compute_route_travel_time_minutes, load_sensor_data
from graph import build_knn_graph
from config import DEVICE, LR


def data_setup(data, scaler):
    """
    Full data pipeline:
    - time-based train/val/test split
    - fit scaler on train, apply to all
    - create sliding windows for each split
    - build regression + classification datasets & loaders
    - compute congestion threshold (on raw speeds)
    """
    # train/val/test split
    train_raw, val_raw, test_raw = train_val_test_split(data)

    # fit scaler on train, transform all
    scaler.fit(train_raw)
    train_norm = scaler.transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)

    # sliding windows
    X_train, y_train = create_sliding_windows(train_norm, WINDOW_SIZE, HORIZON)
    X_val, y_val = create_sliding_windows(val_norm, WINDOW_SIZE, HORIZON)
    X_test, y_test = create_sliding_windows(test_norm, WINDOW_SIZE, HORIZON)

    # regression setup
    train_ds_reg = TrafficDataset(X_train, y_train, task="regression")
    val_ds_reg = TrafficDataset(X_val, y_val, task="regression")
    train_loader_reg = DataLoader(train_ds_reg, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_reg = DataLoader(val_ds_reg, batch_size=BATCH_SIZE, shuffle=False)

    # classification threshold (~64 mph)
    speed_mean = float(scaler.mean[0, 0, 0])
    speed_std = float(scaler.std[0, 0, 0])
    y_train_raw = y_train * speed_std + speed_mean    # (S, N) in mph
    y_val_raw   = y_val   * speed_std + speed_mean    # (S, N) in mph

    congestion_threshold = np.percentile(y_train_raw, 25.0)
    print(f"\nCongestion threshold (mph): {congestion_threshold:.2f}")

    # classification setup
    train_ds_cls = TrafficDataset(
        X_train,
        y_train_raw,
        task="classification",
        congestion_threshold=congestion_threshold,
    )
    val_ds_cls = TrafficDataset(
        X_val,
        y_val_raw,
        task="classification",
        congestion_threshold=congestion_threshold,
    )
    train_loader_cls = DataLoader(train_ds_cls, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_cls = DataLoader(val_ds_cls, batch_size=BATCH_SIZE, shuffle=False)

    T, N, F = data.shape

    ctx = {
        "data": data,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_loader_reg": train_loader_reg,
        "val_loader_reg": val_loader_reg,
        "train_loader_cls": train_loader_cls,
        "val_loader_cls": val_loader_cls,
        "congestion_threshold": congestion_threshold,
        "N": N,
        "F": F,
    }
    return ctx


def run_naive(ctx):
    print("\n=== Naive baseline (last value) on validation set ===")
    evaluate_naive_last_value(ctx["X_val"], ctx["y_val"])


def run_linear(ctx):
    N, F = ctx["N"], ctx["F"]
    train_loader_reg = ctx["train_loader_reg"]
    val_loader_reg = ctx["val_loader_reg"]

    print("\n=== Training linear regression baseline ===")
    lin_model = LinearRegressionModel(WINDOW_SIZE, N, F)
    train_regression_model(lin_model, train_loader_reg, val_loader_reg)
    return lin_model


def run_logistic(ctx):
    N, F = ctx["N"], ctx["F"]
    train_loader_cls = ctx["train_loader_cls"]
    val_loader_cls = ctx["val_loader_cls"]

    print("\n=== Training logistic regression baseline (congestion classification) ===")
    log_model = LogisticRegressionModel(WINDOW_SIZE, N, F)
    train_classification_model(log_model, train_loader_cls, val_loader_cls)
    return log_model

def run_lstm(ctx):
    N, F = ctx["N"], ctx["F"]
    train_loader_reg = ctx["train_loader_reg"]
    val_loader_reg = ctx["val_loader_reg"]

    hidden_dim = getattr(config, "LSTM_HIDDEN_DIM", 128)
    num_layers = getattr(config, "LSTM_NUM_LAYERS", 1)
    dropout = getattr(config, "LSTM_DROPOUT", 0.0)

    print("\n=== Training LSTM regression model ===")
    lstm_model = LSTMRegressor(
        window_size=WINDOW_SIZE,
        num_nodes=N,
        num_features=F,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    train_regression_model(lstm_model, train_loader_reg, val_loader_reg)
    return lstm_model



def sample_from_range(range_tuple, base_val, is_int=False):
    """
    Randomly sample from a range (low, high). If None -> base_val.
    This is RANDOM search, not grid.
    """
    if range_tuple is None:
        return base_val
    low, high = range_tuple
    if is_int:
        return random.randint(int(low), int(high))
    else:
        return random.uniform(float(low), float(high))


def run_mlp_random_search(ctx):
    """
    Random hyperparameter search for the MLP within ranges defined in config.py.
    If no ranges are provided, trains a single MLP with base hyperparameters.
    Returns the best MLP (by validation RMSE).
    """
    N, F = ctx["N"], ctx["F"]
    train_loader_reg = ctx["train_loader_reg"]
    val_loader_reg = ctx["val_loader_reg"]

    print("\n=== Training MLP deep model(s) with random search ===")

    # base hyperparameters 
    base_hidden = getattr(config, "MLP_HIDDEN_DIM", 256)
    base_layers = getattr(config, "MLP_NUM_LAYERS", 3)
    base_dropout = getattr(config, "MLP_DROPOUT", 0.1)

    hidden_range = getattr(config, "MLP_HIDDEN_DIM_RANGE", None)
    layers_range = getattr(config, "MLP_NUM_LAYERS_RANGE", None)
    dropout_range = getattr(config, "MLP_DROPOUT_RANGE", None)

    num_trials = getattr(config, "MLP_NUM_TRIALS", 1)

    best_rmse = None
    best_model = None
    best_cfg = None

    for trial in range(num_trials):
        hidden_dim = sample_from_range(hidden_range, base_hidden, is_int=True)
        num_layers = sample_from_range(layers_range, base_layers, is_int=True)
        dropout = sample_from_range(dropout_range, base_dropout, is_int=False)

        print(
            f"\n[MLP trial {trial + 1}/{num_trials}] "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout:.3f}"
        )

        mlp_model = MLPRegressor(
            WINDOW_SIZE,
            N,
            F,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        train_regression_model(mlp_model, train_loader_reg, val_loader_reg)

        # eval on val
        criterion = torch.nn.MSELoss()
        val_loss, val_rmse, val_mae = evaluate(
            mlp_model, val_loader_reg, criterion, task="regression"
        )
        print(
            f"[MLP trial {trial + 1}] val_loss={val_loss:.4f}, "
            f"RMSE={val_rmse:.4f}, MAE={val_mae:.4f}"
        )

        if (best_rmse is None) or (val_rmse < best_rmse):
            best_rmse = val_rmse
            best_model = mlp_model
            best_cfg = (hidden_dim, num_layers, dropout)

    if best_model is not None:
        h, L, d = best_cfg
        print(
            f"\nBest MLP config: hidden_dim={h}, num_layers={L}, dropout={d:.3f}, "
            f"val_RMSE={best_rmse:.4f}"
        )
        return best_model
    else:
        # fallback: train a single base MLP
        print("\nNo MLP trials ran; training a single base MLP.")
        mlp_model = MLPRegressor(
            WINDOW_SIZE,
            N,
            F,
            hidden_dim=base_hidden,
            num_layers=base_layers,
            dropout=base_dropout,
        )
        train_regression_model(mlp_model, train_loader_reg, val_loader_reg)
        return mlp_model


def evaluate_route_with_model(ctx, model, model_name="model", route_dict=None):
    
    scaler = ctx["scaler"]
    val_loader_reg = ctx["val_loader_reg"]

    #set to routes for all but gnn regressor
    if route_dict is None:
        route_dict = ROUTES


    model.to(DEVICE)
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader_reg:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_hat = model(X_batch)

            all_true.append(y_batch.cpu())
            all_pred.append(y_hat.cpu())

    y_val_true = torch.cat(all_true, dim=0).numpy()   # (S, N)
    y_val_pred = torch.cat(all_pred, dim=0).numpy()   # (S, N)

    print(f"\n[{model_name}] Route-level ETA evaluation:")

    for route_name in route_dict:

        route = route_dict[route_name]             
        idx = route["sensor_indices"] 

        # ---- TRUE ETA (ground truth) ----
        true_tt = compute_route_travel_time_minutes(
            y_val_true,
            scaler,
            route_name=route_name,
            route_dict=route_dict,
        )

        # ✅ NEW SANITY DIAGNOSTIC
        print(
            f"[{route_name}] TRUE ETA stats → "
            f"mean = {true_tt.mean():.2f} min | "
            f"min = {true_tt.min():.2f} | "
            f"max = {true_tt.max():.2f}"
        )

        # ---- PREDICTED ETA ----
        pred_tt = compute_route_travel_time_minutes(
            y_val_pred,
            scaler,
            route_name=route_name,
            route_dict=route_dict,
        )

        # ---- ERROR METRICS ----
        rmse_tt = np.sqrt(((true_tt - pred_tt) ** 2).mean())
        mae_tt = np.abs(true_tt - pred_tt).mean()

        print(
            f"[{model_name} | Route: {route_name}] "
            f"Travel Time RMSE = {rmse_tt:.2f} min, "
            f"MAE = {mae_tt:.2f} min (validation)"
        )

#run gnn+temporal cnn or gat model 
def run_gnn_gat(ctx_gnn):
    N, F = ctx_gnn["N"], ctx_gnn["F"]
    edge_index = ctx_gnn["edge_index"]
    train_loader = ctx_gnn["train_loader_reg"]
    val_loader = ctx_gnn["val_loader_reg"]

    use_gat = getattr(config, "RUN_GAT", False)
    use_gnn = getattr(config, "RUN_GNN", False)

    assert use_gat or use_gnn, "Set RUN_GNN or RUN_GAT = True in config.py"

    # -------------------------------
    # Select model based on flags
    # -------------------------------
    if use_gnn:
        print("\n=== Training ST-GCN (GCN + Temporal CNN) ===")
        model = GNN_TemporalCNN_Regressor(
            num_nodes=N,
            in_features=F,
            hidden_dim=getattr(config, "GNN_HIDDEN_DIM", 32),
            temporal_channels=getattr(config, "GNN_TEMPORAL_CHANNELS", 32),
            edge_index=edge_index
        ).to(DEVICE)
        model_name = "ST-GCN"

    if use_gat:
        print("\n=== Training ST-GAT (GAT + Temporal CNN) ===")
        model = GAT_Temporal_Regressor(
            num_nodes=N,
            in_features=F,
            hidden_dim=getattr(config, "GAT_HIDDEN_DIM", 32),
            heads=getattr(config, "GAT_HEADS", 2),
            temporal_channels=getattr(config, "GAT_TEMPORAL_CHANNELS", 32),
            edge_index=edge_index
        ).to(DEVICE)
        model_name = "ST-GAT"

    # -------------------------------
    # Training loop
    # -------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            y_hat = model(X, edge_index)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        val_loss, val_rmse, val_mae = evaluate(
            model, val_loader, criterion, task="regression"
        )

        print(
            f"[{model_name}][Epoch {epoch}] "
            f"train_loss={total_loss / len(train_loader.dataset):.4f} "
            f"val_RMSE={val_rmse:.4f} MAE={val_mae:.4f}"
        )

    evaluate_route_with_model(ctx_gnn, model, model_name=model_name, route_dict=ctx_gnn["ROUTES"])



# =====================================================================
#                              MAIN()
# =====================================================================
def main():
    # Load raw data
    raw_data = load_raw_data(DATA_PATH)
    print("Raw data shape:", raw_data.shape)

    # Always add time features
    data = add_context_features(raw_data)

    # Optional spatial features
    use_spatial = getattr(config, "USE_SPATIAL_FEATURES", False)
    if use_spatial:
        k = getattr(config, "SPATIAL_K", 5)
        print(f"Adding spatial avg-neighbor feature (k={k})")
        data = add_spatial_avg_feature(data, k=k)
    else:
        print("Running WITHOUT spatial features")

    print("Final feature data shape:", data.shape)

    # ------------------------------------------------------------------
    # 1) Build full ctx (all sensors) for baselines + LSTM
    # ------------------------------------------------------------------
    scaler = Normalizer()
    ctx = data_setup(data, scaler)

    # ------------------------------------------------------------------
    # 2) Build reduced GNN dataset containing only route sensors
    # ------------------------------------------------------------------
    all_route_sensors = sorted({
        idx for route in ROUTES.values()
        for idx in route["sensor_indices"]
    })
    print("Total route sensors for GNN:", len(all_route_sensors))

    data_gnn = data[:, all_route_sensors, :]
    print("GNN data shape:", data_gnn.shape)

    scaler_gnn = Normalizer()
    ctx_gnn = data_setup(data_gnn, scaler_gnn)

    # ------------------------------------------------------------------
    # 3) Build KNN graph for GNN (only on route sensors)
    # ------------------------------------------------------------------
    ids, lats, lons, _ = load_sensor_data()
    route_lats = lats[all_route_sensors]
    route_lons = lons[all_route_sensors]

    edge_index_np = build_knn_graph(route_lats, route_lons, k=5)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(DEVICE)

    ctx_gnn["edge_index"] = edge_index

    # ------------------------------------------------------------------
    # 4) Remap route indices for GNN context
    # ------------------------------------------------------------------
    global_to_local = {g: i for i, g in enumerate(all_route_sensors)}

    ROUTES_GNN = {}
    for name, route in ROUTES.items():
        ROUTES_GNN[name] = {
            "sensor_indices": [global_to_local[i] for i in route["sensor_indices"]],
            "segment_lengths_mi": route["segment_lengths_mi"],
        }

    ctx_gnn["ROUTES"] = ROUTES_GNN

    # ------------------------------------------------------------------
    # Model selections
    # ------------------------------------------------------------------
    run_greedy = getattr(config, "RUN_GREEDY", True)
    run_linear_flag = getattr(config, "RUN_LINEAR", True)
    run_logistic_flag = getattr(config, "RUN_LOGISTIC", True)
    run_mlp_flag = getattr(config, "RUN_MLP", True)
    run_lstm_flag = getattr(config, "RUN_LSTM", True)
    run_gnn_flag = getattr(config, "RUN_GNN", True)
    run_gat_flag = getattr(config, "RUN_GAT", True)

    if run_greedy:  run_naive(ctx)
    if run_linear_flag: run_linear(ctx)
    if run_logistic_flag: run_logistic(ctx)

    if run_mlp_flag:
        best_mlp = run_mlp_random_search(ctx)
        evaluate_route_with_model(ctx, best_mlp, model_name="MLP")

    if run_lstm_flag:
        lstm_model = LSTMRegressor(
            WINDOW_SIZE, ctx["N"], ctx["F"]
        )
        train_regression_model(lstm_model, ctx["train_loader_reg"], ctx["val_loader_reg"])
        evaluate_route_with_model(ctx, lstm_model, model_name="LSTM")

    if run_gnn_flag or run_gat_flag:
        run_gnn_gat(ctx_gnn)


if __name__ == "__main__":
    main()
