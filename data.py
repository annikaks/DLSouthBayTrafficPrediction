# data.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from config import TRAIN_RATIO, VAL_RATIO
from datetime import datetime, timedelta



class Normalizer:
    """
    z-score scaler.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        # data: (T, N, F)
        # Average over time (0) and sensors (1), keep per-feature stats.
        self.mean = data.mean(axis=(0, 1), keepdims=True)      # (1, 1, F)
        self.std = data.std(axis=(0, 1), keepdims=True) + 1e-6 # (1, 1, F)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


PEMS_START_DATETIME = datetime(2022, 1, 1, 0, 0)  # assume start : Jan 1, 2022, 00:00
PEMS_STEP_MINUTES = 5  # 5-minute intervals (from PEMS)

# holiday list (add Stanford events using agents)
HOLIDAYS_2022 = {
    datetime(2022, 1, 1).date(),   # New Year's 
    datetime(2022, 2, 14).date(),   # Valentines Day
    datetime(2022, 7, 4).date(),   # July 4th
    datetime(2022, 11, 24).date(), # Thanksgiving
    datetime(2022, 12, 25).date(), # Christmas
}


def add_context_features(
    data: np.ndarray,
    start_dt: datetime = PEMS_START_DATETIME,
    step_minutes: int = PEMS_STEP_MINUTES,
    add_weather: bool = True,
) -> np.ndarray:
    """
    Augment (T, N, F) traffic data with:
      - hour_of_day(t)          in [0, 23]
      - day_of_week(t)          in [0, 6]  (0=Monday)
      - is_holiday(t)           in {0, 1}
      - weather(t)              placeholder (currently zeros)

    Returns:
        data_aug: (T, N, F + extra_features)
    """
    T, N, F = data.shape

    # Build per-timestep datetimes
    times = [start_dt + timedelta(minutes=step_minutes * t) for t in range(T)]

    hours = np.array([dt.hour for dt in times], dtype=np.float32)          # (T,)
    dows = np.array([dt.weekday() for dt in times], dtype=np.float32)      # (T,)
    is_holiday = np.array(
        [1.0 if dt.date() in HOLIDAYS_2022 else 0.0 for dt in times],
        dtype=np.float32,
    )  # (T,)

    # Broadcast to (T, N, 1)
    hours_feat = np.tile(hours[:, None, None], (1, N, 1))       # (T, N, 1)
    dows_feat = np.tile(dows[:, None, None], (1, N, 1))         # (T, N, 1)
    holiday_feat = np.tile(is_holiday[:, None, None], (1, N, 1))# (T, N, 1)

    features = [data, hours_feat, dows_feat, holiday_feat]

    if add_weather:
        # Placeholder: all zeros. Later you can replace with real weather per time.
        weather_feat = np.zeros_like(hours_feat, dtype=np.float32)  # (T, N, 1)
        features.append(weather_feat)

    data_aug = np.concatenate(features, axis=2)  # (T, N, F + extra)
    return data_aug

def _load_from_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError(f".npy file at {path} did not contain a numpy array, got {type(arr)}")


def load_raw_data(path: str) -> np.ndarray:
    """
    Load preprocessed PeMS-BAY-2022 from .npy
    Make sure you have downloaded and copied the "PEMSBAY_2022.npy" file in data/
    Should be located: "CS230DeepLearningProject/data/PEMSBAY_2022.npy"

    Returns always shape (T, N, F):
      - If data is (T, N) it is promoted to (T, N, 1).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = _load_from_npy(path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .npy")

    if data.ndim == 2:
        # (T, N) -> (T, N, 1)
        data = data[..., None]

    if data.ndim != 3:
        raise ValueError(f"Expected data with 2 or 3 dims, got shape {data.shape}")

    return data.astype(np.float32)  # (T, N, F)


def train_val_test_split(data: np.ndarray): # this is naive, we should change this to not time based
    """
    Time-based split along T using TRAIN_RATIO and VAL_RATIO.
    Returns (train, val, test).
    """
    T = data.shape[0]
    train_end = int(T * TRAIN_RATIO)
    train = data[:train_end]

    val_end = int(T * (TRAIN_RATIO + VAL_RATIO))
    val = data[train_end:val_end]

    test = data[val_end:]

    return train, val, test


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    horizon: int,
):
    """
    data: (T, N, F) normalized
    Returns:
        Essentially, aggregate the last hour of speed as the "X" 
        and the "y" as the speed in the next window (sets up to use 
        the last hour of speed to predict the next five minutes)
        
        X: (S, W, N, F) input windows
        y: (S, N)       target next-step speed (feature 0)

    """
    T, N, F = data.shape
    num_samples = T - window_size - horizon + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough time steps ({T}) for window_size={window_size}, horizon={horizon}"
        )

    X_list = []
    y_list = []
    for t in range(num_samples): # create windows S windows of size WINDOW_SIZE
        X_list.append(data[t : t + window_size])   # (W, N, F)
        target_t = t + window_size + horizon - 1
        y_list.append(data[target_t, :, 0])        # speed only

    X = np.stack(X_list)  # (S, W, N, F)
    y = np.stack(y_list)  # (S, N)
    return X, y


class TrafficDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_speed: np.ndarray,
        task: str = "regression",
        congestion_threshold: float = None,
    ):
        """
        X: (S, W, N, F)
        y_speed: (S, N) speed targets (normalized)
        """
        self.X = torch.from_numpy(X).float()
        self.y_reg = torch.from_numpy(y_speed).float()
        self.task = task

        if task == "classification":
            if congestion_threshold is None:
                raise ValueError("Need congestion_threshold for classification task.")
            
            # Binary labels: 1 if speed < threshold else 0
            y_np = self.y_reg.numpy()
            y_cls_np = (y_np < congestion_threshold).astype(np.float32)
            self.y_class = torch.from_numpy(y_cls_np)
        else:
            self.y_class = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.task == "regression":
            return self.X[idx], self.y_reg[idx]
        else:
            return self.X[idx], self.y_class[idx]


# potential loading functions for other data file types:
def _load_from_npz(path: str) -> np.ndarray:
    npz = np.load(path)
    # Try some common keys; fall back to the first thing we find
    for key in ["data", "speed", "arr_0", "x", "X"]:
        if key in npz:
            return npz[key]
    first_key = list(npz.keys())[0]
    return npz[first_key]


def _load_from_pkl(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # direct array
    if isinstance(obj, np.ndarray):
        return obj

    # dict of arrays
    if isinstance(obj, dict):
        for key in ["data", "speed", "x", "X"]:
            if key in obj and isinstance(obj[key], np.ndarray):
                return obj[key]
        raise ValueError(
            f".pkl file {path} is a dict but none of keys "
            f"['data', 'speed', 'x', 'X'] contained a numpy array. "
            f"Found keys: {list(obj.keys())}"
        )

    raise TypeError(
        f"Unsupported object type in {path}: {type(obj)}. "
        f"Expected numpy array or dict of arrays."
    )

def add_spatial_avg_feature(data: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Adds average neighbor speed as a new feature.
    
    data: (T, N, F) where data[..., 0] is speed
    k: number of neighbors on each side (local window)
    
    Returns:
        (T, N, F+1)
    """
    T, N, F = data.shape
    speed = data[:, :, 0]  # (T, N)

    avg_neighbors = np.zeros_like(speed)

    for i in range(N):
        left = max(0, i - k)
        right = min(N, i + k + 1)
        avg_neighbors[:, i] = speed[:, left:right].mean(axis=1)

    avg_neighbors = avg_neighbors[:, :, None]  # (T, N, 1)

    data_aug = np.concatenate([data, avg_neighbors], axis=2)
    return data_aug
