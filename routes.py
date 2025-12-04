import numpy as np

# Example route; replace with real sensor indices + segment lengths.
ROUTES = {
    "stanford_to_sfo": {
        "sensor_indices": [0, 1, 2],           # TODO: replace with real sensor indices
        "segment_lengths_mi": [1.0, 1.5, 2.0], # TODO: replace with real segment lengths
    },
}


def compute_route_travel_time_minutes(
    y_speed_norm: np.ndarray,
    scaler,
    route_name: str,
) -> np.ndarray:
    """
    Convert sensor-level normalized speed predictions into route travel time.

    y_speed_norm: (S, N) normalized speeds (feature 0 only)
    scaler: Normalizer that was fit on full (T, N, F) data
    route_name: key in ROUTES

    Returns:
        travel_time: (S,) array in minutes
    """
    route = ROUTES[route_name]
    idx = route["sensor_indices"]
    lengths = np.array(route["segment_lengths_mi"], dtype=np.float32)  # (K,)

    # Extract normalized speeds along the route: (S, K)
    speeds_norm_route = y_speed_norm[:, idx]

    # Use only the speed feature (feature index 0) stats:
    speed_mean = float(scaler.mean[0, 0, 0])
    speed_std = float(scaler.std[0, 0, 0])

    # Denormalize to mph:
    speeds_mph = speeds_norm_route * speed_std + speed_mean  # (S, K)

    # Avoid division by zero:
    MIN_SPEED_MPH = 5.0  # or 10.0
    speeds_safe = np.where(speeds_mph > MIN_SPEED_MPH, speeds_mph, MIN_SPEED_MPH)

    # time (hours) = length (miles) / speed (mph)
    time_hours = lengths / speeds_safe            # (K,) broadcast -> (S, K)
    time_hours_total = time_hours.sum(axis=1)     # (S,)
    time_minutes = time_hours_total * 60.0
    return time_minutes
