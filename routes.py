import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# base directory 
BASE_DIR = os.path.join(os.path.dirname(__file__), "data")

#metadata for all 325 sensors; formatted as: ID, Latitude, Longitude
SENSOR_PATH = os.path.join(BASE_DIR, "graph_sensor_locations_bay.csv")

# Full station metadata (3992 rows) with Fwy, Dir, Abs_PM, Length included
FULL_META_PATH = os.path.join(BASE_DIR, "d04_text_meta_2021_10_16.txt")

# Rough estimates of Stanford and SFO latitudes
LAT_STANFORD = 37.2
LAT_SFO = 37.8


def load_sensor_data(path=SENSOR_PATH):
    # Return ids, lat, long, id_mapping
    data = np.loadtxt(path, delimiter=",")

    ids = data[:,0].astype(int)
    lat = data[:,1]
    long = data[:,2]

    id_mapping = {idx:id for id, idx in enumerate(ids)}

    return ids, lat, long, id_mapping

def load_all_metadata(path=FULL_META_PATH):
    #return id_data which maps sensor id to metadata(fwy, direction, length, etc.)

    id_data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")  

        for row in reader:

            try:
                id = int(row["ID"])
                freeway = row["Fwy"]
                direction = row["Dir"]
                position = float(row["Abs_PM"])
                length = float(row["Length"])
                lat = float(row["Latitude"])
                long = float(row["Longitude"])

            except (KeyError, ValueError):
                #ignore
                continue

            id_data[id] = {
                "Fwy": freeway,
                "Dir": direction,
                "Abs_PM": position,
                "Length": length,
                "lat": lat,
                "lon": long,
            }

    return id_data


def get_directional_route(freeway, direction, sensorIds, latitudes, id_data, id_mapping):
    #build a route along the given freeway and direction 

    selectedIds = []

    for id, idx in id_mapping.items(): 

        if id not in id_data:
            continue

        currData = id_data[id]

        #only consider sensors in given freeway and direction
        if str(currData["Fwy"]) != str(freeway):
            continue
        if currData["Dir"].upper() != direction.upper():
            continue 


        #only consider sensors between Stanford and SFO
        if not (LAT_STANFORD <= currData["lat"] <= LAT_SFO):
            continue

        selectedIds.append((idx, currData["Abs_PM"], currData["Length"]))

    selectedIds.sort(key = lambda x:x[1])

    #keep only the number of max_segments for routes to reduce noise; ablation study
    MAX_SEGMENTS = 10
    #select max segment number of sensors, evenly spaced out throughout the route
    if len(selectedIds) > MAX_SEGMENTS:
        step = len(selectedIds) / MAX_SEGMENTS
        chosen = []
        for i in range(MAX_SEGMENTS):
            idx = int(i * step)
            chosen.append(selectedIds[idx])
        selectedIds = chosen

    selectedIndices = [sensor[0] for sensor in selectedIds]
    selectedLengths = [sensor[2] for sensor in selectedIds]

    return selectedIndices, selectedLengths



def build_all_routes():

    #this will build 3 routes to SFO, one along the 101, 280 and 880
    ids, lats, longs, id_to_idx = load_sensor_data()
    metadata = load_all_metadata()
    print("Total metadata stations loaded:", len(metadata))
    routes = {}

    #101 route
    idx_101, len_101 = get_directional_route(
            freeway="101",
            direction="N",
            sensorIds=ids,
            latitudes=lats,
            id_data=metadata,
            id_mapping=id_to_idx
        )
    
    routes["stanford_to_sfo_101"] = {
            "sensor_indices": idx_101,
            "segment_lengths_mi": len_101,
        }
    

    #280 route
    idx_280, len_280 = get_directional_route(
        freeway="280",
        direction="N",
        sensorIds=ids,
        latitudes=lats,
        id_data=metadata,
        id_mapping=id_to_idx
    )
    if len(idx_280) >= 2:
        routes["stanford_to_sfo_280"] = {
            "sensor_indices": idx_280,
            "segment_lengths_mi": len_280,
        }

    #880/small roads route
    idx_880, len_880 = get_directional_route(
        freeway="880",
        direction="N",
        sensorIds=ids,
        latitudes=lats,
        id_data=metadata,
        id_mapping=id_to_idx
    )
    if len(idx_880) >= 2:
        routes["stanford_to_sfo_880"] = {
            "sensor_indices": idx_880,
            "segment_lengths_mi": len_880,
        }

    return routes

ROUTES = build_all_routes()

def plot_routes():
    ids, lats, lons, mapping = load_sensor_data(SENSOR_PATH)

    plt.figure(figsize=(8, 10))


    plt.scatter(lons, lats)
    plt.title("Stanford → SFO Route Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")


    for name, route in ROUTES.items():
        idx = route["sensor_indices"]

        route_lats = lats[idx]
        route_lons = lons[idx]

        plt.plot(route_lons, route_lats, marker="o", linewidth=2, label=name)
        plt.scatter(route_lons[0], route_lats[0], marker="s")   # route start
        plt.scatter(route_lons[-1], route_lats[-1], marker="X") # route end

    plt.legend()
    plt.grid(True)
    plt.show()

def compute_route_travel_time_minutes(
    y_speed_norm: np.ndarray,
    scaler,
    route_name: str,
    route_dict = None
) -> np.ndarray:
    """
    Convert sensor-level normalized speed predictions into route travel time.

    y_speed_norm: (S, N) normalized speeds (feature 0 only)
    scaler: Normalizer that was fit on full (T, N, F) data
    route_name: key in ROUTES

    Returns:
        travel_time: (S,) array in minutes
    """
    if route_dict is None:
        route_dict = ROUTES

    route = route_dict[route_name]
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


    










