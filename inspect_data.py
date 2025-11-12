import numpy as np

path = "../CS230DeepLearningProject/data/PEMSBAY_2022.npy"

data = np.load(path, allow_pickle=True)
print("Type:", type(data))
print("Shape:", getattr(data, "shape", "no shape"))
print("Dtype:", getattr(data, "dtype", "no dtype"))

# Peek at a few values
if isinstance(data, np.ndarray):
    print("Sample values:", data[:2])
else:
    print("Object type:", type(data))
