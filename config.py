import torch

# PeMS-BAY 2022 file (.npz)
DATA_PATH = "../CS230DeepLearningProject/data/PEMSBAY_2022.npy"  # TODO: make sure your env is setup like this

# window + horizon
WINDOW_SIZE = 12  # past 12, 5 min windows = each window is one hour
HORIZON = 6       # predict 1 step (5 mins) ahead

# splits (time-based)   
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1         # test is 1 - TRAIN_RATIO - VAL_RATIO = 0.2
#TEST_RATIO = 0.2       # not sure if we should do train - 0.7, val - 0.15, test - 0.15

# hyperparameters, make this a grid search eventually
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
