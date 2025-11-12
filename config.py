import torch

# PeMS-BAY 2022 file (.npz)
DATA_PATH = "../CS230DeepLearningProject/data/PEMSBAY_2022.npy"  # TODO: make sure your env is setup like this

# window + horizon
WINDOW_SIZE = 12  # past 12, 5 min windows = each window is one hour
HORIZON = 6       # predict 6 step (30 mins) ahead

# splits (time-based)   
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1         # test is 1 - TRAIN_RATIO - VAL_RATIO = 0.2
#TEST_RATIO = 0.2       # not sure if we should do train - 0.7, val - 0.15, test - 0.15

# models
RUN_GREEDY = True
RUN_LINEAR = True
RUN_LOGISTIC = True
RUN_MLP = True


# hyperparameters
# base
MLP_HIDDEN_DIM = 256
MLP_NUM_LAYERS = 3
MLP_DROPOUT = 0.1
BATCH_SIZE = 64
LR = 1e-3          # learning rate
EPOCHS = 10        # number of training epochs

# random search
MLP_HIDDEN_DIM_RANGE = (128, 512)   
MLP_NUM_LAYERS_RANGE = (2, 4)       
MLP_DROPOUT_RANGE = (0.0, 0.3)  


MLP_NUM_TRIALS = 3  # try 3 configurations


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
