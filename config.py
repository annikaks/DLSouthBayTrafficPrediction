import torch

# PeMS-BAY 2022 file (.npz)
DATA_PATH = "../CS230DeepLearningProject/data/PEMSBAY_2022.npy"  # TODO: make sure your env is setup like this

# window + horizon
WINDOW_SIZE = 12  # past 12, 5 min windows = each window is one hour
HORIZON = 12       # predict 12 step (60 mins) ahead

# splits (time-based)   
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1         # test is 1 - TRAIN_RATIO - VAL_RATIO = 0.2
#TEST_RATIO = 0.2       # not sure if we should do train - 0.7, val - 0.15, test - 0.15

USE_SPATIAL_FEATURES = True #false = temporal only and true = spatio-temporal
SPATIAL_K = 5

# models
RUN_GREEDY = False
RUN_LINEAR = False
RUN_LOGISTIC = False
RUN_MLP = False
RUN_LSTM = False   
RUN_GNN = False
RUN_GAT = True


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

# LSTM hyperparameters 
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.0

#GNN hyperparamters
GNN_HIDDEN_DIM = 32
GNN_TEMPORAL_CHANNELS = 32

#GAT hyperparamters
GAT_HIDDEN_DIM = 32
GAT_HEADS = 2
GAT_TEMPORAL_CHANNELS = 32


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
