import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_COVER = "data/train/cover"
TRAIN_DIR_HIDDEN = "data/train/hidden"
VAL_DIR = "data/val"
LR_ENCODER = 0.001
LR_DECODER = 0.001
LR_DISCRIMINATOR = 0.0005
BATCH_SIZE = 16
NUM_EPOCHS = 500
BETA = 0.75
GAMMA = 1
NUM_WORKERS = 3
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_ENC = "enc.pth.tar"
CHECKPOINT_DEC = "dec.pth.tar"