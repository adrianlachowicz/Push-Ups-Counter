import torch


# Image config
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Data paths
TRAIN_DATA_PATH = "./data/youtube_videos/1/train/"
VALID_DATA_PATH = "./data/youtube_videos/1/valid/"
TEST_DATA_PATH = "./data/youtube_videos/1/test/"
CSV_FILENAME = "data.csv"

# Training config
TRAIN_NAME = "test1"
EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 1
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
