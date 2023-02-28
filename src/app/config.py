import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFIER_CHECKPOINT = "src/pipeline/models_checkpoints/basic-classifier.pth"
