import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(in_features=4, out_features=2))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
