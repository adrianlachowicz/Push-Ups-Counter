import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.Linear(in_features=16, out_features=2),
        )

        self.weights = [1 / 193, 1 / 298]
        self.weights = torch.tensor(self.weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, x):
        return self.model(x)
