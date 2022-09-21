import torch.nn as nn
from models.layers import CRB


class CNN1(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN1, self).__init__()

        self.layers = nn.Sequential(
            CRB(3, 64, 3),
            CRB(64, 64, 3),
            nn.MaxPool2d(2, 2),

            CRB(64, 64, 3),
            CRB(64, 64, 3),
            nn.MaxPool2d(2, 2),

            CRB(64, 128, 3),
            CRB(128, 128, 3),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            nn.Linear(128, 128, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, num_classes, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)
