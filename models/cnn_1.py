import torch.nn as nn
from models.layers import CBR


class CNN1(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN1, self).__init__()

        self.layers = nn.Sequential(
            CBR(3, 64, 3),
            CBR(64, 64, 3),
            nn.MaxPool2d(2, 2),

            # 16 x 16 map-size

            CBR(64, 64, 3),
            CBR(64, 64, 3),
            nn.MaxPool2d(2, 2),

            # 8 x 8 map-size

            CBR(64, 128, 3),
            CBR(128, 128, 3),
            nn.MaxPool2d(2, 2),

            # 4 x 4 map-size

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, num_classes, bias=True),
        )

    def forward(self, x):
        return self.layers(x)
