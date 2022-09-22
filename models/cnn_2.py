import torch.nn as nn
from models.layers import CBR, ResBlock, InvertedResidual


# 4 + (16 x 3) = 52 convolution layers


class CNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN2, self).__init__()

        self.layers = nn.Sequential(
            CBR(3, 64, 3, relu=True),

            InvertedResidual(64, 64, 3, 3),
            InvertedResidual(64, 64, 5, 3),
            InvertedResidual(64, 64, 3, 3),

            CBR(64, 128, 3, relu=True),
            nn.MaxPool2d(2, 2),

            # 16 x 16 map-size

            InvertedResidual(128, 128, 3, 3),
            InvertedResidual(128, 128, 5, 3),
            InvertedResidual(128, 128, 3, 3),
            InvertedResidual(128, 128, 5, 3),

            CBR(128, 256, 3, relu=True),
            nn.MaxPool2d(2, 2),

            # 8 x 8 map-size

            InvertedResidual(256, 256, 3, 4),
            InvertedResidual(256, 256, 5, 4),
            InvertedResidual(256, 256, 3, 4),
            InvertedResidual(256, 256, 5, 4),
            InvertedResidual(256, 256, 3, 4),

            CBR(256, 512, 3, relu=True),
            nn.MaxPool2d(2, 2),

            # 4 x 4 map-size
            InvertedResidual(512, 512, 3, 4),
            InvertedResidual(512, 512, 3, 4),
            InvertedResidual(512, 256, 3, 4),
            InvertedResidual(256, 256, 3, 4),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(start_dim=1),

            nn.Linear(256, num_classes, bias=True),
        )

    def forward(self, x):
        return self.layers(x)
