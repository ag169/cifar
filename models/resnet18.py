import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        net = resnet18()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            net.bn1,
            nn.ReLU(),

            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(start_dim=1),

            nn.Linear(512, num_classes, bias=True),
        )

        print()

    def forward(self, x):
        return self.layers(x)

