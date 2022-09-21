import torch.nn as nn


class CRB(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1):
        super(CRB, self).__init__()

        padding = (ksize - stride) // 2

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.layers(x)
