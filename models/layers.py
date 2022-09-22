import torch.nn as nn


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, relu=True):
        super(CBR, self).__init__()

        padding = (ksize - stride) // 2

        layers = [
            nn.Conv2d(in_ch, out_ch, ksize, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
        ]

        if relu:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, exp_fac=3, stride=1):
        super(InvertedResidual, self).__init__()
        layers = list()

        mid_ch = int(in_ch * exp_fac)

        if exp_fac != 1:
            layers.append(CBR(in_ch, mid_ch, 1, relu=True))

        layers.append(nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, (ksize, ksize), stride=(stride, stride), padding=(ksize - 1) // 2,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
        ))

        layers.append(CBR(mid_ch, out_ch, 1, relu=False))

        self.res = stride == 1 and in_ch == out_ch

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        l_out = self.layers(x)
        if self.res:
            l_out = l_out + x
        return l_out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, exp_fac=1, stride=1):
        super(ResBlock, self).__init__()
        layers = list()

        mid_ch = int(in_ch * exp_fac)

        layers.append(CBR(in_ch, mid_ch, ksize, relu=True))

        layers.append(CBR(mid_ch, out_ch, 1, relu=False))

        self.res = stride == 1 and in_ch == out_ch

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        l_out = self.layers(x)
        if self.res:
            l_out = l_out + x
        return l_out
