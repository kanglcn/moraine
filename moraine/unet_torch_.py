"""UNet used by the Noise2Fringe (n2f) and Noise2Fringe with ADI (n2fs3d) models."""

__all__ = ['UNet']

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.normalize(self.conv(x),dim=-3)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, n_filters_at_firstlayer=64, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.n_filters_at_firstlayer = n_filters_at_firstlayer
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, n_filters_at_firstlayer)

        factor = 2 if bilinear else 1
        self.down_path = nn.ModuleList()
        prev_channels = n_filters_at_firstlayer
        for i in range(depth):
            next_channels = prev_channels*2
            if i < depth -1:
                self.down_path.append(Down(prev_channels, next_channels))
            else:
                self.down_path.append(Down(prev_channels, next_channels//factor))
            prev_channels = next_channels

        self.up_path = nn.ModuleList()
        for i in range(depth):
            next_channels = prev_channels//2
            if i < depth -1:
                self.up_path.append(Up(prev_channels, next_channels // factor, bilinear))
            else:
                self.up_path.append(Up(prev_channels, next_channels, bilinear))
            prev_channels = next_channels

        self.outc = OutConv(next_channels, out_channels)

    def forward(self, x):
        x_list = [self.inc(x),]
        for down in self.down_path:
            x_list.append(down(x_list[-1]))
        x = x_list.pop()
        for up in self.up_path:
            x = up(x, x_list.pop())
        out = self.outc(x)
        return out
