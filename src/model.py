"""
NuClick Architecture
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()

        self.increase = in_channels != out_channels
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, padding)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1) if self.increase else None
        self.out = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)

        # Increase input feature dimension if smaller than output dimension
        input = self.conv3(input) if self.increase else input

        return self.out(input + output)


class MultiScaleConvBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, paddings, dilations):
        super(MultiScaleConvBlock, self).__init__()

        self.convs = nn.ModuleList(
            [ConvBlock(4 * channels, channels, kernel_sizes[i], paddings[i], dilations[i])
             for i in range(4)]
        )

    def forward(self, input):
        return torch.cat([conv(input) for conv in self.convs], 1)


class NuClick(nn.Module):
    def __init__(self):
        super(NuClick, self).__init__()

        self.downs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(5, 64, 7, 3),
                ConvBlock(64, 32, 5, 2),
                ConvBlock(32, 32)
            ),
            nn.Sequential(
                ResidualBlock(32, 64),
                ResidualBlock(64, 64)
            ),
            nn.Sequential(
                ResidualBlock(64, 128),
                MultiScaleConvBlock(32, [3, 3, 5, 5], [1, 3, 6, 12], [1, 3, 3, 6]),
                ResidualBlock(128, 128)
            ),
            nn.Sequential(
                ResidualBlock(128, 256),
                ResidualBlock(256, 256),
                ResidualBlock(256, 256)
            ),
            nn.Sequential(
                ResidualBlock(256, 512),
                ResidualBlock(512, 512),
                ResidualBlock(512, 512)
            ),
            nn.Sequential(
                ResidualBlock(512, 1024),
                ResidualBlock(1024, 1024),
                nn.ConvTranspose2d(1024, 512, 2, 2)
            )
        ])

        self.ups = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(1024, 512),
                ResidualBlock(512, 256),
                nn.ConvTranspose2d(256, 256, 2, 2)
            ),
            nn.Sequential(
                ResidualBlock(512, 256),
                MultiScaleConvBlock(64, [3, 3, 5, 5], [1, 3, 4, 6], [1, 3, 2, 3]),
                ResidualBlock(256, 256),
                nn.ConvTranspose2d(256, 128, 2, 2)
            ),
            nn.Sequential(
                ResidualBlock(256, 128),
                ResidualBlock(128, 128),
                nn.ConvTranspose2d(128, 64, 2, 2)
            ),
            nn.Sequential(
                ResidualBlock(128, 64),
                MultiScaleConvBlock(16, [3, 3, 5, 7], [1, 3, 6, 18], [1, 3, 3, 6]),
                ResidualBlock(64, 64),
                nn.ConvTranspose2d(64, 32, 2, 2)
            ),
            nn.Sequential(
                ConvBlock(64, 64),
                ConvBlock(64, 32),
                ConvBlock(32, 32)
            )
        ])

        self.head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        outs = []

        for down in self.downs[:-1]:
            x = down(x)
            outs.append(x)
            x = self.max_pool(x)

        x = self.downs[-1](x)

        for up in self.ups:
            x = torch.cat([x, outs.pop()], 1)
            x = up(x)

        return self.head(x)
