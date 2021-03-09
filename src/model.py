"""
Neural network architecture
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    """
    Class representing a convolutional block.

    Attributes
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int (default=3)
        The kernel size.
    padding : int (default=1)
        The padding rate.
    dilation : int (default=1)
        The dilation rate.

    Notes
    -----
    A convolutional block is composed of a convolutional layer followed by
    a batch normalization layer and ends with a ReLU activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ResidualBlock(nn.Module):
    """
    Class representing a residual block.

    Attributes
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int (default=3)
        The kernel size.
    padding : int (default=1)
        The padding rate.

    Methods
    -------
    forward(input)
        Compute the forward pass given an input.

    Notes
    -----
    A residual block is the concatenation of two or three convolutional blocks
    followed by a ReLU activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.increase = in_channels != out_channels
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, padding)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1) if self.increase else None
        self.out = nn.ReLU(inplace=True)

    def forward(self, input):
        """
        Compute the forward pass given an input.

        Parameters
        ----------
        input : Tensor
            The input of the block.

        Return
        ------
        output : Tensor
            The output of the forward pass.
        """

        output = self.conv1(input)
        output = self.conv2(output)

        # Increase input feature dimension if smaller than output dimension
        input = self.conv3(input) if self.increase else input

        return self.out(input + output)


class MultiScaleConvBlock(nn.Module):
    """
    Class representing a multi-scale convolutional block.

    Attributes
    ----------
    channel : int
        The number of output channels.
    kernel_sizes : list of int
        The kernel size for each layer.
    paddings : list of int
        The padding rates.
    dilations : list of int
        The dilations rates.

    Methods
    -------
    forward(input)
        Compute the forward pass given an input.

    Notes
    -----
    A multi-scale convolutional block is composed of four convolutional blocks
    each taking as input channels one fourth of the output channels and are
    combining at the end of the forward pass.
    """

    def __init__(self, channels, kernel_sizes, paddings, dilations):
        super().__init__()

        self.convs = nn.ModuleList(
            [ConvBlock(4 * channels, channels, kernel_sizes[i], paddings[i], dilations[i])
             for i in range(4)]
        )

    def forward(self, input):
        """
        Compute the forward pass given an input.

        Parameters
        ----------
        input : Tensor
            The input of the block.

        Return
        ------
        output : Tensor
            The output of the forward pass.
        """
        return torch.cat([conv(input) for conv in self.convs], dim=1)


class NuClick(nn.Module):
    """
    Class representing the NuClick neural network architecture.

    Methods
    -------
    forward(x)
        Predict the segmentation mask of an image.

    Notes
    -----
    Reference paper: https://arxiv.org/pdf/2005.14511.pdf
    """

    def __init__(self):
        super().__init__()

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
        """
        Predict the mask of a given input x.

        Parameters
        ----------
        x : Tensor
            The input of the network.

        Return
        ------
        output : Tensor
            The predicted mask.
        """

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
