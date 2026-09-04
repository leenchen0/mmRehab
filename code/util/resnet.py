import math
from functools import partial

import torch
import torch.nn as nn


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def conv_bn_1D(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm1d(out_channels))


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
conv5x5 = partial(Conv2dAuto, kernel_size=5, bias=False)
conv7x7 = partial(Conv2dAuto, kernel_size=7, bias=False)


class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = ((self.kernel_size[0]) // 2,)


conv3 = partial(Conv1dAuto, kernel_size=3, bias=False)
conv5 = partial(Conv1dAuto, kernel_size=5, bias=False)
conv7 = partial(Conv1dAuto, kernel_size=7, bias=False)
conv11 = partial(Conv1dAuto, kernel_size=11, bias=False)


class DeConv1dAuto(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2,)  # dynamic add padding based on the kernel_size
        self.output_padding = (self.stride[0] - 1,)


deconv3 = partial(DeConv1dAuto, kernel_size=3, bias=False)
deconv5 = partial(DeConv1dAuto, kernel_size=5, bias=False)
deconv7 = partial(DeConv1dAuto, kernel_size=7, bias=False)
deconv11 = partial(DeConv1dAuto, kernel_size=11, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', downsampling=1):
        super().__init__()
        self.in_channels, self.out_channels, self.activation, self.downsampling = in_channels, out_channels, activation, downsampling
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels or self.downsampling != 1


class ResNetResidualBlock1D(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3, *args, **kwargs):
        super().__init__(in_channels, out_channels, downsampling=downsampling, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        if self.downsampling < 1:
            upsampling = math.floor(1 / self.downsampling)
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(self.in_channels, self.expanded_channels, kernel_size=(1,),
                                   stride=(upsampling,), output_padding=(upsampling - 1,), bias=False),
                nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=(1,),
                          stride=(self.downsampling,), bias=False),
                nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels or self.downsampling != 1


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock1D(ResNetResidualBlock1D):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if self.downsampling < 1:
            upsampling = math.floor(1 / self.downsampling)
            self.blocks = nn.Sequential(
                conv_bn_1D(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=upsampling),
                activation_func(self.activation),
                conv_bn_1D(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
            )
        else:
            self.blocks = nn.Sequential(
                conv_bn_1D(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
                activation_func(self.activation),
                conv_bn_1D(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
            )


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """

    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, downsampling=None, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        if downsampling is None:
            downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x
