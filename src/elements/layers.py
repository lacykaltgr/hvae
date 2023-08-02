import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from hparams import *


class Interpolate(nn.Module):
    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class UnpooLayer(nn.Module):
    def __init__(self, in_filters, filters, strides):
        super(UnpooLayer, self).__init__()
        self.scale_bias = None
        self.filters = filters

        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = strides

        ops = [Conv2d(in_channels=in_filters, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),
                      padding='same'),
               nn.LeakyReLU(negative_slope=0.1),
               Interpolate(scale=self.strides)]

        self.register_parameter('scale_bias', None)
        self.ops = nn.Sequential(*ops)

    def reset_parameters(self, inputs):
        B, C, H, W = inputs.shape
        self.scale_bias = nn.Parameter(torch.zeros(size=(1, C, H, W), device='cuda'), requires_grad=True)

    def forward(self, x):
        x = self.ops(x)
        if self.scale_bias is None:
            self.reset_parameters(x)
        x = x + self.scale_bias
        return x


class PoolLayer(nn.Module):
    def __init__(self, in_filters, filters, strides):
        super(PoolLayer, self).__init__()
        self.filters = filters

        if isinstance(strides, int):
            strides = (strides, strides)

        ops = [Conv2d(in_channels=in_filters, out_channels=filters,
                      kernel_size=strides, stride=strides, padding='same'),
               nn.LeakyReLU(negative_slope=0.1)]

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2

        self.stride = stride

        self.condition = np.sum(self.pad_values) != 0
        super(Conv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        if self.condition:
            x = F.pad(x, self.pad_values)
        x = super(Conv2d, self).forward(x)
        return x


class GaussianLatentLayer(nn.Module):
    def __init__(self, in_filters=0, num_variates=0, min_std=np.exp(-2)):
        super(GaussianLatentLayer, self).__init__()

        self.projection = torch.nn.Conv2d(
            in_channels=in_filters,
            out_channels=num_variates * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )

        self.softplus = torch.nn.Softplus(beta=model_params.gradient_smoothing_beta)

    def forward(self, mean, std, temperature=None, prior_stats=None, projection=None):
        if projection is not None:
            x = torch.cat([mean, std], dim=1)
            x = self.projection(x)
            mean, std = torch.chunk(x, chunks=2, dim=1)

        if model_params.distribution_base == 'std':
            std = self.softplus(std)
        elif model_params.distribution_base == 'logstd':
            std = torch.exp(model_params.gradient_smoothing_beta * std)

        #TODO: var vagy std különbség? gyököt vonni vagy nem?

        if temperature is not None:
            std = std * temperature

        return self.calculate_z(mean, std)

    @staticmethod
    @torch.jit.script
    def calculate_z(mean, std):
        #TODO: más eloszlások esetleg?
        eps = torch.empty_like(mean, device=torch.device('cuda')).normal_(0., 1.)
        z = eps * std + mean
        return z
