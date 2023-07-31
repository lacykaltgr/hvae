import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

from src.utils import get_same_padding, get_valid_padding, get_causal_padding
from hparams import *

class Interpolate(nn.Module):
    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Unpoolayer(nn.Module):
    def __init__(self, in_filters, filters, strides):
        super(Unpoolayer, self).__init__()
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


class ResidualConvCell(nn.Module):
    def __init__(self, n_layers, in_filters, bottleneck_ratio, kernel_size, init_scaler
                 , residual=True, use_1x1=True, output_ratio=1.0):
        super(ResidualConvCell, self).__init__()

        self.residual = residual
        self.output_ratio = output_ratio
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if self.residual:
            assert self.output_ratio == 1

        output_filters = int(in_filters * output_ratio)
        bottlneck_filters = int(in_filters * bottleneck_ratio)

        convs = [nn.SiLU(inplace=False),
                 nn.Conv2d(in_channels=in_filters,
                           out_channels=bottlneck_filters,
                           kernel_size=(1, 1) if use_1x1 else kernel_size,
                           stride=(1, 1),
                           padding='same')]

        for _ in range(n_layers):
            convs.append(nn.SiLU(inplace=False))
            convs.append(Conv2d(in_channels=bottlneck_filters,
                                out_channels=bottlneck_filters,
                                kernel_size=kernel_size,
                                stride=(1, 1),
                                padding='same'))

        convs += [nn.SiLU(inplace=False),
                  Conv2d(in_channels=bottlneck_filters,
                         out_channels=output_filters,
                         kernel_size=(1, 1) if use_1x1 else kernel_size,
                         stride=(1, 1),
                         padding='same')]

        convs[-1].weight.data *= init_scaler

        self.convs = nn.Sequential(*convs)

    def forward(self, inputs):
        x = inputs
        x = self.convs(x)

        if self.residual:
            outputs = inputs + x
        else:
            outputs = x
        return outputs


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, *args, **kwargs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2

        self.stride = stride

        self.padding_str = padding.upper()
        if self.padding_str == 'SAME':
            self.pad_values = get_same_padding(kernel_size, stride, dilation)

        elif self.padding_str == 'VALID':
            self.pad_values = get_valid_padding()

        elif self.padding_str == 'CAUSAL':
            self.pad_values = get_causal_padding(kernel_size, stride, dilation)

        else:
            raise ValueError

        self.condition = np.sum(self.pad_values) != 0
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, *args, **kwargs)

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        if self.condition:
            x = F.pad(x, self.pad_values)
        x = super(Conv2d, self).forward(x)
        return x


class DmolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_conv = get_conv(H.width, H.num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), low_bit=self.H.dataset in ['ffhq_256'])

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        xhat = (im + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat


class GaussianLatentLayer(nn.Module):
    def __init__(self, in_filters=0, num_variates=0, min_std=np.exp(-2)):
        super(GaussianLatentLayer, self).__init__()

        self.projection = Conv2d(
            in_channels=in_filters,
            out_channels=num_variates * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )

        self.softplus = torch.nn.Softplus(beta=loss_params.gradient_smoothing_beta)

    def forward(self, mean, std, temperature=None, prior_stats=None, projection=None):
        if projection is not None:
            x = torch.cat([mean, std], dim=1)
            x = self.projection(x)
            mean, std = torch.chunk(x, chunks=2, dim=1)

        if loss_params.distribution_base == 'std':
            std = self.softplus(std)
        elif loss_params.distribution_base == 'logstd':
            std = torch.exp(loss_params.gradient_smoothing_beta * std)

        if temperature is not None:
            std = std * temperature

        return self.calculate_z(mean, std)

    @staticmethod
    @torch.jit.script
    def calculate_z(mean, std):
        eps = torch.empty_like(mean, device=torch.device('cuda')).normal_(0., 1.)
        z = eps * std + mean
        return z
