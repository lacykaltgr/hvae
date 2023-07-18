import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

from src.utils import get_same_padding, get_valid_padding, get_causal_padding


class ResidualStack(nn.Module):
    """A stack of ResNet V2 blocks."""

    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 filter_size=3,
                 initializers=None,
                 data_format='NCHW',
                 activation=nn.ReLU(),
                 name='residual_stack'):
        super(ResidualStack, self).__init__()
        self.name = name
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.filter_size = filter_size
        self.initializers = initializers
        self.data_format = data_format
        self.activation = activation

    def forward(self, h):
        for i in range(self.num_residual_layers):
            h_i = self.activation(h)

            h_i = nn.Conv2d(
                in_channels=h_i.shape[1],
                out_channels=self.num_residual_hiddens,
                kernel_size=self.filter_size,
                stride=1,
                padding=self.filter_size // 2,
                bias=False)(
                h_i)
            h_i = self.activation(h_i)

            h_i = nn.Conv2d(
                in_channels=h_i.shape[1],
                out_channels=self.num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False)(
                h_i)
            h = h + h_i
        return self.activation(h)


class SharedConvModule(nn.Module):
    """Convolutional decoder."""

    def __init__(self,
                 filters,
                 kernel_size,
                 activation,
                 strides,
                 name='shared_conv_encoder'):
        super(SharedConvModule, self).__init__()
        self.name = name
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        assert len(strides) == len(filters) - 1
        self.conv_shapes = None

    def forward(self, x):
        self.conv_shapes = [list(x.shape)]  # Needed by deconv module
        conv = x
        for i, (filter_i, stride_i) in enumerate(zip(self.filters, self.strides), 1):
            conv = nn.Conv2d(
                in_channels=conv.shape[1],
                out_channels=filter_i,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=stride_i,
                bias=False)(conv)
            conv = self.activation(conv)
            self.conv_shapes.append(list(conv.shape))
        conv_flat = conv.view(conv.size(0), -1)

        enc_mlp = nn.Sequential(
            nn.Linear(conv_flat.size(1), self.filters[-1]),
            self.activation)
        h = enc_mlp(conv_flat)

        print('Shared conv module layer shapes:')
        for shape in self.conv_shapes:
            print(shape)
        print(h.shape)

        return h


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


