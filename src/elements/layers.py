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


class UpsampleModule(nn.Module):
    """Convolutional decoder.

    If `method` is 'deconv' apply transposed convolutions with stride 2,
    otherwise apply the `method` upsampling function and then smooth with a
    stride 1x1 convolution.

    Params:
    -------
    filters: list, where the first element is the number of filters of the initial
      MLP layer and the remaining elements are the number of filters of the
      upsampling layers.
    kernel_size: the size of the convolutional kernels. The same size will be
      used in all convolutions.
    activation: an activation function, applied to all layers but the last.
    dec_up_strides: list, the upsampling factors of each upsampling convolutional
      layer.
    enc_conv_shapes: list, the shapes of the input and of all the intermediate
      feature maps of the convolutional layers in the encoder.
    n_c: the number of output channels.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 activation,
                 dec_up_strides,
                 enc_conv_shapes,
                 n_c,
                 method='nn',
                 name='upsample_module'):
        super(UpsampleModule, self).__init__(name=name)

        assert len(filters) == len(dec_up_strides) + 1, (
                'The decoder\'s filters should contain one element more than the '
                'decoder\'s up stride list, but has %d elements instead of %d.\n'
                'Decoder filters: %s\nDecoder up strides: %s' %
                (len(filters), len(dec_up_strides) + 1, str(filters),
                 str(dec_up_strides)))

        self._filters = filters
        self._kernel_size = kernel_size
        self._activation = activation

        self._dec_up_strides = dec_up_strides
        self._enc_conv_shapes = enc_conv_shapes
        self._n_c = n_c
        if method == 'deconv':
            self._conv_layer = tf.layers.Conv2DTranspose
            self._method = method
        else:
            self._conv_layer = tf.layers.Conv2D
            self._method = getattr(tf.image.ResizeMethod, method.upper())
        self._method_str = method.capitalize()

    def _build(self, z, is_training=True, test_local_stats=True, use_bn=False):
        batch_norm_args = {
            'is_training': is_training,
            'test_local_stats': test_local_stats
        }

        method = self._method
        # Cycle over the encoder shapes backwards, to build a symmetrical decoder.
        enc_conv_shapes = self._enc_conv_shapes[::-1]
        strides = self._dec_up_strides
        # We store the heights and widths of the encoder feature maps that are
        # unique, i.e., the ones right after a layer with stride != 1. These will be
        # used as a target to potentially crop the upsampled feature maps.
        unique_hw = np.unique([(el[1], el[2]) for el in enc_conv_shapes], axis=0)
        unique_hw = unique_hw.tolist()[::-1]
        unique_hw.pop()  # Drop the initial shape

        # The first filter is an MLP.
        mlp_filter, conv_filters = self._filters[0], self._filters[1:]
        # The first shape is used after the MLP to go to 4D.

        layers = [z]
        # The shape of the first enc is used after the MLP to go back to 4D.
        dec_mlp = snt.nets.MLP(
            name='dec_mlp_projection',
            output_sizes=[mlp_filter, np.prod(enc_conv_shapes[0][1:])],
            use_bias=not use_bn,
            activation=self._activation,
            activate_final=True)

        upsample_mlp_flat = dec_mlp(z)
        if use_bn:
            upsample_mlp_flat = snt.BatchNorm(scale=True)(upsample_mlp_flat,
                                                          **batch_norm_args)
        layers.append(upsample_mlp_flat)
        upsample = tf.reshape(upsample_mlp_flat, enc_conv_shapes[0])
        layers.append(upsample)

        for i, (filter_i, stride_i) in enumerate(zip(conv_filters, strides), 1):
            if method != 'deconv' and stride_i > 1:
                upsample = tf.image.resize_images(
                    upsample, [stride_i * el for el in upsample.shape.as_list()[1:3]],
                    method=method,
                    name='upsample_' + str(i))
            upsample = self._conv_layer(
                filters=filter_i,
                kernel_size=self._kernel_size,
                padding='same',
                use_bias=not use_bn,
                activation=self._activation,
                strides=stride_i if method == 'deconv' else 1,
                name='upsample_conv_' + str(i))(
                upsample)
            if use_bn:
                upsample = snt.BatchNorm(scale=True)(upsample, **batch_norm_args)
            if stride_i > 1:
                hw = unique_hw.pop()
                upsample = utils.maybe_center_crop(upsample, hw)
            layers.append(upsample)

        # Final layer, no upsampling.
        x_logits = nn.Conv2d(
            filters=self._n_c,
            kernel_size=self._kernel_size,
            padding='same',
            use_bias=not use_bn,
            activation=None,
            strides=1,
            name='logits')(
            upsample)
        if use_bn:
            x_logits = snt.BatchNorm(scale=True)(x_logits, **batch_norm_args)
        layers.append(x_logits)

        return x_logits

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

