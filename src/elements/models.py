from layers import *
from torch import nn
from hparams import Hyperparams


# TODO
def get_model(model):
    if model is None:
        return None
    elif isinstance(model, str):
        # Load model from default
        pass
    elif isinstance(model, Hyperparams):
        # Load model from hyperparameter config
        pass
    elif isinstance(model, nn.Module):
        # Load model from nn.Module
        return model
    elif isinstance(model, dict):
        # Load model from dictionary
        pass
    elif isinstance(model, list):
        # Load model from list
        pass
    else:
        raise NotImplementedError("Model type not supported.")


class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, residual=False, activation=nn.ReLU()):
        super(MLPNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.residual = residual

        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(self.activation)

        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs
        x = self.mlp_layers(x)
        if self.residual:
            outputs = inputs + x
        else:
            outputs = x
        return outputs


class ConvNet(nn.Module):
    def __init__(self, n_layers, in_filters, bottleneck_ratio, kernel_size, init_scaler
                 , residual=True, use_1x1=True, output_ratio=1.0, activation=None):
        super(ConvNet, self).__init__()

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
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = self.convs(x)
        if self.residual:
            outputs = inputs + x
        else:
            outputs = x
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


