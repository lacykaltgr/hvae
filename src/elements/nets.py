from src.elements.layers import *
from hparams import get_hparams, Hyperparams
from src.utils import SerializableModule, SerializableSequential as Sequential


def get_net(model) -> Sequential:
    params = get_hparams()

    if model is None:
        return Sequential()

    elif isinstance(model, str):
        # Load model from default
        if model == 'mlp':
            return Sequential(MLPNet.from_hparams(params.mlp_params))
        elif model == 'conv':
            return Sequential(ConvNet.from_hparams(params.cnn_params))
        else:
            raise NotImplementedError("Model type not supported.")
    elif isinstance(model, Hyperparams):
        # Load model from hyperparameter config
        if "type" not in model.keys():
            raise ValueError("Model type not specified.")
        if model.type == 'mlp':
            return Sequential(MLPNet.from_hparams(model))
        elif model.type == 'conv':
            return Sequential(ConvNet.from_hparams(model))
        else:
            raise NotImplementedError("Model type not supported.")

    elif isinstance(model, SerializableModule):
        # Load model from nn.Module
        return Sequential(model)

    elif isinstance(model, Sequential):
        return model

    elif isinstance(model, list):
        # Load model from list
        return Sequential(*list(map(get_net, model)))

    else:
        raise NotImplementedError("Model type not supported.")


class MLPNet(SerializableModule):
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

    @staticmethod
    def from_hparams(hparams: Hyperparams):
        return MLPNet(
            input_size=hparams.input_size,
            hidden_sizes=hparams.hidden_sizes,
            output_size=hparams.output_size,
            activation=hparams.activation,
            residual=hparams.residual
        )

    def serialize(self):
        return dict(
            type=self.__class__.__name__,
            state_dict=self.state_dict(),
            params=dict(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=self.output_size,
                activation=self.activation,
                residual=self.residual
            )
        )

    @staticmethod
    def deserialize(serialized):
        net = MLPNet(serialized["params"])
        net.load_state_dict(serialized["state_dict"])



#TODO: pool parameterezhető legyen
class ConvNet(SerializableModule):
    def __init__(self, n_layers, in_filters, bottleneck_ratio, kernel_size, init_scaler
                 , residual=True, use_1x1=True, pool_strides=0, unpool_strides=0, output_ratio=1.0, activation=nn.SiLU()):
        super(ConvNet, self).__init__()

        self.residual = residual
        self.output_ratio = output_ratio
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if self.residual:
            assert self.output_ratio == 1

        output_filters = int(in_filters * output_ratio)
        bottlneck_filters = int(in_filters * bottleneck_ratio)

        convs = []
        if pool_strides > 0:
            convs.append(PoolLayer(
                in_filters=in_filters,
                filters=output_filters,
                strides=pool_strides,
            ))

        convs += [activation,
                  nn.Conv2d(in_channels=in_filters,
                            out_channels=bottlneck_filters,
                            kernel_size=(1, 1) if use_1x1 else kernel_size,
                            stride=(1, 1),
                            padding='same')]

        for _ in range(n_layers):
            convs.append(activation)
            convs.append(Conv2d(in_channels=bottlneck_filters,
                                out_channels=bottlneck_filters,
                                kernel_size=kernel_size,
                                stride=(1, 1),
                                padding='same'))

        convs += [activation,
                  Conv2d(in_channels=bottlneck_filters,
                         out_channels=output_filters,
                         kernel_size=(1, 1) if use_1x1 else kernel_size,
                         stride=(1, 1),
                         padding='same')]

        convs[-1].weight.data *= init_scaler

        if unpool_strides > 0:
            convs.append(UnpooLayer(
                in_filters=output_filters,
                filters=output_filters,
                strides=unpool_strides,
            ))

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

    @staticmethod
    def from_hparams(hparams):
        return ConvNet(
            n_layers=hparams.n_layers,
            in_filters=hparams.in_filters,
            bottleneck_ratio=hparams.bottleneck_ratio,
            output_ratio=hparams.output_ratio,
            kernel_size=hparams.kernel_size,
            init_scaler=hparams.init_scaler,
            use_1x1=hparams.use_1x1,
            pool_strides=hparams.pool_strides,
            unpool_strides=hparams.unpool_strides,
            activation=hparams.activation,
            residual=hparams.residual,
        )

    def serialize(self):
        return dict(
            type=self.__class__.__name__,
            state_dict=self.state_dict(),
            params=dict(
                n_layers=self.n_layers,
                in_filters=self.in_filters,
                bottleneck_ratio=self.bottleneck_ratio,
                output_ratio=self.output_ratio,
                kernel_size=self.kernel_size,
                init_scaler=self.init_scaler,
                use_1x1=self.use_1x1,
                pool_strides=self.pool_strides,
                unpool_strides=self.unpool_strides,
                activation=self.activation,
                residual=self.residual,
            )
        )

    @staticmethod
    def deserialize(serialized):
        net = ConvNet(serialized["params"])
        net.load_state_dict(serialized["state_dict"])
        return net
