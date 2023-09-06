from collections import OrderedDict
from torch import tensor

from src.elements.layers import *
from src.hparams import get_hparams, Hyperparams
from src.utils import SerializableModule, SerializableSequential as Sequential


def get_net(model) -> Sequential:
    """
    Get net from
    -string model type,
    -hyperparameter config
    -SerializableModule or SerializableSequential object
    -or list of the above

    :param model: str, Hyperparams, SerializableModule, SerializableSequential, list
    :return: SerializableSequential
    """

    if model is None:
        return Sequential()

    # Load model from default hyperparameter config
    elif isinstance(model, str):
        params = get_hparams()
        if model == 'mlp':
            return Sequential(MLPNet.from_hparams(params.mlp_params))
        elif model == 'conv':
            return Sequential(ConvNet.from_hparams(params.cnn_params))
        else:
            raise NotImplementedError("Model type not supported.")


    # Load model from hyperparameter config
    elif isinstance(model, Hyperparams):

        if "type" not in model.keys():
            raise ValueError("Model type not specified.")
        if model.type == 'mlp':
            return Sequential(MLPNet.from_hparams(model))
        elif model.type == 'conv':
            return Sequential(ConvNet.from_hparams(model))
        else:
            raise NotImplementedError("Model type not supported.")

    elif isinstance(model, BlockNet):
        return Sequential(model)


    # Load model from SerializableModule
    elif isinstance(model, SerializableModule):
        return Sequential(model)


    # Load model from SerializableSequential
    elif isinstance(model, Sequential):
        return model


    # Load model from list of any of the above
    elif isinstance(model, list):
        # Load model from list
        return Sequential(*list(map(get_net, model)))

    else:
        raise NotImplementedError("Model type not supported.")


class MLPNet(SerializableModule):

    """
    Parametric multilayer perceptron network

    :param input_size: int, the size of the input
    :param hidden_sizes: list of int, the sizes of the hidden layers
    :param output_size: int, the size of the output
    :param residual: bool, whether to use residual connections
    :param activation: torch.nn.Module, the activation function to use
    """
    def __init__(self, input_size, hidden_sizes, output_size, residual=False, activation=nn.ReLU(), activate_output=True):
        super(MLPNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.residual = residual
        self.activate_output = activate_output

        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or self.activate_output:
                layers.append(self.activation)

        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs
        x = self.mlp_layers(x)
        outputs = x if not self.residual else inputs + x
        return outputs


    @staticmethod
    def from_hparams(hparams: Hyperparams):
        return MLPNet(
            input_size=hparams.input_size,
            hidden_sizes=hparams.hidden_sizes,
            output_size=hparams.output_size,
            activation=hparams.activation,
            residual=hparams.residual,
            activate_output=hparams.activate_output
        )

    def serialize(self):
        return dict(
            type=self.__class__,
            state_dict=self.state_dict(),
            params=dict(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=self.output_size,
                activation=self.activation,
                residual=self.residual,
                activate_output=self.activate_output
            )
        )

    @staticmethod
    def deserialize(serialized):
        net = MLPNet(**serialized["params"])
        net.load_state_dict(serialized["state_dict"])
        return net


class ConvNet(SerializableModule):
    """
    Parametric convolutional network
    based on Efficient-VDVAE paper

    :param n_layers: int, the number of convolutional layers
    :param in_filters: int, the number of input filters
    :param bottleneck_ratio: float, the ratio of bottleneck filters to input filters
    :param kernel_size: int or tuple of int, the size of the convolutional kernel
    :param init_scaler: float, the scaler for the initial weights_imagenet
    :param residual: bool, whether to use residual connections
    :param use_1x1: bool, whether to use 1x1 convolutions
    :param pool_strides: int or tuple of int, the strides for the pooling layers
    :param unpool_strides: int or tuple of int, the strides for the unpooling layers
    :param output_ratio: float, the ratio of output filters to input filters
    :param activation: torch.nn.Module, the activation function to use
    """
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
            type=self.__class__,
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
        net = ConvNet(**serialized["params"])
        net.load_state_dict(serialized["state_dict"])
        return net


class BlockNet(SerializableModule):

    def __init__(self, **blocks):
        from src.hvae.block import InputBlock
        super(BlockNet, self).__init__()

        self.input_block, output = next(((block, output) for output, block in blocks.items()
                                         if isinstance(block, InputBlock)), None)
        self.input_block.set_output(output)
        self.output_block = next((block for _, block in blocks.items()
                                  if isinstance(block, self.OutputBlock)), None)

        self.blocks = nn.ModuleDict()
        for output, block in blocks.items():
            if not isinstance(block, (InputBlock, self.OutputBlock)):
                block.set_output(output)
                self.blocks.update({output: block})

    def forward(self, inputs):
        computed = self.input_block(inputs)
        computed = self.propogate_blocks(computed)
        output = self.output_block(computed)
        return output

    def propogate_blocks(self, computed):
        for block in self.blocks.values():
            output = block(computed=computed)
            if isinstance(output, tuple):
                computed, _ = output
            else:
                computed = output
        return computed

    def serialize(self):
        blocks = list()
        blocks.append(self.input_block.serialize())
        for block in self.blocks.values():
            blocks.append(block.serialize())
        blocks.append(self.output_block.serialize())
        return dict(
            type=self.__class__,
            blocks=blocks
        )

    @staticmethod
    def deserialize(serialized):
        blocks = OrderedDict()
        for block in serialized["blocks"]:
            blocks[block["output"]] = block["type"].deserialize(block)
        return BlockNet(**blocks)

    class OutputBlock(SerializableModule):
        """
        Final block of the model
        Functions like a SimpleBlock
        Only for use in BlockNet
        """
        def __init__(self, input_id: str):
            super(BlockNet.OutputBlock, self).__init__()
            self.input = input_id

        def forward(self, computed: dict) -> (tensor, dict, tuple):
            output = computed[self.input]
            return output

        def serialize(self):
            return dict(
                type=self.__class__,
                input=self.input,
                output="output"
            )

        @staticmethod
        def deserialize(serialized: dict):
            return BlockNet.OutputBlock(input_id=serialized["input"])




