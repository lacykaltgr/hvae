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

    # Load model from hyperparameter config
    elif isinstance(model, Hyperparams):

        if "type" not in model.keys():
            raise ValueError("Model type not specified.")
        if model.type == 'mlp':
            return Sequential(MLPNet.from_hparams(model))
        elif model.type == 'conv':
            return Sequential(ConvNet.from_hparams(model))
        elif model.type == 'pool':
            return Sequential(PoolLayer.from_hparams(model))
        elif model.type == 'unpool':
            return Sequential(UnPooLayer.from_hparams(model))
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

    :param in_filters: int, the number of input filters
    :param filters: list of int, the number of filters for each layer
    :param kernel_size: int or tuple of int, the size of the convolutional kernel
    :param pool_strides: int or tuple of int, the strides for the pooling layers
    :param unpool_strides: int or tuple of int, the strides for the unpooling layers
    :param activation: torch.nn.Module, the activation function to use
    """
    def __init__(self, in_filters, filters, kernel_size, pool_strides=0, unpool_strides=0,
                 activation=nn.LeakyReLU(), activate_output=False):
        super(ConvNet, self).__init__()
        self.in_filters = in_filters
        self.filters = filters
        self.kernel_size = kernel_size

        assert pool_strides == 0 or unpool_strides == 0, \
            "Cannot have both pooling and unpooling layers"
        self.pool_strides = pool_strides
        self.unpool_strides = unpool_strides

        self.activation = activation
        self.activate_output = activate_output

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        convs = []
        for i in range(self.pool_strides):
            convs.append(PoolLayer(
                in_filters=in_filters,
                filters=in_filters,
                strides=2,
                activation=activation
            ))

        for i in range(self.unpool_strides):
            convs.append(UnPooLayer(
                in_filters=in_filters,
                filters=in_filters,
                strides=2,
                activation=activation
            ))

        filters = [in_filters] + filters
        for i in range(len(filters) - 1):
            convs.append(nn.Conv2d(in_channels=filters[i],
                                   out_channels=filters[i + 1],
                                   kernel_size=kernel_size,
                                   stride=(1, 1),
                                   padding='same'))
            convs.append(nn.BatchNorm2d(filters[i + 1]))
            convs.append(activation)

        self.convs = nn.Sequential(*convs)

    def forward(self, inputs):
        x = inputs
        x = self.convs(x)
        if self.activate_output is not None:
            x = self.activation(x)
        return x

    @staticmethod
    def from_hparams(hparams):
        return ConvNet(
            in_filters=hparams.in_filters,
            filters=hparams.filters,
            kernel_size=hparams.kernel_size,
            pool_strides=hparams.pool_strides,
            unpool_strides=hparams.unpool_strides,
            activation=hparams.activation,
            activate_output=hparams.activate_output
        )

    def serialize(self):
        return dict(
            type=self.__class__,
            state_dict=self.state_dict(),
            params=dict(
                in_filters=self.in_filters,
                filters=self.filters,
                kernel_size=self.kernel_size,
                pool_strides=self.pool_strides,
                unpool_strides=self.unpool_strides,
                activation=self.activation,
                activate_output=self.activate_output
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
        def __init__(self, input_id):
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




