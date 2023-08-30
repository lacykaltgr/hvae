import os
from torch import nn
import numpy as np

import torch
from .read_hparams.hparams import HParams
from src.utils import SerializableSequential as Sequential
from src.elements.nets import MLPNet, ConvNet
from src.elements.layers import PoolLayer, UnpooLayer
from checkpoint import Checkpoint


class EfficientVDVAEMigrationAgent:
    def __init__(self,
                 path="migration/EfficientVDVAE_migration/weights_imagenet/",
                 weights_filename="checkpoints-imagenet32_baseline",
                 config_filename="hparams-imagenet32_baseline"):

        self.hparams = HParams(path,
                               hparams_filename=config_filename,
                               name="migration")

        weights_file = os.path.join(path, weights_filename)
        checkpoint = torch.load(weights_file, map_location=torch.device('cpu'))

        levels_up, levels_up_downsample, input_conv, skip_projections = self.build_bottom_up()
        levels_down, levels_down_upsample, output_conv, trainable_h = self.build_top_down()

        levels_up_weights, levels_up_downsample_weights, levels_down_weights, \
            levels_down_upsample_weights, skip_projections_weights, input_conv_weights, \
            output_conv_weights, trainable_h_weights = self.process_checkpoint(checkpoint['model_state_dict'])

        #for i in range(len(checkpoint["model_state_dict"].keys())):
        #    print(list(checkpoint["model_state_dict"].values())[i].shape)
        #return


        global_step = checkpoint['global_step']
        optimizer_state = checkpoint['optimizer_state_dict']
        scheduler_state = checkpoint['scheduler_state_dict']

        levels_up_i = 0
        for level in levels_up:
            for net in level.modules():
                for layer, weight_dict in zip(net.convs.modules(), levels_up_weights[levels_up_i]):
                    layer.weight.copy_(torch.tensor(weight_dict['w']).T)
                    layer.bias.copy_(torch.tensor(weight_dict['b']))
                levels_up_i += 1

        levels_up_downsample_i = 0
        for net in levels_up_downsample:
            for layer, weight_dict in zip(net.modules(), levels_up_downsample_weights[levels_up_downsample_i]):
                layer.weight.copy_(torch.tensor(weight_dict['w']).T)
                layer.bias.copy_(torch.tensor(weight_dict['b']))
            levels_up_downsample_i += 1

        levels_down_i = 0
        for level in levels_down:
            for step in level:
                for net_name, net in step:
                    for layer, weight_dict in zip(net.modules(), levels_down_weights[levels_down_i][net_name]):
                        layer.weight.copy_(torch.tensor(weight_dict['w']).T)
                        layer.bias.copy_(torch.tensor(weight_dict['b']))
                levels_down_i += 1






    def __getitem__(self, item):
        pass

    def get_global_step(self):
        return self.global_step

    def get_optimizer(self, optimizer):
        return optimizer

    def get_schedule(self, schedule):
        return schedule

    def _find_net(self, net_name, i, split_element, keys, values):
        in_block = []
        while keys[i].split(".")[split_element] == net_name:
            in_block.append(dict(
                w=values[i],
                b=values[i+1])
            )
            i += 2
        return in_block, i

    def process_checkpoint(self, state):
        keys = list(state.keys())
        values = list(state.values())

        levels_up_weights = []
        levels_up_downsample_weights = []
        levels_down_weights = []
        levels_down_upsample_weights = []
        skip_projections_weights = []
        input_conv_weights = None
        output_conv_weights = None
        trainable_h_weights = None

        top_down_block = dict()
        top_down_number = 3

        i = 0
        while i < len(keys):
            split = keys[i].split(".")  # bottom_up, levels_up_downsample 0, residual_block, 0, convs, 1, weight
            if split[0] == "bottom_up":
                if split[1] == "levels_up_downsample":

                    if split[3] == "residual_block":
                        net, i = self._find_net("residual_block", i, 3, keys, values)
                        levels_up_downsample_weights.append(net)
                    elif split[3] == "skip_projection":
                        skip_projections_weights.append(dict(
                            w=values[i],
                            b=values[i+1]
                        ))
                    else: raise NotImplementedError(f'Variable {keys[i]} not implemented.')

                elif split[1] == "levels_up":

                    if split[3] == "residual_block":
                        net, i = self._find_net("residual_block", i, 3, keys, values)
                        levels_up_weights.append(net)
                    else:raise NotImplementedError(f'Variable {keys[i]} not implemented.')

                elif split[0] == "input_conv":
                    input_conv_weights = dict(
                        w=values[i],
                        b=values[i+1]
                    )

                else:raise NotImplementedError(f'Variable {keys[i]} not implemented.')

            elif split[0] == "top_down":
                if split[1] == 'levels_down_upsample':
                    top_down_number = 3
                elif split[1] == 'levels_down':
                    top_down_number = 4

                if split[0] == "trainable_h":
                    trainable_h_weights = dict(
                        w=values[i],
                        b=values[i+1]
                    )
                elif split[1] == "output_conv":
                    output_conv_weights = dict(
                        w=values[i],
                        b=values[i+1]
                    )
                elif split[top_down_number] == 'residual_block':
                    net, i = self._find_net("residual_block", i, top_down_number, keys, values)
                    top_down_block = dict(residual_block=net)
                elif split[top_down_number] == 'prior_net':
                    net, i = self._find_net("prior_net", i, top_down_number, keys, values)
                    top_down_block['prior_net'] = net
                elif split[top_down_number] == 'posterior_net':
                    net, i = self._find_net("posterior_net", i, top_down_number, keys, values)
                    top_down_block['posterior_net'] = net
                elif split[top_down_number] == 'prior_layer':
                    top_down_block['prior_layer'] = dict(
                        w=values[i],
                        b=values[i+1]
                    )
                    i += 2
                elif split[top_down_number] == 'posterior_layer':
                    top_down_block['posterior_layer'] = dict(
                        w=values[i],
                        b=values[i+1]
                    )
                    i += 2
                elif split[top_down_number] == 'z_projection':
                    top_down_block['posterior_layer'] = dict(
                        w=values[i],
                        b=values[i+1]
                    )
                    i += 2
                    if split[1] == 'levels_down_upsample':
                        levels_down_upsample_weights.append(top_down_block)
                    elif split[1] == 'levels_down':
                        levels_down_weights.append(top_down_block)
                    else:
                        raise NotImplementedError(f'Variable {keys[i]} not implemented.')
                else:
                    raise NotImplementedError(f'Variable {keys[i]} not implemented.')

            else: raise NotImplementedError(f'Variable {keys[i]} not implemented.')

        return levels_up_weights, levels_up_downsample_weights, levels_down_weights, levels_down_upsample_weights, \
            skip_projections_weights, input_conv_weights, output_conv_weights, trainable_h_weights

    def build_bottom_up(self):
        in_channels_up = [self.hparams.model.input_conv_filters] + self.hparams.model.up_filters[0:-1]

        levels_up = nn.ModuleList([])
        levels_up_downsample = nn.ModuleList([])
        skip_projections = nn.ModuleList([])

        for i, stride in enumerate(self.hparams.model.up_strides):
            elements = nn.ModuleList([])
            for j in range(self.hparams.model.up_n_blocks_per_res[i]):
                elements.append(Sequential(*[ConvNet(
                    n_layers=self.hparams.model.up_n_layers[i],
                    in_filters=in_channels_up[i],
                    bottleneck_ratio=self.hparams.model.up_mid_filters_ratio[i],
                    kernel_size=self.hparams.model.up_kernel_size[i],
                    init_scaler=np.sqrt(1. / float(sum(self.hparams.model.down_n_blocks_per_res) +
                                                   len(self.hparams.model.down_strides)))
                                                    if self.hparams.model.stable_init else 1.,
                    activation=nn.SiLU(),
                    output_ratio=1.0,
                    residual=True,
                    use_1x1=self.hparams.model.use_1x1_conv,
                    pool_strides=0,
                    unpool_strides=0,
                ) for _ in range(self.hparams.up_n_blocks[i])]))

            levels_up.extend([elements])

            levels_up_downsample.extend([Sequential(*[
                Sequential(*[
                    ConvNet(
                        n_layers=self.hparams.model.up_n_layers[i],
                        in_filters=in_channels_up[i],
                        bottleneck_ratio=self.hparams.model.up_mid_filters_ratio[i],
                        kernel_size=self.hparams.model.up_kernel_size[i],
                        init_scaler=np.sqrt(1. / float(sum(self.hparams.model.down_n_blocks_per_res) +
                                                       len(self.hparams.model.down_strides)))
                                                        if self.hparams.model.stable_init else 1.,
                        use_1x1=self.hparams.model.use_1x1_conv,
                    ),
                    PoolLayer(
                        in_filters=in_channels_up[i],
                        filters=self.hparams.model.up_filters[i],
                        strides=stride
                    )
                ]) for _ in range(self.hparams.model.up_n_blocks[i])])]),

            skip_projections.extend([nn.Conv2d(
                in_channels=in_channels_up[i], out_channels=self.hparams.model.up_skip_filters[i], kernel_size=(1, 1), stride=(1, 1),
                padding='same'
            )])

        input_conv = nn.Conv2d(in_channels=self.hparams.data.channels,
                               out_channels=self.hparams.model.input_conv_filters,
                               kernel_size=self.hparams.model.input_kernel_size,
                               stride=(1, 1),
                               padding='same')

        return levels_up, levels_up_downsample, input_conv, skip_projections

    def build_top_down(self):

        H = W = self.hparams.data.target_res // np.prod(self.hparams.model.down_strides)

        trainable_h = torch.nn.Parameter(data=torch.empty(size=(1, self.hparams.model.down_filters[0], H, W)),
                                         requires_grad=True)
        nn.init.kaiming_uniform_(trainable_h, nonlinearity='linear')

        in_channels_down = [self.hparams.model.down_filters[0]] + self.hparams.model.down_filters[0:-1]
        levels_down, levels_down_upsample = [], []

        for i, stride in enumerate(self.hparams.model.down_strides):
            levels_down_upsample.extend([self.build_top_down_level(
                n_blocks=self.hparams.model.down_n_blocks[i],
                n_layers=self.hparams.model.down_n_layers[i],
                in_filters=in_channels_down[i],
                filters=self.hparams.model.down_filters[i],
                bottleneck_ratio=self.hparams.model.down_mid_filters_ratio[i],
                kernel_size=self.hparams.model.down_kernel_size[i],
                strides=stride,
                skip_filters=self.hparams.model.up_skip_filters[::-1][i],
                latent_variates=self.hparams.model.down_latent_variates[i],
                first_block=i == 0,
                last_block=False
            )])

            levels_down.extend([
                [self.build_top_down_level(
                    n_blocks=self.hparams.model.down_n_blocks[i],
                    n_layers=self.hparams.model.down_n_layers[i],
                    in_filters=self.hparams.model.down_filters[i],
                    filters=self.hparams.model.down_filters[i],
                    bottleneck_ratio=self.hparams.model.down_mid_filters_ratio[i],
                    kernel_size=self.hparams.model.down_kernel_size[i],
                    strides=1,
                    skip_filters=self.hparams.model.up_skip_filters[::-1][i],
                    latent_variates=self.hparams.model.down_latent_variates[i],
                    first_block=False,
                    last_block=i == len(self.hparams.model.down_strides) - 1 and j ==
                               self.hparams.model.down_n_blocks_per_res[i] - 1)
                    for j in range(self.hparams.model.down_n_blocks_per_res[i])]])

        output_conv = nn.Conv2d(in_channels=self.hparams.model.down_filters[-1],
                                out_channels=1 if self.hparams.data.dataset_source == 'binarized_mnist'
                                else self.hparams.model.num_output_mixtures * (3 * self.hparams.data.channels + 1),
                                kernel_size=self.hparams.model.output_kernel_size,
                                stride=(1, 1),
                                padding='same')

        return levels_down, levels_down_upsample, output_conv, trainable_h

    def build_top_down_level(self, n_blocks, n_layers, in_filters, filters, bottleneck_ratio, kernel_size,
                             strides, skip_filters, latent_variates, first_block, last_block) -> dict:

        assert not (first_block and last_block)

        if strides > 1:
            unpool = UnpooLayer(in_filters, filters, strides)
            in_filters = filters
        else:
            unpool = None

        residual_block = nn.Sequential(*[
            ConvNet(
                n_layers=n_layers,
                in_filters=in_filters,
                bottleneck_ratio=bottleneck_ratio,
                kernel_size=kernel_size,
                init_scaler=np.sqrt(1. / float(sum(self.hparams.model.down_n_blocks_per_res) +
                                               len(self.hparams.model.down_strides)))
                if self.hparams.model.stable_init else 1.,
                use_1x1=self.hparams.model.use_1x1_conv
            ) for _ in range(n_blocks)
        ])

        posterior_net = nn.Sequential(ConvNet(
            n_layers=n_layers,
            in_filters=in_filters + skip_filters,
            bottleneck_ratio=bottleneck_ratio * 0.5,
            kernel_size=kernel_size,
            init_scaler=1.,
            residual=False,
            use_1x1=self.hparams.model.use_1x1_conv,
            output_ratio=0.5  # Assuming skip_filters == in_filters
        ))

        prior_net = nn.Sequential(ConvNet(
            n_layers=n_layers,
            in_filters=in_filters,
            bottleneck_ratio=bottleneck_ratio,
            kernel_size=kernel_size,
            init_scaler=0. if self.hparams.model.initialize_prior_weights_as_zero else 1.,
            residual=False,
            use_1x1=self.hparams.model.use_1x1_conv,
            output_ratio=2.0
        ))

        prior_projection = nn.Conv2d(
                in_channels=in_filters,
                out_channels=latent_variates * 2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding='same'
        )

        posterior_projection = nn.Conv2d(
            in_channels=in_filters,
            out_channels=latent_variates * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )

        z_projection = nn.Conv2d(
            in_channels=latent_variates,
            out_channels=filters,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )
        z_projection.weight.data *= np.sqrt(
            1. / float(sum(self.hparams.model.down_n_blocks_per_res) + len(self.hparams.model.down_strides)))

        return dict(
            unpool=unpool,
            residual_block=residual_block,
            prior_net=prior_net,
            posterior_layer=posterior_net,
            z_projection=z_projection,
            prior_projection=prior_projection,
            posterior_projection=posterior_projection
        )
