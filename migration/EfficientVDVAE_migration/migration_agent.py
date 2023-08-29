import torch
from src.utils import SerializableSequential as Sequential
from src.elements.nets import MLPNet
from checkpoint import Checkpoint


class EfficientVDVAEMigrationAgent:
    def __init__(self, path="migration/EfficientVDVAE_migration/weights/checkpoints-imagenet32_baseline"):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        for i in range(len(checkpoint["model_state_dict"].keys())):
            print(list(checkpoint["model_state_dict"].keys())[i])
        return

        state = checkpoint["model_state_dict"]
        keys = list(state.keys())
        values = list(state.values())
        levels_up = []
        levels_up_downsample = []
        levels_down = []
        levels_down_upsample = []
        i = 0
        while i < len(keys):
            split = keys[i].split(".") # bottom_up, levels_up_downsample 0, residual_block, 0, convs, 1, weight
            if split[0] == "bottom_up":
                if split[1] == "levels_up_downsample":
                    if split[3] == "residual_block":
                        while keys[i].split(".")[3] == "residual_block":
                            levels_up_downsample.extend(dict(
                                type="res_conv",

                            ))
                            i += 2
                    elif split[3] == "skip_projection":
                        pass

                elif split[1] == "levels_up":
                    if split[3] == "residual_block":
                        pass
                    elif split[3] == "skip_projection":
                        pass

                else:
                    raise NotImplementedError(f'Variable {name} not implemented.')

            elif split[0] == "top_down":
                if split[1] == "levels_down_upsample":
                    pass

                elif split[1] == "levels_down":
                    pass

                else:
                    raise NotImplementedError(f'Variable {name} not implemented.')
            elif split[0] == "input_conv":
                self.input_conv = None
            elif split[0] == "trainable_h":
                self.trainable_h = None
            else:
                raise NotImplementedError(f'Variable {name} not implemented.')

            self.n_downsample_ratio = len(levels_up)/len(levels_up_downsample)

            for j in range(hparams.model.up_n_blocks_per_res[i]):
                elements.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                              n_layers=hparams.model.up_n_layers[i],
                                              in_filters=in_channels_up[i],
                                              filters=hparams.model.up_filters[i],
                                              bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                              kernel_size=hparams.model.up_kernel_size[i],
                                              strides=1,
                                              skip_filters=hparams.model.up_skip_filters[i],
                                              use_skip=False)])

            self.levels_up.extend([elements])

            self.levels_up_downsample.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                                           n_layers=hparams.model.up_n_layers[i],
                                                           in_filters=in_channels_up[i],
                                                           filters=hparams.model.up_filters[i],
                                                           bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                                           kernel_size=hparams.model.up_kernel_size[i],
                                                           strides=stride,
                                                           skip_filters=hparams.model.up_skip_filters[i],
                                                           use_skip=True)])

        self.input_conv = Conv2d(in_channels=hparams.data.channels,
                                 out_channels=hparams.model.input_conv_filters,
                                 kernel_size=hparams.model.input_kernel_size,
                                 stride=(1, 1),
                                 padding='same')

    def __getitem__(self, item):
        pass

    def get_global_step(self):
        return self.global_step

    def get_optimizer(self, optimizer):
        return self.optimizer

    def get_schedule(self, schedule):
        return self.schedule








