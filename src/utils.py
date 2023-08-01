from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from hparams import *
import os


"""
-------------------
MODEL UTILS
-------------------
"""


def scale_pixels(img):
    img = np.floor(img / np.uint8(2 ** (8 - data_params.num_bits))) * 2 ** (8 - data_params.num_bits)
    shift = scale = (2 ** 8 - 1) / 2
    img = (img - shift) / scale  # Images are between [-1, 1]
    return img


def effective_pixels():
    return data_params.target_res * data_params.target_res * data_params.channels


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=torch.device('cuda'))
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)
    return y_onehot


def get_variate_masks(stats):
    thresh = np.quantile(stats, 1 - synthesis_params.variates_masks_quantile)
    return stats > thresh


def linear_temperature(min_temp, max_temp, n_layers):
    slope = (max_temp - min_temp) / n_layers

    def get_layer_temp(layer_i):
        return slope * layer_i + min_temp

    return get_layer_temp


"""
-------------------
TRAIN/LOG UTILS
-------------------
"""


def tensorboard_log(model, optimizer, global_step, writer,
                    losses, outputs, targets, means=None, log_scales=None,
                    updates=None, global_norm=None, mode='train'):
    for key, value in losses.items():
        writer.add_scalar(f"Losses/{key}", value, global_step)
    writer.add_histogram("Distributions/target", targets, global_step, bins=20)
    writer.add_histogram("Distributions/output", torch.clamp(outputs, min=-1., max=1.), global_step, bins=20)

    if means is not None:
        assert log_scales is not None
        writer.add_histogram('OutputLayer/means', means, global_step, bins=30)
        writer.add_histogram('OutputLayer/log_scales', log_scales, global_step, bins=30)

    if mode == 'train':
        for variable in model.parameters():
            writer.add_histogram("Weights/{}".format(variable.name), variable, global_step)
        # Get the learning rate from the optimizer
        writer.add_scalar("Schedules/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        if updates is not None:
            for layer, update in updates.items():
                writer.add_scalar("Updates/{}".format(layer), update, global_step)
            max_updates = torch.max(torch.stack(list(updates.values())))
            assert global_norm is not None
            writer.add_scalar("Mean_Max_Updates/Global_norm", global_norm, global_step)
            writer.add_scalar("Mean_Max_Updates/Max_updates", max_updates, global_step)
    writer.flush()


def plot_image(outputs, targets, step, writer):
    writer.add_image(f"{step}/Original_{step}", targets, step)
    writer.add_image(f"{step}/Generated_{step}", outputs, step)


def load_checkpoint_if_exists(checkpoint_path, rank=0):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
    except FileNotFoundError:
        checkpoint = {'global_step': -1,
                      'model_state_dict': None,
                      'ema_model_state_dict': None,
                      'optimizer_state_dict': None,
                      'scheduler_state_dict': None}
    return checkpoint


def create_checkpoint_manager_and_load_if_exists(model_directory='.', rank=0):
    checkpoint_path = os.path.join(model_directory, f'checkpoints-{model_params.name}')
    checkpoint = load_checkpoint_if_exists(checkpoint_path, rank)
    return checkpoint, checkpoint_path


def get_logdir():
    return f'logs-{model_params.name}'


def create_tb_writer(mode):
    logdir = get_logdir()
    tbdir = os.path.join(logdir, mode)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tbdir)
    return writer, logdir


def write_image_to_disk(filepath, image):
    from PIL import Image
    assert len(image.shape) == 3
    assert image.shape[0] == 3

    image = np.round(image * 127.5 + 127.5)
    image = image.astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    im = Image.fromarray(image)
    im.save(filepath, format='png')

