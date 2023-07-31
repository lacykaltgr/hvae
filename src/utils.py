import logging
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from hparams import *
import os


def generate_loc_scale_distr(logits, distr, sigma_nonlin, sigma_param):
    """Generate a location-scale distribution."""

    mu, sigma = torch.split(logits, split_size_or_sections=1, dim=1)

    if sigma_nonlin == 'exp':
        sigma = torch.exp(sigma)
    elif sigma_nonlin == 'softplus':
        sigma = F.softplus(sigma)
    else:
        raise ValueError(f'Unknown sigma_nonlin {sigma_nonlin}')

    if sigma_param == 'var':
        sigma = torch.sqrt(sigma)
    elif sigma_param != 'std':
        raise ValueError(f'Unknown sigma_param {sigma_param}')

    if distr == 'normal':
        return dist.Normal(loc=mu, scale=sigma)
    elif distr == 'laplace':
        return dist.Laplace(loc=mu, scale=sigma)
    else:
        raise ValueError(f'Unknown distr {distr}')


def construct_prior_params(batch_size, n_y):
    """Construct the location-scale prior parameters.

    Args:
      batch_size: int, the size of the batch.
      n_y: int, the number of uppermost model layer dimensions.

    Returns:
      Constant representing the prior parameters, size of [batch_size, 2*n_y].
    """
    loc_scale = torch.zeros((batch_size, 2 * n_y), dtype=torch.float32, requires_grad=False)
    return loc_scale


def maybe_center_crop(layer, target_hw):
    """Center crop the layer to match a target shape."""
    l_height, l_width = layer.shape.as_list()[1:3]
    t_height, t_width = target_hw
    assert t_height <= l_height and t_width <= l_width

    if (l_height - t_height) % 2 != 0 or (l_width - t_width) % 2 != 0:
        logging.warn(
            'It is impossible to center-crop [%d, %d] into [%d, %d].'
            ' Crop will be uneven.', t_height, t_width, l_height, l_width)

    border = int((l_height - t_height) / 2)
    x_0, x_1 = border, l_height - border
    border = int((l_width - t_width) / 2)
    y_0, y_1 = border, l_width - border
    layer_cropped = layer[:, x_0:x_1, y_0:y_1, :]
    return layer_cropped


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

def compute_latent_dimension():
    assert np.prod(hparams.model.down_strides) == np.prod(hparams.model.up_strides)
    return hparams.data.target_res // np.prod(hparams.model.down_strides)


def get_same_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    # Reverse order for F.pad
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("Can't have the stride and dilation rate over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]
        if p % 2 == 0:
            p = (p // 2, p // 2)
        else:
            p = (int(np.ceil(p / 2)), int(np.floor(p / 2)))

        p_ += p
    return tuple(p_)


def get_valid_padding(n_dims=2):
    p_ = (0,) * 2 * n_dims
    return p_


def get_causal_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("can't have the stride and dilation over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]

        p_ += (p, 0)

    return p_


def tensorboard_log(model, optimizer, global_step, writer, losses, outputs, targets, means=None, log_scales=None,
                    updates=None,
                    global_norm=None, train_steps_per_epoch=None, mode='train'):
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
        writer.add_scalar("Schedules/learning_rate", optimizer.param_groups[0]['lr'],
                          global_step)

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


def load_checkpoint_if_exists(checkpoint_path, rank):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(rank))
    except FileNotFoundError:
        checkpoint = {'global_step': -1,
                      'model_state_dict': None,
                      'ema_model_state_dict': None,
                      'optimizer_state_dict': None,
                      'scheduler_state_dict': None}
    return checkpoint


def create_checkpoint_manager_and_load_if_exists(model_directory='.', rank=0):
    checkpoint_path = os.path.join(model_directory, f'checkpoints-{hparams.run.name}')
    checkpoint = load_checkpoint_if_exists(checkpoint_path, rank)

    return checkpoint, checkpoint_path


def get_logdir():
    return f'logs-{hparams.run.name}'


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


def get_variate_masks(stats):
    thresh = np.quantile(stats, 1 - synthesis_params.variates_masks_quantile)
    return stats > thresh


def linear_temperature(min_temp, max_temp, n_layers):
    slope = (max_temp - min_temp) / n_layers

    def get_layer_temp(layer_i):
        return slope * layer_i + min_temp

    return get_layer_temp


def reshape_distribution(dist_list, variate_mask):
    """
    :param dist_list: n_layers, 2*  [ batch_size n_variates, H , W]
    :return: Tensors  of shape batch_size, H, W ,n_variates, 2
    H, W , n_variates will be different from each other in the list depending on which layer you are in.
    """
    dist = torch.stack(dist_list, dim=0)  # 2, batch_size, n_variates, H ,W
    dist = dist[:, :, variate_mask, :, :]  # Only take effective variates
    dist = torch.permute(dist, (1, 3, 4, 2, 0))  # batch_size, H ,W ,n_variates (subset), 2
    # dist = torch.unbind(dist, dim=0)  # Return a list of tensors of length batch_size
    return dist
