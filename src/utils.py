import os
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from hparams import get_hparams
"""
-------------------
MODEL UTILS
-------------------
"""


def scale_pixels(img, data_num_bits):
    img = np.floor(img / np.uint8(2 ** (8 - data_num_bits))) * 2 ** (8 - data_num_bits)
    shift = scale = (2 ** 8 - 1) / 2
    img = (img - shift) / scale  # Images are between [-1, 1]
    return img


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=torch.device('cuda'))
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)
    return y_onehot


def get_variate_masks(stats):
    p = get_hparams()
    thresh = np.quantile(stats, 1 - p.synthesis_params.variates_masks_quantile)
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


def get_experiment_dir():
    p = get_hparams()
    return f'{p.model_params.dir}{p.model_params.name}'


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

    # Save artifacts
    plot_image(outputs[0], targets[0], global_step, writer=writer)
    writer.flush()


def plot_image(outputs, targets, step, writer):
    writer.add_image(f"{step}/Original_{step}", targets, step)
    writer.add_image(f"{step}/Generated_{step}", outputs, step)


def load_experiment_for(mode: str = 'test'):
    p = get_hparams()
    if mode == 'test':
        experiment_directory = p.eval_params.load_from
    elif mode == 'train':
        experiment_directory = p.train_params.load_from
    elif mode == 'synthesis':
        experiment_directory = p.synthesis_params.load_from
    else:
        raise ValueError(f"Unknown mode {mode}")

    # not load experiment
    if experiment_directory is None:
        experiment_directory = ''

    # load latest experiment
    if experiment_directory == 'latest':
        experiment_directory = sorted(os.listdir(get_experiment_dir()))[-1]

    path = os.path.join(get_experiment_dir(), experiment_directory)
    os.makedirs(path, exist_ok=True)

    if experiment_directory is None:
        return None, path

    file_path = os.path.join(path, 'experiment.pt')
    experiment = torch.load(file_path) if os.path.exists(file_path) else None
    return experiment, path


def create_tb_writer_for(mode: str, checkpoint_path: str):
    logdir = os.path.join(checkpoint_path, 'tensorboard')
    tbdir = os.path.join(logdir, mode)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tbdir)
    return writer


def write_image_to_disk(filepath, image):
    from PIL import Image
    assert len(image.shape) == 3
    assert image.shape[0] == 3

    image = np.round(image * 127.5 + 127.5)
    image = image.astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    im = Image.fromarray(image)
    im.save(filepath, format='png')


def setup_logger(checkpoint_path: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('logger')
    file_handler = logging.FileHandler(os.path.join(checkpoint_path, 'log.txt'))
    logger.addHandler(file_handler)
    return logger


def detach_all(results: dict):
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            results[key] = value.detach().cpu().item()
        elif isinstance(value, list):
            results[key] = list(map(lambda x: x.detach().cpu(), value))
    return results


def prepare_for_log(results: dict):
    p = get_hparams()
    losses = dict(
        reconstruction_loss=results["reconstruction_loss"],
        kl_div=results["kl_div"],
        nelbo=results['elbo'],
        train_var_loss=np.sum([v for v in results["avg_var_prior_losses"]]),
        n_active_groups=np.sum([v >= p.eval_params.latent_active_threshold
                                  for v in results["avg_var_prior_losses"]])
    )
    losses = detach_all(losses)
    losses["means"] = results["means"].detach().cpu(),
    losses["log_scales"] = results["log_scales"].detach().cpu(),
    losses.update({f'latent_kl_{i}': v for i, v in enumerate(results["avg_var_prior_losses"])})
    return losses

