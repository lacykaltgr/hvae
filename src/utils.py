import os
import logging

import numpy
import numpy as np
import torch
from torch.nn import Sequential, Module
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


def get_variate_masks(stats):
    p = get_hparams()
    thresh = np.quantile(stats, 1 - p.synthesis_params.variates_masks_quantile)
    return stats > thresh


def linear_temperature(min_temp, max_temp, n_layers):
    slope = (max_temp - min_temp) / n_layers

    def get_layer_temp(layer_i):
        return slope * layer_i + min_temp

    return get_layer_temp


def split_mu_sigma(x, chunks=2, dim=1):
    assert x.shape[dim] % chunks == 0
    mu, sigma = torch.chunk(x, chunks, dim=dim)
    if mu.shape[dim] == 1:
        mu = mu.squeeze(dim)
        sigma = sigma.squeeze(dim)
    return mu, sigma


"""
-------------------
TRAIN/LOG UTILS
-------------------
"""


def get_root_dir(path: str):
    if path is None:
        return None
    path_split = path.split("/")
    assert path_split[-2] == "checkpoints"

    path = path.split("checkpoints")[0]
    return path


def get_save_load_paths(mode='train'):
    p = get_hparams().log_params
    if mode == 'test':
        load_from = p.load_from_eval
        assert load_from is not None
        load_from_file = f'{p.dir}{p.name}/{load_from}'
        return load_from_file, None

    elif mode == 'train':
        load_from = p.load_from_train
        load_from_file = f'{p.dir}{p.name}/{load_from}' if load_from is not None else None
        import datetime
        p = get_hparams().log_params
        if p.dir_naming_scheme == 'timestamp':
            save_dir = f"{p.dir}{p.name}/{datetime.datetime.now().strftime('%Y-%m-%d__%H-%M')}"
        else:
            save_dir = f"{p.dir}{p.name}/{p.dir_naming_scheme}"
        # else:
        #    raise NotImplementedError(f"Unknown dir_naming_scheme {p.dir_naming_scheme}")
        os.makedirs(save_dir, exist_ok=True)
        return load_from_file, save_dir

    elif mode == 'synthesis':
        load_from = p.load_from_synthesis
        assert load_from is not None
        load_from_file = f'{p.dir}{p.name}/{load_from}'
        save_dir = os.path.join(get_root_dir(load_from_file), "synthesis")
        os.makedirs(save_dir, exist_ok=True)
        return load_from_file, save_dir

    elif mode == 'migration':
        import datetime
        save_dir = os.path.join(f"{p.dir}{p.name}/migration", f"{datetime.datetime.now().strftime('%Y-%m-%d__%H-%M')}", )
        os.makedirs(save_dir, exist_ok=True)
        return None, save_dir

    else:
        raise ValueError(f"Unknown mode {mode}")


def tensorboard_log(model, optimizer, global_step, writer,
                    losses, outputs, targets,
                    updates=None, global_norm=None, mode='train'):
    for key, value in losses.items():
        if isinstance(value, (torch.Tensor, numpy.ndarray)) and len(value.shape) == 0:
            writer.add_scalar(f"Losses/{key}", value, global_step)
    writer.add_histogram("Distributions/target", targets, global_step, bins=20)
    writer.add_histogram("Distributions/output", torch.clamp(outputs, min=-1., max=1.), global_step, bins=20)

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
    from src.checkpoint import Checkpoint
    load_from_file, save_to_path = get_save_load_paths(mode)

    experiment = None
    # load experiment from checkpoint
    if load_from_file is not None and os.path.isfile(load_from_file):
        # print(f"Loading experiment from {load_from_file}")
        experiment = Checkpoint.load(load_from_file)
    return experiment, save_to_path


def create_tb_writer_for(mode: str, checkpoint_path: str):
    logdir = os.path.join(checkpoint_path, 'tensorboard')
    tbdir = os.path.join(logdir, mode)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tbdir)
    return writer


def write_image_to_disk(filepath, image):
    from PIL import Image

    if image.shape[0] == 3:
        image = np.round(image * 127.5 + 127.5)
        image = image.astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        im = Image.fromarray(image)
        im.save(filepath, format='png')
    else:
        for im in image:
            im = im.astype(np.uint8)
            while len(im.shape) > 2:
                im = np.squeeze(im, axis=0)
            im = Image.fromarray(im)
            im.save(filepath, format='png')


def setup_logger(checkpoint_path: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('logger')
    file_handler = logging.FileHandler(os.path.join(checkpoint_path, 'log.txt'))
    logger.addHandler(file_handler)
    return logger


def prepare_for_log(results: dict):
    p = get_hparams().eval_params
    results["elbo"] = results["elbo"].detach().cpu().item()
    results["reconstruction_loss"] = results["reconstruction_loss"].detach().cpu().item()
    results["kl_div"] = results["kl_div"].detach().cpu().item()
    results["avg_reconstruction_loss"] = results["avg_reconstruction_loss"].detach().cpu().item()
    results["var_loss"] = np.sum([v.detach().cpu().item() for v in results["avg_var_prior_losses"]])
    results["n_active_groups"] = np.sum([v >= p.latent_active_threshold
                                         for v in results["avg_var_prior_losses"]])
    results.update({f'latent_kl_{i}': v for i, v in enumerate(results["avg_var_prior_losses"])})
    return results


def params_to_file(params, filepath):
    with open(filepath, 'a') as file:
        text = "PARAMETERS\n"
        for param_group in params.keys():
            text += f"{param_group}:\n" \
                   f"-------------------\n"
            for param in params[param_group].keys():
                text += f"{param}: {params[param_group][param]}\n"
            text += f"-------------------\n"
        file.write(text)
        file.close()
    return True


def print_line(logger: logging.Logger, newline_after: False):
    logger.info('\n' + '-' * 89 + ('\n' if newline_after else ''))


"""
-------------------
SERIALIZATION UTILS
-------------------
"""


class SerializableSequential(Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def serialize(self):
        return [layer.serialize() for layer in self._modules.values()]

    @staticmethod
    def deserialize(serialized):
        sequential = SerializableSequential(*[
            layer["type"].deserialize(layer)
            if isinstance(layer, dict)
            else SerializableSequential.deserialize(layer)
            for layer in serialized
        ])
        return sequential


class SerializableModule(Module):

    def __init__(self):
        super().__init__()

    def serialize(self):
        return dict(type=self.__class__, params=None)

    @staticmethod
    def deserialize(serialized):
        return serialized["type"]
