import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
from tqdm import tqdm

from src.utils import NumpyEncoder, get_variate_masks
from src.hparams import get_hparams
from torch.utils.data import Dataset, DataLoader
from src.hvae.model import reconstruct, device, kl_divergence


class Decodability_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def decodability_model(decodability_model, optimizer, loss, epochs, batch_size, dataset):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    accuracy = []
    # calculate model accuracy
    for epoch in range(epochs):
        for batch in dataloader:
            X, Y = batch
            optimizer.zero_grad()
            output = decodability_model(X)
            loss = loss(output, Y)
            loss.backward()
            optimizer.step()
    return accuracy


def decodability(model, labeled_loader, filepath):
    p = get_hparams()
    decode_from_list = p.analysis_params.decodability.decode_from
    X = {layer: [] for layer in decode_from_list}
    Y = []
    for batch in labeled_loader:
        inp, label = batch
        _, _, distributions = model(inp)
        for decode_from in decode_from_list:
            X[decode_from].append(distributions[decode_from].mean.numpy())
            Y.append(label)
    Y = np.concatenate(Y, axis=0)

    accuracies = dict()
    for decode_from in decode_from_list:
        X[decode_from] = np.concatenate(X[decode_from], axis=0)
        num_input_dims = X[decode_from].shape[1]
        num_classes = Y.shape[1]
        decodability_model = p.analysis_params.decodability.model(num_input_dims, num_classes)
        decodability_dataset = Decodability_dataset(X[decode_from], Y)
        optimizer = p.analysis_params.decodability.optimizer(
            decodability_model.parameters(), lr=p.analysis_params.decodability.learning_rate)
        loss = p.analysis_params.decodability.loss()
        accuracy = decodability_model(decodability_model, optimizer, loss, p.analysis_params.decodability.epochs,
                                      p.analysis_params.decodability.batch_size, decodability_dataset)
        accuracies[decode_from] = accuracy

    with open(filepath, "w") as f:
        json.dump(accuracies, f, cls=NumpyEncoder)


def plot_reconstruction(net, dataloader, save_path, logger):
    # Variate Masks
    if get_hparams().analysis_params.reconstruction.mask_reconstruction:
        div_stats = np.load(os.path.join(save_path, 'div_stats.npy'))
        variate_masks = get_variate_masks(div_stats).astype(np.float32)
    else:
        variate_masks = None

    # Reconstruction
    io_pairs = reconstruct(net, dataloader, variate_masks=variate_masks, logger=logger)

    # Plot
    row_titles = ["Original", "Sampled", "Mean"]
    n = len(io_pairs)
    m = len(row_titles)
    fig, axes = plt.subplots(nrows=m, ncols=n, figsize=(12, 8))
    for ax, row in zip(axes[:, 0], row_titles):
        ax.set_title(row, size='large')
    for j in range(n):
        for i in range(m):
            if io_pairs[i][j].shape[0] == 1:
                io_pairs[i][j] = io_pairs[i][j][0]
            image = io_pairs[i][j]
            axes[j, i].imshow(image, interpolation='none', cmap='gray')
            axes[j, i].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"reconstruction.png"), facecolor="white")


def latent_step_analysis(model, sample, target_block, save_path, n_cols=10, diff=1, value=1, n_dims=70):
    compute_target_block = model.compute_function(target_block)
    target_computed, _ = compute_target_block(sample)
    input_0 = target_computed[target_block]

    compute_output = model.compute_function('output')
    output_computed, _ = compute_output(target_computed, use_mean=True)
    output_0 = torch.mean(output_computed['output'], dim=0)

    n_rows = int(np.ceil(n_dims / n_cols))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_rows * 2))

    for i in range(n_dims):
        input_i = np.zeros([1, n_dims])
        input_i[0, i] = value
        input_i = input_0 + input_i
        target_computed[target_block] = input_i

        trav_output_computed, _ = compute_output(target_computed, use_mean=True)
        output_i = torch.mean(trav_output_computed, dim=0)

        ax[i // n_cols][i % n_cols].imshow(output_i - diff * output_0, interpolation='none', cmap='gray')
        ax[i // n_cols][i % n_cols].set_title(f"{i}")
        ax[i // n_cols][i % n_cols].axis('off')

    path = os.path.join(save_path, f"{target_block}_trav.png")
    plt.title(f"{target_block} traversal")
    fig.savefig(path, facecolor="white")


def white_noise_analysis(model, target_block, save_path, shape, n_samples=100, sigma=0.6, n_cols=10):
    import scipy

    if shape[0] == 1:
        shape = shape[1:]

    white_noise = np.random.normal(size=(n_samples, np.prod(shape)), loc=0.0, scale=1.).astype(np.float32)

    # apply ndimage.gaussian_filter with sigma=0.6
    #for i in range(n_samples):
    #    white_noise[i, :] = scipy.ndimage.gaussian_filter(
    #        white_noise[i, :].reshape(shape), sigma=sigma).reshape(np.prod(shape))

    compute_target_block = model.compute_function(target_block)
    target_computed, _ = compute_target_block(torch.zeros((1, *shape)))
    target_block_dim = target_computed[target_block].shape[1:]
    target_block_values = np.zeros((n_samples, *target_block_dim), dtype=np.float32)

    #loop over a batch of 128 white_noise images
    for i in range(0, n_samples, 128):
        batch = white_noise[i:i+128, :]
        computed_target, _ = compute_target_block(torch.from_numpy(batch), use_mean=True)
        target_block_values[i:i+128, :] = computed_target[target_block].detach().numpy()

    #multiply transpose of target block_values with white noise tensorially
    receptive_fields = np.matmul(target_block_values.transpose(), white_noise) / np.sqrt(n_samples)

    n_dims = receptive_fields.shape[0]
    n_rows = int(np.ceil(n_dims / n_cols))

    #plot receptive fields in a grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    for i in range(n_dims):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(receptive_fields[i, :].reshape(shape), interpolation='none', cmap="gray")
        ax.set_title(f"{i}")
        ax.axis("off")
    fig.tight_layout()

    wna_path = os.path.join(save_path, f"white_noise_analysis")
    os.makedirs(wna_path, exist_ok=True)
    np.save(os.path.join(wna_path, f"{target_block}_reverse_correlation.npy"), receptive_fields)
    fig.savefig(os.path.join(wna_path, f"{target_block}_reverse_correlation.png"), facecolor="white")



def model_summary(net):
    """
    Print the model summary
    :param net: nn.Module, the network
    :return: None
    """
    from torchinfo import summary
    shape = (1,) + get_hparams().data_params.shape
    return summary(net, input_size=shape, depth=7)


def generate_mei(model, target_block, query_config):
    from meitorch.mei import MEI

    def get_target_block(x):
        compute_function = model.compute_function(target_block, use_mean=True)
        computed, _ = compute_function(x)
        return computed[target_block]

    mei = MEI(models=[get_target_block], shape=get_hparams().data_params.shape)
    meip = mei.generate(**query_config)           # Whether to clip the range of the image to be in valid range

    from matplotlib.pyplot import imsave
    imsave('mei.png', meip.image.detach().cpu().numpy())
    return meip


def get_optimal_gabor(model, target_block, query):
    from meitorch.gabor import Gabor

    def get_target_block(x):
        compute_function = model.compute_function(target_block)
        computed, _ = compute_function(x)
        return computed[target_block]

    gabor = Gabor(models=[get_target_block], shape=(3, 32, 32))
    gaborp = gabor.optimal_gabor(query)
    return gaborp


def compute_per_dimension_divergence_stats(net, dataset: DataLoader) -> tensor:
    """
    Compute the per-dimension KL divergence statistics for the given network and dataset
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param dataset: DataLoader, the dataset
    :return: tensor, the per-dimension KL divergence statistics
    """
    per_dim_divs = None
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(dataset)):
            inputs = inputs.to(device, non_blocking=True)
            _, distributions = net(inputs)
            avg_losses = []

            for block_name, (prior, posterior) in distributions.items():
                if block_name == 'output' or posterior is None:
                    continue
                _, avg_loss = kl_divergence(prior, posterior)
                avg_losses.append(avg_loss)
            kl_div = torch.stack(avg_losses)
            per_dim_divs = kl_div if per_dim_divs is None else per_dim_divs + kl_div
            if step > get_hparams().analysis_params.div_stats.div_stats_subset_ratio * len(dataset):
                break
    per_dim_divs /= (step + 1)
    return per_dim_divs


