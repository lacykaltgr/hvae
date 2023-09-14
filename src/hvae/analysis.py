import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import tqdm
import random
import torch
from torch import nn

from src.utils import NumpyEncoder
from src.hparams import get_hparams
from torch.utils.data import Dataset, DataLoader
from src.hvae.model import reconstruct

NUM_TEXT_FAMILY = 5


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


def decodability(model, labeled_loader, filter_dict=None):
    p = get_hparams()
    decode_from_list = p.synthesis_params.decodability.decode_from
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
        decodability_model = p.synthesis_params.decodability.model(num_input_dims, num_classes)
        decodability_dataset = Decodability_dataset(X[decode_from], Y)
        optimizer = p.synthesis_params.decodability.optimizer(
            decodability_model.parameters(), lr=p.synthesis_params.decodability.learning_rate)
        loss = p.synthesis_params.decodability.loss()
        accuracy = decodability_model(decodability_model, optimizer, loss, p.synthesis_params.decodability.epochs,
                                             p.synthesis_params.decodability.batch_size, decodability_dataset)
        accuracies[decode_from] = accuracy


def plot_reconstruction(net, dataloader, checkpoint_path, logger):
    io_pairs = reconstruct(net, dataloader, latents_folder=None, logger=logger)

    row_titles = ["Original", "Sampled", "Mean"]
    n = len(io_pairs)
    m = len(row_titles)
    fig, axes = plt.subplots(nrows=m, ncols=n, figsize=(12, 8))
    for ax, row in zip(axes[:, 0], row_titles):
        ax.set_title(row, size='large')
    for i in range(n):
        for j in range(m):
            axes[i, j].imshow(io_pairs[i][j], interpolation='none', cmap='gray')
            axes[i, j].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(checkpoint_path, "analysis", f"reconstruction.png"), facecolor="white")


def latent_traversal (model, sample, target_block, n_cols, diff=0 , value=1, checkpoint_path=None, n_dims=70):

    compute_target_block = model.compute_function(target_block)
    target_computed, _ = compute_target_block(sample)
    input_0 = target_computed[target_block]

    compute_output = model.compute_function('output')
    output_computed, _ = compute_output(target_computed, use_mean=True)
    output_0 = torch.mean(output_computed['output'], dim=0)

    n_rows = int(np.ceil(n_dims/n_cols))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*2, n_rows*2))

    for i in range(n_dims):
        input_i = np.zeros([1, n_dims])
        input_i[0, i] = value
        input_i = input_0 + input_i
        target_computed[target_block] = input_i

        trav_output_computed, _ = compute_output(target_computed, use_mean=True)
        output_i = torch.mean(trav_output_computed, dim=0)

        ax[i // n_cols][i % n_cols].imshow(output_i-diff*output_0, interpolation='none', cmap='gray')
        ax[i // n_cols][i % n_cols].set_title(f"{i}")
        ax[i // n_cols][i % n_cols].axis('off')

    path = os.path.join(checkpoint_path, "analysis", f"Z2_trav.png")
    plt.title(f"{target_block} traversal")
    fig.savefig(path, facecolor="white")
    plt.show()


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def get_Z2_post(model, ex, filter_dict=None, class_label=None, return_KL=False):
    if class_label is not None:
        image_batch = ex[0][ex[1] == class_label]
    else:
        image_batch = ex[0]

    Z1 = model.q_z1_x_model(image_batch).mean()

    if filter_dict:
        filters = filter_dict["filter_dims"]
        non_filters = filter_dict["non_filter_dims"]
        Z1 = Z1.numpy()
        Z1[:, filters] = shuffle_along_axis(Z1[:, filters], axis=1)

    Z2_mean = model.q_z2_z1_model(Z1).mean()
    Z2_std = model.q_z2_z1_model(Z1).stddev()

    if return_KL:
        #    import pdb; pdb.set_trace()
        KL = tfp.distributions.kl_divergence(model.q_z2_z1_model(Z1), model.q_z2_z1_model.prior)
        return Z2_mean, Z2_std, KL
    else:
        return Z2_mean, Z2_std


def get_Z2_mean_std(model, ds, filter_dict=None, class_label=None):

    Z2_means=[]
    Z2_stds=[]

    for ex in ds:
        next_mean, next_std = get_Z2_post(model, ex, filter_dict, class_label=class_label)
        Z2_means.append(next_mean)
        Z2_stds.append(next_std)

    average_Z2_means=np.mean(np.concatenate(Z2_means, axis=0), axis=0)
    average_abs_Z2_means=np.mean(np.concatenate(np.abs(Z2_means), axis=0), axis=0)
    std_Z2_means=np.std(np.concatenate(Z2_means, axis=0), axis=0)
    average_Z2_stds=np.mean(np.concatenate(Z2_stds, axis=0), axis=0)
    #import pdb; pdb.set_trace()

    fig, axs=plt.subplots(ncols=2,figsize=(12,6))


    axs[0].hist(average_Z2_means.flatten(), bins=30)
    axs[0].set_title("mean")


    axs[1].hist(average_Z2_stds.flatten(), bins=30)
    axs[1].set_title("std")

    print(f"Dims with std < 0.95 : {np.count_nonzero(average_Z2_stds<0.95)}")
    print(f"Dims with std of means < 0.95 : {np.count_nonzero(std_Z2_means<0.95)}")
    print(f"Dims with abs(mean)>0.05  : {np.count_nonzero(np.abs(average_Z2_means)>0.05)}")

    return average_Z2_means, average_Z2_stds, average_abs_Z2_means, std_Z2_means


def generate_active_dim_plots(experiment_path):
    text_mean, text_std, text_abs_mean, std_mean = {}, {}, {}, {}

    for label in range(NUM_TEXT_FAMILY):
        text_mean[label], text_std[label], text_abs_mean[label], std_mean[label] = \
            get_Z2_mean_std(exp.model, ds_test, class_label=label)

    text_mean_df = pd.DataFrame(data=text_mean)       #mean of posterior means
    std_mean_df = pd.DataFrame(data=std_mean)         #std of posterior means
    text_std_df = pd.DataFrame(data=text_std)         #mean of posterior std
    save_dir = os.path.join(experiment_path, "analysis", "z2_dim_plots")
    os.makedirs(save_dir, exist_ok=True)

    text_mean_path = os.path.join(save_dir, f"text_mean.png")
    std_mean_path = os.path.join(save_dir, f"std_mean.png")
    text_std_path = os.path.join(save_dir, f"text_std.png")
    fig1 = text_mean_df.plot.bar(figsize=(20,10), title='Average posterior means by texture families for each z2 latent variable')
    fig1.set_xlabel('z2 latent variable index')
    fig1.figure.savefig(text_mean_path)
    fig2 = std_mean_df.plot.bar(figsize=(20,10), title='Std of posterior means by texture families for each z2 latent variable')
    fig2.set_xlabel('z2 latent variable index')
    fig2.figure.savefig(std_mean_path)
    fig3 = text_std_df.plot.bar(figsize=(20,10), title='Average posterior standard deviation by texture families for each z2 latent variable')
    fig3.figure.savefig(text_std_path)
    fig3.set_xlabel('z2 latent variable index')



def Z1_filters(experiment, step_size=1, num_dims=450):
    from sklearn.cluster import k_means

    MSPs = []
    for i in range(num_dims):
        input_i = np.zeros([1, num_dims])
        input_i[0, i] = step_size
        output_i = torch.mean(experiment.model.p_x_z1_model(input_i).mean(), dim=0)

        MSP = torch.mean(torch.square(output_i))
        MSPs.append(MSP)

    hist = np.histogram(MSPs, bins=int(num_dims/10))

    sample_weight = hist[0]
    sample_weight[hist[0] < 3] = 0
    _, label, _ = k_means(hist[1][:-1].reshape(-1, 1), 2, sample_weight=sample_weight)
    bin_threshold = np.argmax(label != label[0])
    threshold = hist[1][bin_threshold+1]
    msp_array = np.array(MSPs)

    filter_count = np.sum(msp_array > threshold)
    filter_dims = np.arange(num_dims)[np.where(msp_array > threshold)]
    non_filter_dims = np.arange(num_dims)[np.where(msp_array < threshold)]

    filter_dict = {
        "filter_count": filter_count,
        "filter_dims": filter_dims,
        "non_filter_dims": non_filter_dims
    }

    save_path = os.path.join(experiment.directory, "analysis", f"Z1_filters.json")
    with open(save_path, "w") as f:
        json.dump(filter_dict, f, cls=NumpyEncoder)
    return filter_dict


