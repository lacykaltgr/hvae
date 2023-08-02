import os
import pickle

from src.utils import create_checkpoint_manager_and_load_if_exists, get_logdir, write_image_to_disk
from src.model import compute_per_dimension_divergence_stats, generate, reconstruct

from hparams import *
import torch
import numpy as np


def divergence_stats_mode(model, dataset, latents_folder):
    stats_filepath = os.path.join(latents_folder, 'div_stats.npy')
    per_dim_divs = compute_per_dimension_divergence_stats(dataset, model)
    np.save(stats_filepath, per_dim_divs.detach().cpu().numpy())


def generation_mode(model, artifacts_folder):
    artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/generated')
    os.makedirs(artifacts_folder, exist_ok=True)
    outputs = generate(model)
    for temp_i, temp_outputs in enumerate(outputs):
        sample_i = 0
        for step_output in outputs:
            for output in step_output:
                write_image_to_disk(os.path.join(artifacts_folder, f'setup-{temp_i:01d}-image-{sample_i:04d}.png'),
                                    output.detach().cpu().numpy())


def reconstruction_mode(model, test_dataset, artifacts_folder=None, latents_folder=None):
    reconstruct(test_dataset, model, artifacts_folder, latents_folder)


def synthesize(model, data, logdir, mode):
    artifacts_folder = os.path.join(logdir, 'synthesis-images')
    latents_folder = os.path.join(logdir, 'latents')
    os.makedirs(artifacts_folder, exist_ok=True)
    os.makedirs(latents_folder, exist_ok=True)

    if mode == 'reconstruction':
        reconstruction_mode(model, data, artifacts_folder, latents_folder)
    elif mode == 'generation':
        generation_mode(model, artifacts_folder)
    elif mode == 'div_stats':
        divergence_stats_mode(model, data, latents_folder)
    else:
        raise ValueError(f'Unknown Mode {mode}')


def main():
    model = model_params.model()
    with torch.no_grad():
        _ = model(torch.ones((1, data_params.channels, data_params.target_res, data_params.target_res)).cuda())

    checkpoint, checkpoint_path = create_checkpoint_manager_and_load_if_exists(rank=0)

    if synthesis_params.load_ema_weights:
        assert checkpoint['ema_model_state_dict'] is not None
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print('EMA model is loaded')
    else:
        assert checkpoint['model_state_dict'] is not None
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model Checkpoint is loaded')
    print(checkpoint_path)

    data_loader = None if synthesis_params.synthesis_mode == "generation" \
        else data_params.dataset.get_test_loader() if synthesis_params.synthesis_mode == 'reconstruction' \
        else data_params.dataset.get_train_loader() if synthesis_params.synthesis_mode in ['encoding', 'div_stats'] \
        else None

    # Synthesis using pretrained model
    logdir = get_logdir()
    synthesize(model, data_loader, logdir, mode=synthesis_params.synthesis_mode)


if __name__ == '__main__':
    main()
