import os
import logging

import numpy as np
import torch

from hparams import get_hparams
from src.model import compute_per_dimension_divergence_stats, generate, reconstruct
from src.utils import write_image_to_disk, setup_logger, load_experiment_for


def divergence_stats_mode(model, dataset, latents_folder):
    stats_filepath = os.path.join(latents_folder, 'div_stats.npy')
    per_dim_divs = compute_per_dimension_divergence_stats(model, dataset)
    np.save(stats_filepath, per_dim_divs.detach().cpu().numpy())


def generation_mode(model, artifacts_folder, logger: logging.Logger = None):
    artifacts_folder = artifacts_folder.replace('synthesized-images', 'synthesized-images/generated')
    os.makedirs(artifacts_folder, exist_ok=True)
    outputs = generate(model, logger)
    for temp_i, temp_outputs in enumerate(outputs):
        sample_i = 0
        for step_output in outputs:
            for output in step_output:
                write_image_to_disk(os.path.join(artifacts_folder, f'setup-{temp_i:01d}-image-{sample_i:04d}.png'),
                                    output.detach().cpu().numpy())


def reconstruction_mode(model, test_dataset, artifacts_folder=None, latents_folder=None, logger: logging.Logger = None):
    reconstruct(model, test_dataset, artifacts_folder, latents_folder, logger)


def synthesize(model, data, logdir, mode, logger: logging.Logger = None):
    artifacts_folder = os.path.join(logdir, 'synthesized-images')
    latents_folder = os.path.join(logdir, 'latents')
    os.makedirs(artifacts_folder, exist_ok=True)
    os.makedirs(latents_folder, exist_ok=True)

    if mode == 'reconstruction':
        reconstruction_mode(model, data, artifacts_folder, latents_folder, logger)
    elif mode == 'generation':
        generation_mode(model, artifacts_folder, logger)
    elif mode == 'div_stats':
        divergence_stats_mode(model, data, latents_folder)
    else:
        logger.error(f'Unknown Mode {mode}')
        raise ValueError(f'Unknown Mode {mode}')


def main():
    p = get_hparams()
    checkpoint, checkpoint_path = load_experiment_for('synthesis')
    logger = setup_logger(checkpoint_path)

    assert checkpoint is not None
    model = checkpoint.get_model()
    logger.info(f'Model Checkpoint is loaded from {p.log_params.load_from_synthesis}')
    with torch.no_grad():
        _ = model(torch.ones((1, *p.data_params.shape)))

    model = model.to(p.model_params.device)

    data_loader = None if p.synthesis_params.synthesis_mode == "generation" \
        else p.data_params.dataset.get_test_loader() if p.synthesis_params.synthesis_mode == 'reconstruction' \
        else p.data_params.dataset.get_train_loader() if p.synthesis_params.synthesis_mode in ['encoding', 'div_stats'] \
        else None

    synthesize(model, data_loader, checkpoint_path, mode=p.synthesis_params.synthesis_mode, logger=logger)


if __name__ == '__main__':
    main()
