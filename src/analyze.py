import os
import logging

import numpy as np
import torch

from src.hparams import get_hparams
from src.hvae.model import generate
from src.hvae.analysis_tools import compute_per_dimension_divergence_stats, decodability, generate_mei, \
    get_optimal_gabor, latent_step_analysis, plot_reconstruction, white_noise_analysis
from src.utils import write_image_to_disk, setup_logger, load_experiment_for


def divergence_stats_mode(model, dataset, save_path, logger: logging.Logger = None):
    logger.info('Computing Divergence Stats')
    stats_filepath = os.path.join(save_path, 'div_stats.npy')
    per_dim_divs = compute_per_dimension_divergence_stats(model, dataset)
    np.save(stats_filepath, per_dim_divs.detach().cpu().numpy())
    logger.info(f'Divergence Stats saved to {stats_filepath}')


def generation_mode(model, save_path, logger: logging.Logger = None):
    logger.info('Generating Images')
    artifacts_folder = os.path.join(save_path, 'generated_images')
    os.makedirs(artifacts_folder, exist_ok=True)
    outputs = generate(model, logger)
    for temp_i, temp_outputs in enumerate(outputs):
        for sample_i, output in enumerate(temp_outputs):
            write_image_to_disk(os.path.join(artifacts_folder, f'setup-{temp_i:01d}-image-{sample_i:04d}.png'),
                                output.detach().cpu().numpy())
    logger.info(f'Generated Images saved to {artifacts_folder}')


def reconstruction_mode(model, test_dataset, save_path, logger: logging.Logger = None):
    logger.info('Reconstructing Images')
    rec_save_path = os.path.join(save_path, 'reconstructed_images')
    os.makedirs(rec_save_path, exist_ok=True)
    plot_reconstruction(model, test_dataset, rec_save_path, logger)
    logger.info(f'Reconstructed Images saved to {rec_save_path}')


def decodability_mode(model, labeled_loader, save_path, logger: logging.Logger = None):
    logger.info('Computing Decodability')
    decodability_filepath = os.path.join(save_path, 'decodability.json')
    decodability(model, labeled_loader, decodability_filepath)
    logger.info(f'Decodability saved to {decodability_filepath}')


def mei_mode(model, save_path, logger: logging.Logger = None):
    logger.info('Generating Most Exciting Inputs (MEI)')
    mei_folder = os.path.join(save_path, 'mei')
    os.makedirs(mei_folder, exist_ok=True)
    mei_filepath = os.path.join(save_path, 'mei.pth')
    processes = dict()
    for target_block, config in get_hparams().analysis_params.mei.queries.items():
        meip = generate_mei(model, target_block, config)
        write_image_to_disk(os.path.join(mei_folder, f'{target_block}.png'), meip.image.detach().cpu().numpy())
        processes[target_block] = meip
    torch.save(processes, mei_filepath)
    logger.info(f'MEIs saved to {mei_filepath}')


def gabor_mode(model, save_path, logger: logging.Logger = None):
    logger.info('Generating Optimal Gabor Filters')
    gabor_folder = os.path.join(save_path, 'gabor')
    os.makedirs(gabor_folder, exist_ok=True)
    gabor_filepath = os.path.join(save_path, 'gabor.pth')
    processes = dict()
    for target_block, config in get_hparams().analysis_params.gabor.queries.items():
        gabor = get_optimal_gabor(model, target_block, config)
        write_image_to_disk(os.path.join(gabor_folder, f'{target_block}.png'), gabor.image.detach().cpu().numpy())
        processes[target_block] = gabor
    torch.save(processes, gabor_filepath)
    logger.info(f'Gabor Filters saved to {gabor_filepath}')


def latent_step_analysis_mode(model, dataloader, save_path, logger: logging.Logger = None):
    logger.info('Generating Images with Latent Step Analysis')
    traversal_folder = os.path.join(save_path, 'latent_step_analysis')
    os.makedirs(traversal_folder, exist_ok=True)
    sample = next(iter(dataloader))
    for target_block, config in get_hparams().analysis_params.latent_step.queries.items():
        latent_step_analysis(model, sample, target_block, traversal_folder, **config)
    logger.info(f'Latent Traversal Images saved to {traversal_folder}')


def white_noise_analysis_mode(model, save_path, logger: logging.Logger = None):
    logger.info('Generating Images with White Noise Analysis')
    save_folder = os.path.join(save_path, 'white_noise_analysis')
    os.makedirs(save_folder, exist_ok=True)
    shape = get_hparams().data_params.shape
    for target_block, config in get_hparams().analysis_params.white_noise_analysis.queries.items():
        white_noise_analysis(model, target_block, save_folder, shape, **config)
    logger.info(f'White Noise Analysis Images saved to {save_folder}')


def main():
    p = get_hparams()
    checkpoint, save_path = load_experiment_for('analysis')
    logger = setup_logger(save_path)

    assert checkpoint is not None
    model = checkpoint.get_model()
    logger.info(f'Model Checkpoint is loaded from {p.log_params.load_from_analysis}')
    with torch.no_grad():
        _ = model(torch.ones((1, *p.data_params.shape)))

    model = model.to(p.model_params.device)

    dataset = p.data_params.dataset(**p.data_params.params)
    for operation in p.analysis_params.ops:
        if operation == 'reconstruction':
            dataloader = dataset.get_test_loader(p.eval_params.batch_size)
            reconstruction_mode(model, dataloader, save_path, logger)
        elif operation == 'generation':
            generation_mode(model, save_path, logger)
        elif operation == 'div_stats':
            dataloader = dataset.get_val_loader(p.eval_params.batch_size)
            divergence_stats_mode(model, dataloader, save_path, logger)
        elif operation == 'decodability':
            decodability_mode(model, dataset, save_path, logger)
        elif operation == 'mei':
            mei_mode(model, save_path, logger)
        elif operation == 'gabor':
            gabor_mode(model, save_path, logger)
        elif operation == 'white_noise_analysis':
            white_noise_analysis_mode(model, save_path, logger)
        elif operation == 'latent_step':
            dataloader = dataset.get_val_loader(p.eval_params.batch_size)
            latent_step_analysis_mode(model, dataloader, save_path, logger)
        else:
            logger.error(f'Unknown Mode {operation}')
            raise ValueError(f'Unknown Mode {operation}')


if __name__ == '__main__':
    main()
