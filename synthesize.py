import os
import pickle

from src.elements.losses import StructureSimilarityIndexMap
from src.utils import create_checkpoint_manager_and_load_if_exists, get_logdir, write_image_to_disk
from src.model import compute_per_dimension_divergence_stats, encode, generate, reconstruct

from hparams import *
import torch
import numpy as np


def divergence_stats(dataset, model, latents_folder):
    stats_filepath = os.path.join(latents_folder, 'div_stats.npy')
    per_dim_divs = compute_per_dimension_divergence_stats(dataset, model)
    np.save(stats_filepath, per_dim_divs.detach().cpu().numpy())


def encoding_mode(latents_folder, dataset, model):
    encodings = encode(dataset, model, latents_folder)
    print('Saving Encoded File')
    assert encodings['images'].keys() == encodings['latent_codes'][0].keys()
    with open(os.path.join(latents_folder, f'encodings_seed_{run_params.seed}.pkl'), 'wb') as handle:
        pickle.dump(encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generation_mode(artifacts_folder, model):
    artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/generated')
    os.makedirs(artifacts_folder, exist_ok=True)
    outputs = generate(model)
    for temp_i, temp_outputs in enumerate(outputs):
        sample_i = 0
        for step_output in outputs:
            for output in step_output:
                write_image_to_disk(os.path.join(artifacts_folder, f'setup-{temp_i:01d}-image-{sample_i:04d}.png'),
                                    output.detach().cpu().numpy())


def reconstruction_mode(test_dataset, model, artifacts_folder=None, latents_folder=None):
    reconstruct(test_dataset, model, artifacts_folder, latents_folder)


def synthesize(model, data, logdir, mode):
    artifacts_folder = os.path.join(logdir, 'synthesis-images')
    latents_folder = os.path.join(logdir, 'latents')
    os.makedirs(artifacts_folder, exist_ok=True)
    os.makedirs(latents_folder, exist_ok=True)

    if mode == 'reconstruction':
        reconstruction_mode(artifacts_folder, latents_folder, data, model)
    elif mode == 'generation':
        generation_mode(artifacts_folder, model)
    elif mode == 'encoding':
        encoding_mode(latents_folder, data, model)
    elif mode == 'div_stats':
        divergence_stats(data, model, latents_folder)
    else:
        raise ValueError(f'Unknown Mode {mode}')


"""
def synth_data():
    if data_params.dataset_source in ['ffhq', 'celebAHQ', 'celebA', 'custom']:
        return synth_generic_data()
    elif data_params.dataset_source == 'cifar-10':
        return synth_cifar_data()
    elif data_params.dataset_source == 'binarized_mnist':
        return synth_mnist_data()
    elif data_params.dataset_source == 'imagenet':
        return synth_imagenet_data()
    else:
        raise ValueError(f'Dataset {data_params.dataset_source} is not included.')


def encode_data():
    if data_params.dataset_source in ['ffhq', 'celebAHQ', 'celebA', 'custom']:
        return encode_generic_data()
    elif data_params.dataset_source == 'cifar-10':
        return encode_cifar_data()
    elif data_params.dataset_source == 'binarized_mnist':
        return encode_mnist_data()
    elif data_params.dataset_source == 'imagenet':
        return encode_imagenet_data()
    else:
        raise ValueError(f'Dataset {data_params.dataset_source} is not included.')


def stats_data():
    if data_params.dataset_source in ['ffhq', 'celebAHQ', 'celebA', 'custom']:
        return stats_generic_data()
    elif data_params.dataset_source == 'cifar-10':
        return stats_cifar_data()
    elif data_params.dataset_source == 'binarized_mnist':
        return stats_mnist_data()
    elif data_params.dataset_source == 'imagenet':
        return stats_imagenet_data()
    else:
        raise ValueError(f'Dataset {data_params.dataset_source} is not included.')
"""

def main():
    model = run_params.model
    model = model.to(model.device)

    with torch.no_grad():
        _ = model(torch.ones((1, data_params.channels, data_params.target_res, data_params.target_res)).cuda())
    # count_parameters(model)
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

    if synthesis_params.synthesis_mode == 'reconstruction':
        data_loader = synth_data()
    elif synthesis_params.synthesis_mode == 'encoding':
        data_loader = encode_data()
    elif synthesis_params.synthesis_mode == 'div_stats':
        data_loader = stats_data()
    elif synthesis_params.synthesis_mode == 'generation':
        data_loader = None
    else:
        raise ValueError(f'Unknown Mode {synthesis_params.synthesis_mode}')

    # Synthesis using pretrained model
    logdir = get_logdir()
    synthesize(model, data_loader, logdir, mode=synthesis_params.synthesis_mode)


if __name__ == '__main__':
    main()
