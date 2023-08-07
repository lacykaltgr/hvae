import logging
import os
import time

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.block import DecBlock, TopBlock
from experiment import Experiment
from hparams import get_hparams
from src.elements.losses import StructureSimilarityIndexMap, get_reconstruction_loss, get_kl_loss
from src.elements.samplers import get_output_sampler
from src.elements.schedules import get_beta_schedule, get_gamma_schedule

from src.utils import tensorboard_log, get_variate_masks, write_image_to_disk, linear_temperature, \
    prepare_for_log

prms = get_hparams()
device = prms.model_params.device
kldiv_schedule = get_beta_schedule()
gamma_schedule = get_gamma_schedule()
reconstruction_loss = get_reconstruction_loss()
kl_divergence = get_kl_loss()


def compute_loss(targets: tensor, predictions: tensor, distributions: list, step_n: int) -> dict:
    # Use custom loss funtion if provided
    if prms.loss_params.custom_loss is not None:
        return prms.loss_params.custom_loss(targets=targets, predictions=predictions, distributions=distributions,
                                         step_n=step_n)

    feature_matching_loss, avg_feature_matching_loss, means, log_scales \
        = reconstruction_loss(targets=targets, predictions=predictions)

    global_variational_prior_losses = []
    avg_global_var_prior_losses = []
    for p, q in distributions:
        q_mu, q_sigma = q
        p_mu, p_sigma = p
        loss, avg_loss = kl_divergence(q_mu=q_mu, q_sigma=q_sigma, p_mu=p_mu, p_sigma=p_sigma)
        global_variational_prior_losses.append(loss)
        avg_global_var_prior_losses.append(avg_loss)
    global_variational_prior_losses = torch.stack(global_variational_prior_losses)
    global_variational_prior_loss = torch.sum(global_variational_prior_losses) \
        if not prms.loss_params.use_gamma_schedule \
        else gamma_schedule(global_variational_prior_losses,
                            avg_global_var_prior_losses,
                            step_n=step_n)
    global_var_loss = kldiv_schedule(step_n) * global_variational_prior_loss  # beta
    total_generator_loss = feature_matching_loss + global_var_loss
    scalar = np.log(2.)
    # True bits/dim kl div
    kl_div = torch.sum(global_variational_prior_losses) / scalar
    return dict(
        elbo=total_generator_loss,
        reconstruction_loss=feature_matching_loss,
        avg_rec_loss=avg_feature_matching_loss,
        avg_var_prior_losses=avg_global_var_prior_losses,
        means=means,
        log_scales=log_scales,
        kl_div=kl_div
    )


def global_norm(net):
    parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = torch.tensor(0.0)
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    return total_norm


def gradient_clip(net):
    if prms.optimizer_params.clip_gradient_norm:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                    max_norm=prms.optimizer_params.gradient_clip_norm_value)
    else:
        total_norm = global_norm(net)
    return total_norm


def gradient_skip(global_norm):
    if prms.optimizer_params.gradient_skip:
        if torch.any(torch.isnan(global_norm)) or global_norm >= prms.optimizer_params.gradient_skip_threshold:
            skip = True
            gradient_skip_counter_delta = 1.
        else:
            skip = False
            gradient_skip_counter_delta = 0.
    else:
        skip = False
        gradient_skip_counter_delta = 0.
    return skip, gradient_skip_counter_delta


def reconstruction_step(net, inputs: tensor, variates_masks=None, step_n=None):
    net.eval()
    with torch.no_grad():
        predictions, computed, distributions = net(inputs, variates_masks)
        if step_n is None:
            step_n = max(prms.loss_params.vae_beta_anneal_steps, prms.loss_params.gamma_max_steps) * 10.
        results = compute_loss(inputs, predictions, distributions, step_n=step_n)
        outputs = net.sample(predictions)
        return outputs, computed, results


def reconstruct(net, dataset: DataLoader, artifacts_folder=None, latents_folder=None, logger: logging.Logger = None):
    ssim_metric = StructureSimilarityIndexMap(image_channels=prms.data_params.channels)
    if artifacts_folder is not None:
        artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/reconstructed')
        os.makedirs(artifacts_folder, exist_ok=True)
    if prms.synthesis_params.mask_reconstruction:
        div_stats = np.load(os.path.join(latents_folder, 'div_stats.npy'))
        variate_masks = get_variate_masks(div_stats).astype(np.float32)
    else:
        variate_masks = None

    nelbos, ssims = 0., 0.
    sample_i = 0

    io_pairs = list()
    for step, inputs in enumerate(dataset):
        inputs = inputs.to(device)
        outputs, _, loss = reconstruction_step(net, inputs, variates_masks=variate_masks)
        targets = inputs

        reconstruction_loss = loss['reconstruction_loss']
        kl_div = loss['kl_div']
        nelbo = reconstruction_loss + kl_div
        ssim_per_batch = ssim_metric(targets, outputs, global_batch_size=prms.synthesis_params.batch_size)
        ssims += ssim_per_batch
        nelbos += nelbo

        # Save images to disk
        if artifacts_folder is not None:
            for batch_i, (target, output) in enumerate(zip(targets, outputs)):
                if prms.synthesis_params.save_target_in_reconstruction:
                    write_image_to_disk(
                        os.path.join(artifacts_folder, f'target-{sample_i:04d}.png'),
                        target.detach().cpu().numpy())
                write_image_to_disk(
                    os.path.join(artifacts_folder, f'image-{sample_i:04d}.png'),
                    output.detach().cpu().numpy())
                io_pairs.append((target, output))

                sample_i += 1
        logger.info(
            f'Step: {step:04d}  | NELBO: {nelbo:.4f} | Reconstruction: {reconstruction_loss:.4f} | '
            f'kl_div: {kl_div:.4f}| SSIM: {ssim_per_batch:.4f} ')

    nelbo = nelbos / (step + 1)
    ssim = ssims / (step + 1)
    logger.info('===========================================')
    logger.info(f'NELBO: {nelbo:.6f} | SSIM: {ssim:.6f}')
    return io_pairs


def generation_step(net, temperatures: list):
    outputs, computed = net.sample_from_prior(prms.synthesis_params.batch_size, temperatures=temperatures)
    samples = net.sample(outputs)
    return samples


def generate(net, logger: logging.Logger):
    all_outputs = list()
    for temp_i, temperature_setting in enumerate(prms.synthesis_params.temperature_settings):
        logger.info(f'Generating for temperature setting {temp_i:01d}')
        if isinstance(temperature_setting, list):
            temperatures = temperature_setting
        elif isinstance(temperature_setting, float):
            temperatures = [temperature_setting] * len(
                list(filter(lambda x: isinstance(x, (DecBlock, TopBlock)), net.decoder._decoder_blocks)))
        elif isinstance(temperature_setting, tuple):
            # Fallback to function defined temperature. Function params are defined with 3 arguments in a tuple
            assert len(temperature_setting) == 3
            down_blocks = list(filter(lambda x: isinstance(x, (DecBlock, TopBlock)), net.decoder._decoder_blocks))
            temp_fn = linear_temperature(*temperature_setting, n_layers=down_blocks)
            temperatures = [temp_fn(layer_i) for layer_i in range(len(down_blocks))]
        else:
            logger.error(f'Temperature Setting {temperature_setting} not interpretable!!')
            raise ValueError(f'Temperature Setting {temperature_setting} not interpretable!!')

        temp_outputs = list()
        for step in range(prms.synthesis_params.n_generation_batches):
            outputs = generation_step(net, temperatures=temperatures)
            temp_outputs.append(outputs)

            logger.info(f'Step: {step:04d}')
        all_outputs.append(temp_outputs)
    return all_outputs


def train_step(net, optimizer, inputs, step_n):
    predictions, _, distibutions = net(inputs)
    outputs = net.sample(predictions)
    results = compute_loss(inputs, predictions, distibutions, step_n=step_n)

    results["elbo"].backward()

    global_norm = gradient_clip(net)
    skip, gradient_skip_counter_delta = gradient_skip(global_norm)

    if not skip:
        optimizer.step()

    optimizer.zero_grad()
    return outputs, results, global_norm, gradient_skip_counter_delta


def train(net,
          optimizer, schedule,
          train_loader: DataLoader, val_loader: DataLoader,
          checkpoint_start_step: int,
          tb_writer_train: SummaryWriter, tb_writer_val: SummaryWriter,
          checkpoint_path: str, logger: logging.Logger) -> None:
    ssim_metric = StructureSimilarityIndexMap(image_channels=prms.data_params.channels)
    global_step = checkpoint_start_step
    gradient_skip_counter = 0.

    net.train()
    total_train_epochs = int(np.ceil(prms.train_params.total_train_steps / len(train_loader)))
    for epoch in range(total_train_epochs):
        for batch_n, train_inputs in enumerate(train_loader):
            global_step += 1
            train_inputs = train_inputs.to(device, non_blocking=True)

            start_time = time.time()
            train_outputs, train_results, global_norm, gradient_skip_counter_delta = \
                train_step(net, optimizer, train_inputs, global_step)
            end_time = round((time.time() - start_time), 2)
            schedule.step()

            gradient_skip_counter += gradient_skip_counter_delta

            train_results = prepare_for_log(train_results)
            logger.info((global_step,
                         ('Time/Step (sec)', end_time),
                         ('Reconstruction Loss', round(train_results["distortion"], 3)),
                         ('KL loss', round(train_results["rate"], 3)),
                         ('nelbo', round(train_results["elbo"], 4)),
                         ('average KL loss', round(train_results["train_var_loss"], 3)),
                         ('Beta', round(kldiv_schedule(global_step).detach().cpu().item(), 4)),
                         ('N° active groups', train_results["active_groups"]),
                         ('GradNorm', round(global_norm.detach().cpu().item(), 1)),
                         ('GradSkipCount', gradient_skip_counter),))

            """
            EVALUATION AND CHECKPOINTING
            """
            net.eval()
            if global_step % prms.train_params.eval_interval_in_steps == 0 or global_step == 0:
                train_ssim = ssim_metric(train_inputs, train_outputs, global_batch_size=prms.train_params.batch_size)
                logger.info(
                    f'Train Stats for global_step {global_step} | NELBO {train_results["elbo"]} | 'f'SSIM: {train_ssim}')
                val_results, val_outputs, val_inputs = evaluate(net, val_loader, global_step)
                # Tensorboard logging
                logger.info('Logging to Tensorboard..')
                tensorboard_log(net, optimizer, global_step,
                                tb_writer_train, train_results,
                                train_outputs, train_inputs, global_norm=global_norm)
                tensorboard_log(net, optimizer, global_step,
                                tb_writer_val, val_results,
                                val_outputs, val_inputs,
                                means=val_results["means"], log_scales=val_results["log_scales"],
                                mode='val')

            if global_step % prms.train_params.checkpoint_interval_in_steps == 0 or global_step == 0:
                # Save checkpoint (only if better than best)
                logger.info(f'Saving checkpoint for global_step {global_step}..')

                experiment = Experiment(global_step, net, prms)
                experiment.save(checkpoint_path)
            net.train()

            if global_step >= prms.train_params.total_train_steps:
                logger.info(f'Finished training after {global_step} steps!')
                return


def compute_per_dimension_divergence_stats(net, dataset: DataLoader) -> tensor:
    per_dim_divs = None
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(dataset)):
            inputs = inputs.to(device, non_blocking=True)
            _, _, distributions = net(inputs)
            avg_losses = []
            for p, q in distributions:
                q_mu, q_sigma = q
                p_mu, p_sigma = p
                _, avg_loss = prms.loss_params.kl_divergence(q_mu=q_mu, q_sigma=q_sigma, p_mu=p_mu, p_sigma=p_sigma)
                avg_losses.append(avg_loss)
            kl_div = torch.stack(avg_losses)
            per_dim_divs = kl_div if per_dim_divs is None else per_dim_divs + kl_div
            if step > prms.synthesis_params.div_stats_subset_ratio * len(dataset):
                break
    per_dim_divs /= (step + 1)
    return per_dim_divs


def sample(logits):
    return get_output_sampler()(logits)


ssim_metric = StructureSimilarityIndexMap(image_channels=prms.data_params.channels)


def evaluate(net, val_loader: DataLoader, global_step: int = None, logger: logging.Logger = None) -> tuple:
    net.eval()

    """
    elbo=total_generator_loss,
    reconstruction_loss=feature_matching_loss,
    avg_rec_loss=avg_feature_matching_loss,
    avg_var_prior_losses=avg_global_var_prior_losses,
    means=means,
    log_scales=log_scales,
    kl_div=kl_div
    """

    val_inputs, val_outputs, val_results = None, None, None
    val_step = 0
    val_feature_matching_losses = 0
    val_global_varprior_losses = None
    val_ssim = 0
    val_kl_divs = 0
    for val_step, val_inputs in enumerate(val_loader):
        val_inputs = val_inputs.to(device, non_blocking=True)
        val_outputs, val_computed, val_results = \
            reconstruction_step(net, inputs=val_inputs, step_n=global_step)

        val_ssim_per_batch = ssim_metric(val_inputs, val_outputs, global_batch_size=prms.eval_params.batch_size)
        val_feature_matching_losses += val_results["reconstruction_loss"]
        val_ssim += val_ssim_per_batch
        val_kl_divs += val_results["kl_div"]
        val_global_varprior_losses = val_results["avg_var_prior_losses"] \
            if val_global_varprior_losses is None \
            else [u + v for u, v in zip(val_global_varprior_losses, val_results["avg_var_prior_losses"])]

    global_results = val_results
    global_results["feature_matching_loss"] \
        = val_feature_matching_losses / (val_step + 1)
    global_results["ssim"] \
        = val_ssim / (val_step + 1)
    global_results["kl_div"] \
        = val_kl_divs / (val_step + 1)

    global_results["avg_var_prior_losses"] \
        = val_global_varprior_losses
    global_results["global_varprior_losses"] \
        = [v / (val_step + 1) for v in val_global_varprior_losses]
    global_results["varprior_loss"] \
        = np.sum([v.detach().cpu() for v in val_global_varprior_losses])
    global_results["elbo"] = global_results["kl_div"] + global_results["feature_matching_loss"]

    global_results = prepare_for_log(global_results)
    val_results.update({f'latent_kl_{i}': v for i, v in enumerate(val_global_varprior_losses)})

    logger.info(
        f' Validation Stats|' if global_step is None else
        f' Validation Stats for global_step {global_step} |'
        f' Reconstruction Loss {global_results["reconstruction_loss"]:.4f} |'
        f' KL Div {global_results["kl_div"]:.4f} |'f'NELBO {global_results["elbo"]:.6f} |'
        f' SSIM: {global_results["ssim"]:.6f}')

    return val_results, val_outputs, val_inputs
