import logging
import time

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader

from src.hvae.block import GenBlock, OutputBlock, SimpleGenBlock
from src.checkpoint import Checkpoint
from src.hparams import get_hparams
from src.elements.losses import StructureSimilarityIndexMap, get_reconstruction_loss, get_kl_loss
from src.elements.schedules import get_beta_schedule, get_gamma_schedule
from src.utils import linear_temperature, prepare_for_log, print_line, wandb_log_results

prms = get_hparams()
device = prms.model_params.device
kldiv_schedule = get_beta_schedule()
gamma_schedule = get_gamma_schedule()
reconstruction_loss = get_reconstruction_loss()
kl_divergence = get_kl_loss()
ssim_metric = StructureSimilarityIndexMap(image_channels=prms.data_params.shape[0])


def compute_loss(targets: tensor, distributions: dict, logits: tensor = None, step_n: int = 0) -> dict:
    """
    Compute loss for VAE (custom or default)
    based on Efficient-VDVAE paper

    :param targets: tensor, the target images
    :param distributions: list, the distributions for each generator block
    :param logits: tensor, the logits for the output block
    :param step_n: int, the current step number
    :return: dict, containing the loss values
    """
    # Use custom loss funtion if provided
    if prms.loss_params.custom_loss is not None:
        return prms.loss_params.custom_loss(targets=targets, logits=logits,
                                            distributions=distributions, step_n=step_n)

    output_distribution = distributions.pop('output')
    feature_matching_loss, avg_feature_matching_loss = reconstruction_loss(targets, output_distribution)

    global_variational_prior_losses = []
    avg_global_var_prior_losses = []
    for block_name, (prior, posterior) in distributions.items():
        if block_name == 'output' or posterior is None:
            continue
        loss, avg_loss = kl_divergence(prior, posterior)
        global_variational_prior_losses.append(loss)
        avg_global_var_prior_losses.append(avg_loss)

    global_variational_prior_losses = torch.stack(global_variational_prior_losses)
    global_variational_prior_loss = torch.sum(global_variational_prior_losses) \
        if not prms.loss_params.use_gamma_schedule \
        else gamma_schedule(global_variational_prior_losses,
                            avg_global_var_prior_losses,
                            step_n=step_n)
    global_var_loss = kldiv_schedule(step_n) * global_variational_prior_loss  # beta
    total_generator_loss = -feature_matching_loss + global_var_loss

    kl_div = torch.sum(global_variational_prior_losses) / np.log(2.)
    return dict(
        elbo=total_generator_loss,
        reconstruction_loss=feature_matching_loss,
        avg_reconstruction_loss=avg_feature_matching_loss,
        kl_div=kl_div,
        avg_var_prior_loss=torch.sum(torch.stack(avg_global_var_prior_losses)),
    )


def gradient_norm(net):
    """
    Compute the global norm of the gradients of the network parameters
    based on Efficient-VDVAE paper
    :param net: hVAE, the network
    """
    parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = torch.tensor(0.0)
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    return total_norm


def gradient_clip(net):
    """
    Clip the gradients of the network parameters
    based on Efficient-VDVAE paper
    """
    if prms.optimizer_params.clip_gradient_norm:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                    max_norm=prms.optimizer_params.gradient_clip_norm_value)
    else:
        total_norm = gradient_norm(net)
    return total_norm


def gradient_skip(global_norm):
    """
    Skip the gradient update if the global norm is too high
    based on Efficient-VDVAE paper
    :param global_norm: tensor, the global norm of the gradients
    """
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


def reconstruction_step(net, inputs: tensor, variates_masks=None, step_n=None, use_mean=False):
    """
    Perform a reconstruction with the given network and inputs
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param inputs: tensor, the input images
    :param variates_masks: list, the variate masks
    :param step_n: int, the current step number
    :param use_mean: use the mean of the distributions instead of sampling
    :return: tensor, tensor, dict, the output images, the computed features, the loss values
    """
    net.eval()
    with torch.no_grad():
        computed, distributions = net(inputs, variates_masks, use_mean=use_mean)
        if step_n is None:
            step_n = max(prms.loss_params.vae_beta_anneal_steps, prms.loss_params.gamma_max_steps) * 10.
        results = compute_loss(inputs, distributions, step_n=step_n)
        return computed, distributions, results


def reconstruct(net, dataset: DataLoader, variate_masks=None, logger: logging.Logger = None):
    """
    Reconstruct the images from the given dataset
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param dataset: DataLoader, the dataset
    :param variate_masks: list, the variate masks
    :param logger: logging.Logger, the logger
    :return: list, the input/output pairs
    """
    n_samples = prms.analysis_params.reconstruction.n_samples_for_reconstruction
    results, (original, output_samples, output_means) = \
        evaluate(net, dataset, n_samples=n_samples, variates_masks=variate_masks, logger=logger)
    return original, output_samples, output_means


def generation_step(net, temperatures: list):
    """
    Perform a generation with the given network
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param temperatures: list, the temperatures for each generator block
    :return: tensor, the generated images
    """
    computed, _ = net.sample_from_prior(prms.analysis_params.batch_size, temperatures=temperatures)
    return computed['output']


def generate(net, logger: logging.Logger):
    """
    Generate images with the given network
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param logger: logging.Logger, the logger
    :return: list, the generated images
    """
    all_outputs = list()
    for temp_i, temperature_setting in enumerate(prms.analysis_params.generation.temperature_settings):
        logger.info(f'Generating for temperature setting {temp_i:01d}')
        if isinstance(temperature_setting, list):
            temperatures = temperature_setting
        elif isinstance(temperature_setting, float):
            temperatures = [temperature_setting] * len(
                list(filter(lambda x: isinstance(x, (GenBlock, OutputBlock, SimpleGenBlock)),
                            net.generator.blocks)))
        elif isinstance(temperature_setting, tuple):
            # Fallback to function defined temperature. Function params are defined with 3 arguments in a tuple
            assert len(temperature_setting) == 3
            down_blocks = list(filter(lambda x: isinstance(x, (GenBlock, OutputBlock, SimpleGenBlock)),
                                      net.generator.blocks))
            temp_fn = linear_temperature(*(temperature_setting[1:]), n_layers=len(down_blocks))
            temperatures = [temp_fn(layer_i) for layer_i in range(len(down_blocks))]
        else:
            logger.error(f'Temperature Setting {temperature_setting} not interpretable!!')
            raise ValueError(f'Temperature Setting {temperature_setting} not interpretable!!')

        temp_outputs = list()
        for step in range(prms.analysis_params.generation.n_generation_batches):
            outputs = generation_step(net, temperatures=temperatures)
            temp_outputs.append(outputs)

            logger.info(f'Step: {step:04d}')
        all_outputs.append(temp_outputs)
    return all_outputs


def train_step(net, optimizer, schedule, inputs, step_n):
    """
    Perform a training step with the given network and inputs
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param optimizer: torch.optim.Optimizer, the optimizer
    :param schedule: torch.optim.lr_scheduler.LRScheduler, the scheduler
    :param inputs: tensor, the input images
    :param step_n: int, the current step number
    :return: tensor, dict, tensor, the output images, the loss values, the global norm of the gradients
    """
    computed, distributions = net(inputs)
    output_sample = computed['output']
    results = compute_loss(inputs, distributions, step_n=step_n)

    results["elbo"].backward()

    global_norm = gradient_clip(net)
    skip, gradient_skip_counter_delta = gradient_skip(global_norm)

    if not skip:
        optimizer.step()
        schedule.step()

    optimizer.zero_grad()
    return output_sample, results, global_norm, gradient_skip_counter_delta


def train(net,
          optimizer, schedule,
          train_loader: DataLoader, val_loader: DataLoader,
          start_step: int, wandb,
          checkpoint_path: str, logger: logging.Logger) -> None:
    """
    Train the network
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param optimizer: torch.optim.Optimizer, the optimizer
    :param schedule: torch.optim.lr_scheduler.LRScheduler, the scheduler
    :param train_loader: DataLoader, the training dataset
    :param val_loader: DataLoader, the validation dataset
    :param start_step: int, the step number to start from
    :param wandb: wandb run object
    :param checkpoint_path: str, the path to save the checkpoints to
    :param logger: logging.Logger, the logger
    :return: None
    """
    global_step = start_step
    gradient_skip_counter = 0.

    total_train_epochs = int(np.ceil(prms.train_params.total_train_steps / len(train_loader)))
    for epoch in range(total_train_epochs):
        for batch_n, train_inputs in enumerate(train_loader):
            net.train()
            global_step += 1
            train_inputs = train_inputs.to(device, non_blocking=True)
            start_time = time.time()
            train_outputs, train_results, global_norm, gradient_skip_counter_delta = \
                train_step(net, optimizer, schedule, train_inputs, global_step)
            end_time = round((time.time() - start_time), 2)
            gradient_skip_counter += gradient_skip_counter_delta

            train_results.update({
                "time": end_time,
                "beta": kldiv_schedule(global_step),
                "grad_norm": global_norm,
                "grad_skip_count": gradient_skip_counter,
            })
            train_results = prepare_for_log(train_results)
            logger.info((global_step,
                         ('Time/Step (sec)', end_time),
                         ('ELBO', round(train_results["elbo"], 4)),
                         ('Reconstruction Loss', round(train_results["reconstruction_loss"], 3)),
                         ('KL loss', round(train_results["kl_div"], 3))))
            wandb_log_results(wandb, train_results, global_step, mode='train')

            """
            EVALUATION AND CHECKPOINTING
            """
            net.eval()
            first_step = global_step == 0
            eval_time = global_step % prms.log_params.eval_interval_in_steps == 0
            checkpoint_time = global_step % prms.log_params.checkpoint_interval_in_steps == 0
            if eval_time or checkpoint_time:
                print_line(logger, newline_after=False)

            if eval_time or first_step:
                train_ssim = ssim_metric(train_inputs, train_outputs, global_batch_size=prms.train_params.batch_size)
                logger.info(
                    f'Train Stats | '
                    f'ELBO {train_results["elbo"]} | '
                    f'Reconstruction Loss {train_results["reconstruction_loss"]:.4f} |'
                    f'KL Div {train_results["kl_div"]:.4f} |'
                    f'SSIM: {train_ssim}')
                val_results, _ = evaluate(net, val_loader,
                                          n_samples=prms.eval_params.n_samples_for_validation,
                                          global_step=global_step, logger=logger)
                val_results = prepare_for_log(val_results)

                wandb_log_results(wandb, {'train_ssim': train_ssim}, global_step, mode='train')
                wandb_log_results(wandb, val_results, global_step, mode='validation')

            if checkpoint_time or first_step:
                # Save checkpoint (only if better than best)
                experiment = Checkpoint(global_step, net, optimizer, schedule, prms)
                path = experiment.save(checkpoint_path, wandb)
                logger.info(f'Saved checkpoint for global_step {global_step} to {path}')

            if eval_time or checkpoint_time:
                print_line(logger, newline_after=True)

            if global_step >= prms.train_params.total_train_steps:
                logger.info(f'Finished training after {global_step} steps!')
                return


def evaluate(net, val_loader: DataLoader, n_samples: int, global_step: int = None,
             use_mean=False, variates_masks=None, logger: logging.Logger = None) -> tuple:
    """
    Evaluate the network on the given dataset
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param val_loader: DataLoader, the dataset
    :param n_samples: number of samples to evaluate
    :param global_step: int, the current step number
    :param use_mean: use the mean of the distributions instead of sampling
    :param variates_masks: variates masks
    :param logger: logging.Logger, the logger
    :return: dict, tensor, tensor, the loss values, the output images, the input images
    """
    net.eval()

    val_step = 0
    global_results, original, output_samples, output_means = None, None, None, None

    for val_step, val_inputs in enumerate(val_loader):
        n_samples -= prms.eval_params.batch_size
        val_inputs = val_inputs.to(device, non_blocking=True)
        val_computed, val_distributions, val_results = \
            reconstruction_step(net, inputs=val_inputs, variates_masks=variates_masks,
                                step_n=global_step, use_mean=use_mean)
        val_outputs = val_computed["output"]
        val_ssim_per_batch = ssim_metric(val_inputs, val_outputs, global_batch_size=prms.eval_params.batch_size)

        val_inputs = val_inputs.detach().cpu()
        val_outputs = val_outputs.detach().cpu()
        val_output_means = val_distributions['output'].mean.detach().cpu()
        val_ssim_per_batch = val_ssim_per_batch.detach().cpu()
        if global_results is None:
            val_results["ssim"] = val_ssim_per_batch
            global_results = val_results
            original = val_inputs
            output_samples = val_outputs
            output_means = val_output_means
        else:
            val_results["ssim"] = val_ssim_per_batch
            global_results = {k: v + val_results[k] for k, v in global_results.items()}
            original = torch.cat((original, val_inputs), dim=0)
            output_samples = torch.cat((output_samples, val_outputs), dim=0)
            output_means = torch.cat((output_means, val_output_means), dim=0)

        if n_samples <= 0:
            break

    global_results = {k: v / (val_step + 1) for k, v in global_results.items()}
    global_results["avg_elbo"] = global_results["elbo"]
    global_results["elbo"] = global_results["kl_div"] + global_results["reconstruction_loss"]

    log = logger.info if logger is not None else print
    log(
        f'Validation Stats |'
        f' ELBO {global_results["elbo"]:.6f} |'
        f' Reconstruction Loss {global_results["reconstruction_loss"]:.4f} |'
        f' KL Div {global_results["kl_div"]:.4f} |'
        f' SSIM: {global_results["ssim"]:.6f}')

    return global_results, (original, output_samples, output_means)