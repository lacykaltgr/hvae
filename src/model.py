import time
import torch.distributed as dist
import os
from tqdm import tqdm

from src.elements.losses import StructureSimilarityIndexMap, KLDivergenceStats
from src.elements.schedules import *
from utils import tensorboard_log, plot_image, get_variate_masks, write_image_to_disk, reshape_distribution, transpose_dicts, one_hot


from hparams import *

device = run_params.device

if loss_params.variation_schedule == 'Logistic':
    kldiv_schedule = LogisticBetaSchedule(
        activation_step=loss_params.vae_beta_activation_steps,
        growth_rate=loss_params.vae_beta_growth_rate)
elif loss_params.variation_schedule == 'Linear':
    kldiv_schedule = LinearBetaSchedule(
        anneal_start=loss_params.vae_beta_anneal_start,
        anneal_steps=loss_params.vae_beta_anneal_steps,
        beta_min=loss_params.loss.vae_beta_min)
else:
    kldiv_schedule = lambda x: torch.as_tensor(1.)


def compute_loss(targets, predictions, stats):
    from elements.layers import DmolNet
    distortion_per_pixel = DmolNet().nll(predictions, targets)
    rate_per_pixel = torch.zeros_like(distortion_per_pixel)
    ndims = np.prod(targets.shape[1:])
    for statdict in stats:
        rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
    rate_per_pixel /= ndims
    elbo = (distortion_per_pixel + rate_per_pixel).mean()
    return dict(elbo=elbo, distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean())


def _global_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = torch.tensor(0.0)
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    return total_norm


def gradient_clip(model):
    if optimizer_params.clip_gradient_norm:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=optimizer_params.gradient_clip_norm_value)
    else:
        total_norm = _global_norm(model)
    return total_norm


def gradient_skip(global_norm):
    if optimizer_params.gradient_skip:
        if torch.any(torch.isnan(global_norm)) or global_norm >= optimizer_params.gradient_skip_threshold:
            skip = True
            gradient_skip_counter_delta = 1.
        else:
            skip = False
            gradient_skip_counter_delta = 0.
    else:
        skip = False
        gradient_skip_counter_delta = 0.

    return skip, gradient_skip_counter_delta


def eval_step(model, inputs):
    with torch.no_grad():
        predictions, stats = model(inputs)
        results = compute_loss(inputs, predictions, stats)
        outputs = model.sample(predictions)
    return outputs, results


def reconstruction_step(model, inputs, variates_masks=None, mode='recon'):
    model.eval()
    with torch.no_grad():
        predictions, stats = model(inputs, variates_masks)

        if mode == 'recon':
            results = compute_loss(inputs, predictions, stats)
            outputs = model.sample(predictions)

            return outputs, stats
        elif mode == 'encode':
            return stats
        else:
            raise ValueError(f'Unknown Mode {mode}')


def reconstruct(test_dataset, model, artifacts_folder=None, latents_folder=None):
    ssim_metric = StructureSimilarityIndexMap(image_channels=data_params.channels)
    if artifacts_folder is not None:
        artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/reconstructed')
        os.makedirs(artifacts_folder, exist_ok=True)

    if synthesis_params.mask_reconstruction:
        div_stats = np.load(os.path.join(latents_folder, 'div_stats.npy'))
        variate_masks = get_variate_masks(div_stats).astype(np.float32)
    else:
        variate_masks = None

    nelbos, ssims = 0., 0.
    sample_i = 0

    io_pairs = list()
    for step, inputs in enumerate(test_dataset):
        inputs = inputs.to(device)
        outputs, reconstruction_loss, kl_div = reconstruction_step(model, inputs, variates_masks=variate_masks)
        targets = inputs

        nelbo = reconstruction_loss + kl_div
        ssim_per_batch = ssim_metric(targets, outputs, global_batch_size=synthesis_params.batch_size)
        ssims += ssim_per_batch
        nelbos += nelbo

        # Save images to disk
        if artifacts_folder is not None:
            for batch_i, (target, output) in enumerate(zip(targets, outputs)):
                if synthesis_params.save_target_in_reconstruction:
                    write_image_to_disk(
                        os.path.join(artifacts_folder, f'target-{sample_i:04d}.png'),
                        target.detach().cpu().numpy())
                write_image_to_disk(
                    os.path.join(artifacts_folder, f'image-{sample_i:04d}.png'),
                    output.detach().cpu().numpy())
                io_pairs.append((target, output))

                sample_i += 1
        print(f'Step: {step:04d}  | NELBO: {nelbo:.4f} | Reconstruction: {reconstruction_loss:.4f} | '
              f'kl_div: {kl_div:.4f}| SSIM: {ssim_per_batch:.4f} ', end='\r')

    nelbo = nelbos / (step + 1)
    ssim = ssims / (step + 1)
    print()
    print()
    print('===========================================')
    print(f'NELBO: {nelbo:.6f} | SSIM: {ssim:.6f}')
    return io_pairs


def generation_step(model, temperatures):
    y, prior_zs = model.sample_from_prior(synthesis_params.batch_size, temperatures=temperatures)
    outputs = model.sample(y)
    return outputs, prior_zs


def generate(model):
    all_outputs = list()
    # Generation supports runs with several temperature configs to avoid rebuilding each time
    for temp_i, temperature_setting in enumerate(synthesis_params.temperature_settings):
        print(f'Generating for temperature setting {temp_i:01d}')
        # Make per layer temperatures of the setting
        if isinstance(temperature_setting, list):
            # Use defined list of temperatures
            assert len(temperature_setting) == len(hparams.model.down_strides)
            temperatures = temperature_setting

        elif isinstance(temperature_setting, float):
            # Use the same float valued temperature for all layers
            temperatures = [temperature_setting] * len(hparams.model.down_strides)

        elif isinstance(temperature_setting, tuple):
            # Fallback to function defined temperature. Function params are defined with 3 arguments in a tuple
            assert len(temperature_setting) == 3
            temp_fn = getattr(temperature_functions, temperature_setting[0])(
                temperature_setting[1],
                temperature_setting[2], n_layers=len(hparams.model.down_strides))
            temperatures = [temp_fn(layer_i) for layer_i in range(len(hparams.model.down_strides))]

        else:
            raise ValueError(f'Temperature Setting {temperature_setting} not interpretable!!')

        temp_outputs = list()
        for step in range(synthesis_params.n_generation_batches):
            outputs, prior_zs = generation_step(model, temperatures=temperatures)
            temp_outputs.append(outputs)

            print(f'Step: {step:04d} ', end='\r')
            print()
        all_outputs.append(temp_outputs)
    return all_outputs


def encode(dataset, model, latents_folder=None):

    #TODO: nem látom miért kéne ezt elvárni
    if not os.path.isfile(os.path.join(latents_folder, 'div_stats.npy')):
        raise FileNotFoundError('No div_stats found')

    # Load div stats from disk
    div_stats = np.load(os.path.join(latents_folder, 'div_stats.npy'))
    variate_masks = get_variate_masks(div_stats)

    encodings = {'images': {}, 'latent_codes': {}}
    model = model.eval()
    print('Starting Encoding mode \n')
    with torch.no_grad():
        for step, (inputs, filenames) in enumerate(tqdm(dataset)):
            inputs = inputs.to(device, non_blocking=True)
            # posterior_dist_list : n_layers, 2 (mean,std), batch_size, n_variates, H, W (List of List of Tensors)
            posterior_dist_list = reconstruction_step(model, inputs, variates_masks=variate_masks, mode='encode')

            # If the mask states all variables of a layer are not effective we don't collect any latents from that layer
            # n_layers , batch_size, [H, W, n_variates, 2]
            dist_dict = {}
            for i, (dist_list, variate_mask) in enumerate(zip(posterior_dist_list, variate_masks)):
                if variate_mask.any():
                    x = reshape_distribution(dist_list, variate_mask).detach().cpu().numpy()
                    v = {name: xa for name, xa in zip(filenames, list(x))}
                    dist_dict[i] = v

            if encodings['latent_codes'] == {}:
                # Put first batch
                encodings['latent_codes'] = dist_dict
            else:
                # Update files of each layer
                assert dist_dict.keys() == encodings['latent_codes'].keys()
                for layer_key, layer_dict in dist_dict.items():
                    encodings['latent_codes'][layer_key].update(layer_dict)

            inputs = inputs.detach().cpu().numpy()
            assert len(filenames) == len(inputs)
            for filename, input_image in zip(filenames, inputs):
                encodings['images'][filename] = input_image

    encodings['latent_codes'] = transpose_dicts(encodings['latent_codes'])
    return encodings


def update_ema(vae, ema_vae, ema_rate):
    for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def _compiled_train_step(model, inputs):
    predictions, stats = model(inputs)
    results = compute_loss(inputs, predictions, stats)

    results["elbo"].backward()

    total_norm = gradient_clip(model)
    skip, gradient_skip_counter_delta = gradient_skip(total_norm)

    outputs = model.module.top_down.sample(predictions)

    return outputs, results, total_norm, gradient_skip_counter_delta, skip


def train_step(model, ema_model, optimizer, inputs):
    outputs, results, global_norm, gradient_skip_counter_delta, skip = _compiled_train_step(model, inputs)

    if not skip:
        optimizer.step()
        update_ema(model, ema_model, train_params.ema_decay)

    optimizer.zero_grad()
    return outputs, results, global_norm, gradient_skip_counter_delta


def train(model, ema_model, optimizer, schedule, train_dataset, val_dataset, checkpoint_start, tb_writer_train,
          tb_writer_val, checkpoint_path, device, rank):
    ssim_metric = StructureSimilarityIndexMap(image_channels=data_params.channels)
    global_step = checkpoint_start
    gradient_skip_counter = 0.

    model.train()

    # let all processes sync up before starting with a new epoch of models
    total_train_epochs = int(np.ceil(train_params.total_train_steps / len(train_dataset)))
    val_epoch = 0
    for epoch in range(0, total_train_epochs):
        train_dataset.sampler.set_epoch(epoch)
        if rank == 0:
            print(f'\nEpoch: {epoch + 1}')
        dist.barrier()
        for batch_n, train_inputs in enumerate(train_dataset):
            # update global step
            global_step += 1

            train_inputs = train_inputs.to(device, non_blocking=True)
            # torch.cuda.synchronize()
            start_time = time.time()
            train_outputs, results, global_norm, gradient_skip_counter_delta = train_step(
                model, ema_model, optimizer, train_inputs)
            # torch.cuda.synchronize()
            end_time = round((time.time() - start_time), 2)
            schedule.step()

            gradient_skip_counter += gradient_skip_counter_delta

            train_var_loss = np.sum([v.detach().cpu() for v in train_global_varprior_losses])
            train_nelbo = results["elbo"]
            # global_norm = global_norm / (hparams.data.target_res * hparams.data.target_res * hparams.data.channels)
            if rank == 0:
                print(global_step,
                      ('Time/Step (sec)', end_time),
                      ('Reconstruction Loss', round(train_feature_matching_loss.detach().cpu().item(), 3)),
                      ('KL loss', round(train_kl_div.detach().cpu().item(), 3)),
                      ('nelbo', round(train_nelbo.detach().cpu().item(), 4)),
                      ('average KL loss', round(train_var_loss.item(), 3)),
                      ('Beta', round(kldiv_schedule(global_step).detach().cpu().item(), 4)),
                      ('N° active groups', np.sum([v.detach().cpu() >= eval_params.latent_active_threshold
                                                   for v in train_global_varprior_losses])),
                      ('GradNorm', round(global_norm.detach().cpu().item(), 1)),
                      ('GradSkipCount', gradient_skip_counter),
                      # ('learning_rate', optimizer.param_groups[0]['lr']),
                      end="\r")

            """
            CHECKPOINTING AND EVALUATION
            """
            if global_step % train_params.checkpoint_and_eval_interval_in_steps == 0 or global_step == 0:
                model.eval()
                # Compute SSIM at the end of the global_step
                train_ssim = ssim_metric(train_inputs, train_outputs,
                                         global_batch_size=train_params.batch_size // train_params.num_gpus)
                if rank == 0:
                    train_losses = {'reconstruction_loss': train_feature_matching_loss,
                                    'kl_div': train_kl_div,
                                    'average_kl_div': train_var_loss,
                                    'variational_beta': kldiv_schedule(global_step),
                                    'ssim': train_ssim,
                                    'nelbo': train_nelbo}

                    train_losses.update({f'latent_kl_{i}': v for i, v in enumerate(train_global_varprior_losses)})

                    print(
                        f'\nTrain Stats for global_step {global_step} | NELBO {train_nelbo:.6f} | '
                        f'SSIM: {train_ssim:.6f}')

                # Evaluate model
                val_feature_matching_losses = 0
                val_global_varprior_losses = None
                val_ssim = 0
                val_kl_divs = 0
                val_epoch += 1

                val_dataset.sampler.set_epoch(val_epoch)
                for val_step, val_inputs in enumerate(val_dataset):
                    # Val inputs contains val_Data and filenames
                    val_inputs = val_inputs.to(device, non_blocking=True)
                    val_outputs, results, val_means, val_log_scales = eval_step(model, inputs=val_inputs)

                    val_ssim_per_batch = ssim_metric(val_inputs, val_outputs,
                                                     global_batch_size=eval_params.batch_size // train_params.num_gpus)

                    val_feature_matching_losses += val_feature_matching_loss
                    val_ssim += val_ssim_per_batch
                    val_kl_divs += val_kl_div

                    if val_global_varprior_losses is None:
                        val_global_varprior_losses = val_global_varprior_loss
                    else:
                        val_global_varprior_losses = [u + v for u, v in
                                                      zip(val_global_varprior_losses, val_global_varprior_loss)]

                val_feature_matching_loss = val_feature_matching_losses / (val_step + 1)
                val_ssim = val_ssim / (val_step + 1)
                val_kl_div = val_kl_divs / (val_step + 1)

                val_global_varprior_losses = [v / (val_step + 1) for v in val_global_varprior_losses]

                val_varprior_loss = np.sum([v.detach().cpu() for v in val_global_varprior_losses])
                val_nelbo = val_kl_div + val_feature_matching_loss

                if rank == 0:
                    val_losses = {'reconstruction_loss': val_feature_matching_loss,
                                  'kl_div': val_kl_div,
                                  'average_kl_div': val_varprior_loss,
                                  'ssim': val_ssim,
                                  'nelbo': val_nelbo}

                    val_losses.update({f'latent_kl_{i}': v for i, v in enumerate(val_global_varprior_losses)})

                    print(
                        f'Validation Stats for global_step {global_step} |'
                        f' Reconstruction Loss {val_feature_matching_loss:.4f} |'
                        f' KL Div {val_kl_div:.4f} |'f'NELBO {val_nelbo:.6f} |'
                        f'SSIM: {val_ssim:.6f}')

                    # Save checkpoint (only if better than best)
                    print(f'Saving checkpoint for global_step {global_step}..')

                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'ema_model_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': schedule.state_dict()
                    }, checkpoint_path)

                    # Tensorboard logging
                    print('Logging to Tensorboard..')
                    train_losses['skips_count'] = gradient_skip_counter / train_params.total_train_steps
                    tensorboard_log(model, optimizer, global_step, tb_writer_train, train_losses, train_outputs,
                                    train_inputs,
                                    global_norm=global_norm)
                    tensorboard_log(model, optimizer, global_step, tb_writer_val, val_losses, val_outputs, val_inputs,
                                    means=val_means, log_scales=val_log_scales, mode='val')

                    # Save artifacts
                    plot_image(train_outputs[0], train_inputs[0], global_step, writer=tb_writer_train)
                    plot_image(val_outputs[0], val_inputs[0], global_step, writer=tb_writer_val)
                model.train()
            dist.barrier()

            if global_step >= train_params.total_train_steps:
                print(f'Finished training after {global_step} steps!')
                exit()


def compute_per_dimension_divergence_stats(dataset, model):
    kl_stats = KLDivergenceStats()

    per_dim_divs = None
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(dataset)):
            inputs = inputs.to(device, non_blocking=True)
            predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)
            kl_div = kl_stats(posterior_dist_list, prior_kl_dist_list, synthesis_params.batch_size)

            if per_dim_divs is None:
                per_dim_divs = kl_div
            else:
                per_dim_divs += kl_div

    per_dim_divs /= (step + 1)
    return per_dim_divs


def sample_from_mol(logits, min_pix_value=0, max_pix_value=255):

    def _compute_scales(logits):
        from torch import nn
        softplus = nn.Softplus(beta=hparams.model.output_gradient_smoothing_beta)
        if hparams.model.output_distribution_base == 'std':
            scales = torch.maximum(softplus(logits), torch.as_tensor(np.exp(hparams.loss.min_mol_logscale)))

        elif hparams.model.output_distribution_base == 'logstd':
            log_scales = torch.maximum(logits, torch.as_tensor(np.array(hparams.loss.min_mol_logscale)))
            scales = torch.exp(hparams.model.output_gradient_smoothing_beta * log_scales)
        else:
            raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')

        return scales

    B, _, H, W = logits.size()  # B, M*(3*C+1), H, W,

    logit_probs = logits[:, :model.num_output_mixtures, :, :]  # B, M, H, W
    l = logits[:, hparams.model.num_output_mixtures:, :, :]  # B, M*C*3 ,H, W
    l = l.reshape(B, data_params.channels, 3 * hparams.model.num_output_mixtures, H, W)  # B, C, 3 * M, H, W

    model_means = l[:, :, :hparams.model.num_output_mixtures, :, :]  # B, C, M, H, W
    scales = _compute_scales(
        l[:, :, hparams.model.num_output_mixtures: 2 * hparams.model.num_output_mixtures, :, :])  # B, C, M, H, W
    model_coeffs = torch.tanh(
        l[:, :, 2 * hparams.model.num_output_mixtures: 3 * hparams.model.num_output_mixtures, :,
        :])  # B, C, M, H, W

    # Gumbel-max to select the mixture component to use (per pixel)
    gumbel_noise = -torch.log(-torch.log(
        torch.Tensor(logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
    logit_probs = logit_probs / synthesis_params.output_temperature + gumbel_noise
    lambda_ = one_hot(torch.argmax(logit_probs, dim=1), logit_probs.size()[1], dim=1)  # B, M, H, W

    lambda_ = lambda_.unsqueeze(1)  # B, 1, M, H, W

    # select logistic parameters
    means = torch.sum(model_means * lambda_, dim=2)  # B, C, H, W
    scales = torch.sum(scales * lambda_, dim=2)  # B, C, H, W
    coeffs = torch.sum(model_coeffs * lambda_, dim=2)  # B, C,  H, W

    # Samples from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()
    x = means + scales * synthesis_params.output_temperature * (
            torch.log(u) - torch.log(1. - u))  # B, C,  H, W

    # Autoregressively predict RGB
    x0 = torch.clamp(x[:, 0:1, :, :], min=min_pix_value, max=max_pix_value)  # B, 1, H, W
    x1 = torch.clamp(x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, min=min_pix_value,
                     max=max_pix_value)  # B, 1, H, W
    x2 = torch.clamp(x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1,
                     min=min_pix_value,
                     max=max_pix_value)  # B, 1, H, W

    x = torch.cat([x0, x1, x2], dim=1)  # B, C, H, W
    return x


