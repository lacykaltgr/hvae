import time
import torch.distributed as dist
from src.elements.schedules import *
from utils import tensorboard_log, plot_image


from hparams import *


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


def compute_loss(targets, predictions, posterior_dist_list, prior_kl_dist_list, step_n, global_batch_size):

    feature_matching_loss, avg_feature_matching_loss, means, log_scales = loss_params.reconstruction_loss(
        targets=targets,
        logits=predictions,
        global_batch_size=global_batch_size)

    global_variational_prior_losses, avg_global_varprior_losses = [], []
    for posterior_dist, prior_kl_dist in zip(posterior_dist_list, prior_kl_dist_list):
        global_variational_prior_loss, avg_global_varprior_loss = loss_params.kldiv_loss(
            p=posterior_dist,
            q=prior_kl_dist,
            global_batch_size=global_batch_size
        )
        global_variational_prior_losses.append(global_variational_prior_loss)
        avg_global_varprior_losses.append(avg_global_varprior_loss)

    global_variational_prior_losses = torch.stack(global_variational_prior_losses, dim=0)

    if loss_params.use_gamma_schedule:
        gamma_schedule = GammaSchedule(max_steps=loss_params.gamma_max_steps)
        global_variational_prior_loss = gamma_schedule(global_variational_prior_losses,
                                                       avg_global_varprior_losses,
                                                       step_n=step_n)
    else:
        global_variational_prior_loss = torch.sum(global_variational_prior_losses)

    global_var_loss = kldiv_schedule(step_n) * global_variational_prior_loss  # beta

    total_generator_loss = feature_matching_loss + global_var_loss

    scalar = np.log(2.)

    # True bits/dim kl div
    kl_div = torch.sum(global_variational_prior_losses) / scalar

    return avg_feature_matching_loss, avg_global_varprior_losses, total_generator_loss, means, log_scales, kl_div


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


def eval_step(model, inputs, step_n):
    with torch.no_grad():
        predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)

        avg_feature_matching_loss, avg_global_varprior_losses, total_generator_loss, means, \
            log_scales, kl_div = compute_loss(inputs,
                                              predictions,
                                              posterior_dist_list=posterior_dist_list,
                                              prior_kl_dist_list=prior_kl_dist_list,
                                              step_n=step_n,
                                              global_batch_size=eval_params.batch_size // train_params.num_gpus)

        outputs = model.module.top_down.sample(predictions)

    return outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, means, log_scales


def reconstruction_step(model, inputs, variates_masks=None, mode='recon'):
    model.eval()
    with torch.no_grad():
        predictions, posterior_dist_list, prior_kl_dist_list = model(inputs, variates_masks)

        if mode == 'recon':
            feature_matching_loss, global_varprior_losses, total_generator_loss, means, \
                log_scales, kl_div = compute_loss(inputs,
                                                  predictions,
                                                  posterior_dist_list=posterior_dist_list,
                                                  prior_kl_dist_list=prior_kl_dist_list,
                                                  step_n=max(loss_params.vae_beta_anneal_steps,
                                                             loss_params.gamma_max_steps) * 10.,
                                                  # any number bigger than schedule is fine
                                                  global_batch_size=inference_params.batch_size)

            outputs = model.top_down.sample(predictions)

            return outputs, feature_matching_loss, kl_div
        elif mode == 'encode':
            return posterior_dist_list
        else:
            raise ValueError(f'Unknown Mode {mode}')


def generation_step(model, temperatures):
    y, prior_zs = model.top_down.sample_from_prior(inference_params.batch_size, temperatures=temperatures)
    outputs = model.top_down.sample(y)
    return outputs, prior_zs


def encode_step(model, inputs):
    predictions, attn_weights_post_list, attn_weights_prior_kl_list, _, _, _, _, _ = model(inputs)
    return attn_weights_prior_kl_list


def update_ema(vae, ema_vae, ema_rate):
    for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def _compiled_train_step(model, inputs, step_n):
    predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)
    avg_feature_matching_loss, avg_global_varprior_losses, total_generator_loss, means, \
        log_scales, kl_div = compute_loss(inputs,
                                          predictions,
                                          posterior_dist_list=posterior_dist_list,
                                          prior_kl_dist_list=prior_kl_dist_list,
                                          step_n=step_n,
                                          global_batch_size=train_params.batch_size // train_params.num_gpus)

    total_generator_loss.backward()

    total_norm = gradient_clip(model)
    skip, gradient_skip_counter_delta = gradient_skip(total_norm)

    outputs = model.module.top_down.sample(predictions)

    return outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, total_norm, \
        gradient_skip_counter_delta, skip


def train_step(model, ema_model, optimizer, inputs, step_n):
    outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, global_norm, \
        gradient_skip_counter_delta, skip = _compiled_train_step(model, inputs, step_n)

    if not skip:
        optimizer.step()
        update_ema(model, ema_model, train_params.ema_decay)

    optimizer.zero_grad()
    return outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, \
        global_norm, gradient_skip_counter_delta


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
            train_outputs, train_feature_matching_loss, train_global_varprior_losses, train_kl_div, \
                global_norm, gradient_skip_counter_delta = train_step(model, ema_model, optimizer, train_inputs,
                                                                      global_step)
            # torch.cuda.synchronize()
            end_time = round((time.time() - start_time), 2)
            schedule.step()

            gradient_skip_counter += gradient_skip_counter_delta

            train_var_loss = np.sum([v.detach().cpu() for v in train_global_varprior_losses])
            train_nelbo = train_kl_div + train_feature_matching_loss
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
                    val_outputs, val_feature_matching_loss, \
                        val_global_varprior_loss, val_kl_div, val_means, val_log_scales = eval_step(model,
                                                                                                    inputs=val_inputs,
                                                                                                    step_n=global_step)

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
