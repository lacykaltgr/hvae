from hparams import *
from src.utils import create_checkpoint_manager_and_load_if_exists, create_tb_writer
from src.elements.optimizers import get_optimizer
from src.elements.schedules import get_schedule
import os

local_rank = int(os.environ["LOCAL_RANK"])

from models.TDVAE import hvae_model


def main():
    model = hvae_model
    with torch.no_grad():
        _ = model(torch.ones((1, data_params.channels, data_params.target_res, data_params.target_res)))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f}m.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))

    checkpoint, checkpoint_path = create_checkpoint_manager_and_load_if_exists(rank=local_rank)

    optimizer = get_optimizer(model=model,
                              type=optimizer_params.type,
                              learning_rate=optimizer_params.learning_rate,
                              beta_1=optimizer_params.beta1,
                              beta_2=optimizer_params.beta2,
                              epsilon=optimizer_params.epsilon,
                              weight_decay_rate=0.,
                              checkpoint=checkpoint)
    schedule = get_schedule(optimizer=optimizer,
                            decay_scheme=optimizer_params.learning_rate_scheme,
                            warmup_steps=optimizer_params.warmup_steps,
                            decay_steps=optimizer_params.decay_steps,
                            decay_rate=optimizer_params.decay_rate,
                            decay_start=optimizer_params.decay_start,
                            min_lr=optimizer_params.min_learning_rate,
                            last_epoch=torch.tensor(checkpoint['global_step']),
                            checkpoint=checkpoint)

    if checkpoint['model_state_dict'] is not None:
        if train_params.resume_from_ema:
            print('Resuming from EMA model')
            model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            print('Loaded Model Checkpoint')
            model.load_state_dict(checkpoint['model_state_dict'])

    if checkpoint['ema_model_state_dict'] is not None:
        model.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        print('EMA Loaded from checkpoint')
    else:
        model.ema_model.load_state_dict(model.state_dict())
        print('Copy EMA from model')

    model = model.to(model.device)
    if model.ema_model is not None:
        model.ema_model = model.ema_model.to(model.device)
        model.ema_model.requires_grad_(False)

    """
        if hparams.data.dataset_source in ['ffhq', 'celebAHQ', 'celebA', 'custom']:
        train_files, train_filenames = create_filenames_list(hparams.data.train_data_path)
        val_files, val_filenames = create_filenames_list(hparams.data.val_data_path)
        train_loader, val_loader = train_val_data_generic(train_files, train_filenames, val_files, val_filenames,
                                                          hparams.run.num_gpus, local_rank)
    elif hparams.data.dataset_source == 'cifar-10':
        train_loader, val_loader = train_val_data_cifar10(hparams.run.num_gpus, local_rank)
    elif hparams.data.dataset_source == 'binarized_mnist':
        train_loader, val_loader = train_val_data_mnist(hparams.run.num_gpus, local_rank)
    elif hparams.data.dataset_source == 'imagenet':
        train_loader, val_loader = train_val_data_imagenet(hparams.run.num_gpus, local_rank)
    else:
        raise ValueError(f'Dataset {hparams.data.dataset_source} is not included.')
    """

    # Book Keeping
    writer_train, logdir = create_tb_writer(mode='train')
    writer_val, _ = create_tb_writer(mode='val')

    # Train model
    model.train_model(optimizer, schedule, train_loader, val_loader, checkpoint['global_step'],
                      writer_train, writer_val, checkpoint_path, local_rank)


if __name__ == '__main__':
    main()
