import numpy as np
import torch

from hparams import get_hparams
from src.elements.optimizers import get_optimizer
from src.elements.schedules import get_schedule
from src.utils import load_experiment_for, create_tb_writer_for, setup_logger
from src.model import train


def main():
    p = get_hparams()
    checkpoint, checkpoint_path = load_experiment_for('train')
    logger = setup_logger(checkpoint_path)

    model = p.model_params.model()
    gloabal_step = checkpoint.global_step if checkpoint is not None else -1

    optimizer = get_optimizer(model=model,
                              type=p.optimizer_params.type,
                              learning_rate=p.optimizer_params.learning_rate,
                              beta_1=p.optimizer_params.beta1,
                              beta_2=p.optimizer_params.beta2,
                              epsilon=p.optimizer_params.epsilon,
                              weight_decay_rate=p.optimizer_params.l2_weight,
                              checkpoint=checkpoint)
    schedule = get_schedule(optimizer=optimizer,
                            decay_scheme=p.optimizer_params.learning_rate_scheme,
                            warmup_steps=p.optimizer_params.warmup_steps,
                            decay_steps=p.optimizer_params.decay_steps,
                            decay_rate=p.optimizer_params.decay_rate,
                            decay_start=p.optimizer_params.decay_start,
                            min_lr=p.optimizer_params.min_learning_rate,
                            last_epoch=torch.tensor(gloabal_step),
                            checkpoint=checkpoint)

    with torch.no_grad():
        _ = model(torch.ones((1, p.data_params.channels, p.data_params.target_res, p.data_params.target_res)))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    logger.info('Train step generator trainable params {:.3f}m.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Loaded Model Checkpoint')
    model = model.to(model.device)

    dataset = p.data_params.dataset
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()

    writer_train = create_tb_writer_for('train', checkpoint_path=checkpoint_path)
    writer_val = create_tb_writer_for('val', checkpoint_path=checkpoint_path)

    train(model, optimizer, schedule, train_loader, val_loader, gloabal_step,
          writer_train, writer_val, checkpoint_path, logger)


if __name__ == '__main__':
    main()
