import torch
from hparams import get_hparams
from src.elements.optimizers import get_optimizer
from src.elements.schedules import get_schedule
from src.utils import load_experiment_for
from checkpoint import Checkpoint


# Migration Agent
from migration.EfficientVDVAE_migration.migration_agent import EfficientVDVAEMigrationAgent as MIGRATION_AGENT


def main():
    _, save_path = load_experiment_for('migration')
    p = get_hparams()

    migration = MIGRATION_AGENT(
        path="migration/EfficientVDVAE_migration/weights_imagenet/",
        weights_filename="checkpoints-imagenet32_baseline",
        config_filename="hparams-imagenet32_baseline"
    )
    model = p.model_params.model(migration)
    global_step = migration.get_global_step()

    #with torch.no_grad():
    #    _ = model(torch.ones((1, 1, 40, 40)))
    #print(model.summary())

    optimizer = get_optimizer(model=model,
                              type=p.optimizer_params.type,
                              learning_rate=p.optimizer_params.learning_rate,
                              beta_1=0.99,
                              beta_2=0.9,
                              epsilon=p.optimizer_params.epsilon,
                              weight_decay_rate=p.optimizer_params.l2_weight,
                              checkpoint=None)
    schedule = get_schedule(optimizer=optimizer,
                            decay_scheme=p.optimizer_params.learning_rate_scheme,
                            warmup_steps=p.optimizer_params.warmup_steps,
                            decay_steps=p.optimizer_params.decay_steps,
                            decay_rate=p.optimizer_params.decay_rate,
                            decay_start=p.optimizer_params.decay_start,
                            min_lr=p.optimizer_params.min_learning_rate,
                            last_epoch=torch.tensor(-1),
                            checkpoint=None)

    optimizer = migration.get_optimizer(optimizer)
    schedule = migration.get_schedule(schedule)

    checkpoint = Checkpoint(
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=schedule
    )
    checkpoint.save(save_path)


if __name__ == '__main__':
    main()

