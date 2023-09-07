import torch
from src.hparams import get_hparams
from src.elements.optimizers import get_optimizer
from src.elements.schedules import get_schedule
from src.utils import load_experiment_for
from src.checkpoint import Checkpoint


# Migration Agent
from migration.ChainVAE_migration.migration_agent import ChainVAEMigrationAgent as MIGRATION_AGENT


def main():
    _, save_path = load_experiment_for('migration')
    p = get_hparams()

    migration = MIGRATION_AGENT(
        path="migration/ChainVAE_migration/weights/TD_comparison_40",
    )

    model = p.model_params.model(migration)
    global_step = migration.get_global_step()

    model.summary()

    optimizer = get_optimizer(model=model,
                              type=p.optimizer_params.type,
                              learning_rate=p.optimizer_params.learning_rate,
                              beta_1=p.optimizer_params.beta1,
                              beta_2=p.optimizer_params.beta2,
                              epsilon=p.optimizer_params.epsilon,
                              weight_decay_rate=p.optimizer_params.l2_weight,
                              checkpoint=migration.get_optimizer())
    schedule = get_schedule(optimizer=optimizer,
                            decay_scheme=p.optimizer_params.learning_rate_scheme,
                            warmup_steps=p.optimizer_params.warmup_steps,
                            decay_steps=p.optimizer_params.decay_steps,
                            decay_rate=p.optimizer_params.decay_rate,
                            decay_start=p.optimizer_params.decay_start,
                            min_lr=p.optimizer_params.min_learning_rate,
                            last_epoch=torch.tensor(-1),
                            checkpoint=migration.get_schedule())

    checkpoint = Checkpoint(
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=schedule
    )
    checkpoint.save_migration(save_path)


if __name__ == '__main__':
    main()

