import torch
import os

from src.hparams import get_hparams
from src.utils import load_experiment_for, wandb_init
from src.hvae.model import reconstruct


def main():
    p = get_hparams()
    wandb = wandb_init(name=p.log_params.name, config=p.to_json())
    checkpoint, checkpoint_path = load_experiment_for('test', wandb)
    device = p.model_params.device

    assert checkpoint is not None
    model = checkpoint.get_model()
    print(f'Model Checkpoint is loaded from {p.log_params.load_from_eval}')

    with torch.no_grad():
        _ = model(torch.ones((1, *p.data_params.shape)))
    model = model.to(device)

    dataset = p.data_params.dataset(**p.data_params.params)
    test_loader = dataset.get_test_loader(p.eval_params.batch_size)

    reconstruct(
        net=model,
        dataset=test_loader,
        wandb_run=wandb,
        use_mean=p.eval_params.use_mean
    )


if __name__ == '__main__':
    main()
