import numpy as np
import torch

from hparams import get_hparams
from src.utils import load_experiment_for, setup_logger
from src.model import evaluate


def main():
    p = get_hparams()
    checkpoint, checkpoint_path = load_experiment_for('test')

    assert checkpoint is not None
    model = checkpoint.get_model()
    print('Model Checkpoint is loaded')

    with torch.no_grad():
        _ = model(torch.ones((1, *p.data_params.shape)))

    model = model.to(model.device)

    dataset = p.data_params.dataset
    val_loader = dataset.get_val_loader()

    evaluate(model, val_loader, global_step=None, logger=None)


if __name__ == '__main__':
    main()
