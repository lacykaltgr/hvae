import numpy as np
import torch

from hparams import get_hparams
from src.utils import load_experiment_for, setup_logger
from src.model import evaluate


def main():
    p = get_hparams()
    checkpoint, checkpoint_path = load_experiment_for('test')
    logger = setup_logger(checkpoint_path)

    model = p.model_params.model()
    with torch.no_grad():
        _ = model(torch.ones((1, p.data_params.channels, p.data_params.target_res, p.data_params.target_res)))

    assert checkpoint['model_state_dict'] is not None
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Model Checkpoint is loaded')

    model = model.to(model.device)

    dataset = p.data_params.dataset
    val_loader = dataset.get_val_loader()

    evaluate(model, val_loader)


if __name__ == '__main__':
    main()
