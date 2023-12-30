import torch

from src.hparams import get_hparams
from src.utils import load_experiment_for
from src.hvae.model import evaluate


def main():
    p = get_hparams()
    checkpoint, checkpoint_path = load_experiment_for('test')
    device = p.model_params.device

    assert checkpoint is not None
    model = checkpoint.get_model()
    print(f'Model Checkpoint is loaded from {p.log_params.load_from_eval}')

    with torch.no_grad():
        _ = model(torch.ones((1, *p.data_params.shape)))
    model = model.to(device)

    dataset = p.data_params.dataset(**p.data_params.params)
    test_loader = dataset.get_test_loader(p.eval_params.batch_size)

    evaluate(model, test_loader,
             n_samples=p.eval_params.n_samples_for_validation,
             use_mean=p.eval_params.use_mean,
             global_step=None,
             logger=None)


if __name__ == '__main__':
    main()
