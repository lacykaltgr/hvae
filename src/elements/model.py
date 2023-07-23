from torch import nn
from hparams import Hyperparams


def get_model(model):
    if model is None:
        return None
    elif isinstance(model, str):
        # Load model from default
        pass
    elif isinstance(model, Hyperparams):
        # Load model from hyperparameter config
        pass
    elif isinstance(model, nn.Module):
        # Load model from nn.Module
        return model
    elif isinstance(model, dict):
        # Load model from dictionary
        pass
    elif isinstance(model, list):
        # Load model from list
        pass
    else:
        raise NotImplementedError("Model type not supported.")


class MLP(nn.Module):
    def __init__(self, hparams):
        super(MLP, self).__init__()


class CNN(nn.Module):
    def __init__(self, hparams):
        super(CNN, self).__init__()


class ResNet(nn.Module):
    def __init__(self, hparams):
        super(ResNet, self).__init__()

