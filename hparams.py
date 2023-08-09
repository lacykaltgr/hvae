def get_hparams():
    # SET WHICH params TO USE HERE
    # |    |    |    |    |    |
    # v    v    v    v    v    v
    import models.TDVAE as params

    return Hyperparams(
        log_params=params.log_params,
        model_params=params.model_params,
        data_params=params.data_params,
        train_params=params.train_params,
        optimizer_params=params.optimizer_params,
        loss_params=params.loss_params,
        eval_params=params.eval_params,
        synthesis_params=params.synthesis_params,

        mlp_params=params.mlp_params,
        conv_params=params.cnn_params,
        pool_params=params.pool_params,
        unpool_params=params.unpool_params,
    )


class Hyperparams:
    def __init__(self, **config):
        self.config = config

    def __getattr__(self, name):
        if name == 'config':
            return super().__getattribute__(name)
        return self.config[name]

    def __setattr__(self, name, value):
        if name == 'config':
            super().__setattr__(name, value)
        else:
            self.config[name] = value

    def __getstate__(self):
        return self.config

    def __setstate__(self, state):
        self.config = state

    def keys(self):
        return self.config.keys()

