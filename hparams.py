
# SET WHICH MODEL TO USE HERE
# |    |    |    |    |    |
# v    v    v    v    v    v
import models.TDVAE as model

optimizer_params = model.optimizer_params
loss_params = model.loss_params
model_params = model.model_params
data_params = model.data_params
train_params = model.train_params
eval_params = model.eval_params
synthesis_params = model.synthesis_params

mlp_params = model.mlp_params
conv_params = model.conv_params
pool_params = model.pool_params
unpool_params = model.unpool_params
# ^    ^    ^    ^    ^    ^


class Hyperparams:
    def __init__(self, **config):
        self.config = config

    def __getattr__(self, name):
        return self.config[name]

    def __setattr__(self, name, value):
        self.config[name] = value

