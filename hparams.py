# This file contains all the hyperparameters for the model.
class Hyperparams:
    def __init__(self, **config):
        self.config = config

    def __getattr__(self, name):
        return self.config[name]

    def __setattr__(self, name, value):
        self.config[name] = value



"""
--------------------
RUN HYPERPARAMETERS
--------------------
"""
run_params = Hyperparams(
    seed=420,
    device='cuda',
)


"""
--------------------
DATA HYPERPARAMETERS
--------------------
"""
data_params = Hyperparams(
    dataset="mnist",
)


"""
--------------------
TRAINING HYPERPARAMETERS
--------------------
"""
train_params = Hyperparams(
    name="train",
    seed=42,
    num_epochs=100,
    batch_size=64,
    num_workers=4,
    num_gpus=1,
)

"""
--------------------
EVALUATION HYPERPARAMETERS
--------------------
"""
eval_params = Hyperparams(
    latent_active_threshold=1e-4,
    name="train",
    seed=42,
    num_epochs=100,
    batch_size=64,
    num_workers=4
)

"""
--------------------
SYNTHESIS HYPERPARAMETERS
--------------------
"""
synthesis_params = Hyperparams(
    name="train",
    seed=42,
    num_epochs=100,
    batch_size=64,
    num_workers=4
)

"""
--------------------
OPTIMIZER HYPERPARAMETERS
--------------------
"""
optimizer_params = Hyperparams(
    # Optimizer can be one of ('Radam', 'Adam', 'Adamax').
    # Adam and Radam should be avoided on datasets when the global norm is large!!
    type='Adamax',
    # Learning rate decay scheme can be one of ('cosine', 'constant', 'exponential', 'noam')
    learning_rate_scheme='cosine',
    # Defines the initial learning rate value
    learning_rate=1e-3,

    # noam/constant/cosine warmup (not much of an effect, done in VDVAE)
    warmup_steps=100.,

    # exponential or cosine
    # Defines the number of steps over which the learning rate decays to the minimim value (after decay_start)
    decay_steps=750000,
    # Defines the update at which the learning rate starts to decay
    decay_start=50000,
    # Defines the minimum learning rate value
    min_learning_rate=2e-4,

    # exponential only
    # Defines the decay rate of the exponential learning rate decay
    decay_rate=0.5,

    # Adam/Radam/Adamax parameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,

    # Gradient
    # Gradient clip_norm value should be defined for nats/dim loss.
    clip_gradient_norm=False,
    gradient_clip_norm_value=300.,

    # Whether or not to use gradient skipping. This is usually unnecessary when using gradient smoothing but it
    # doesn't hurt to keep it for safety.
    gradient_skip=True,
    # Defines the threshold at which to skip the update. Also defined for nats/dim loss.
    gradient_skip_threshold=800.
)

"""
--------------------
LOSS HYPERPARAMETERS
--------------------
"""
from src.elements.losses import *

loss_params = Hyperparams(
    reconstruction_loss=DiscMixLogistic(),
    kldiv_loss=KLDivergence(),

    # Defines the minimum logscale of the MoL layer (exp(-250 = 0) so it's disabled). Look at section 6 of the paper.
    min_mol_logscale=-250.,

    # ELBO beta warmup (from NVAE). Doesn't make much of an effect
    # but it's safe to use it to avoid posterior collapses as NVAE suggests.
    # lambda of variational prior loss
    # schedule can be in ('None', 'Logistic', 'Linear')
    variation_schedule='Linear',

    # linear beta schedule
    vae_beta_anneal_start=21,
    vae_beta_anneal_steps=5000,
    vae_beta_min=1e-4,

    # logistic beta schedule
    vae_beta_activation_steps=10000,
    vae_beta_growth_rate=1e-5,

    # Balancing the KL divergence between levels of the hierarchy using a gamma balancing (from NVAE).
    # Doesn't make much of an effect but it's safe to use it to avoid posterior collapses as NVAE suggests.
    # It's usually safe to set it to only work for as many steps as the beta anneal or double that.
    # Gamma schedule of variational loss
    use_gamma_schedule=True,
    gamma_max_steps=10000,
    scaled_gamma=True,

    # L2 weight decay
    use_weight_decay=True,
    l2_weight=1e-2
)

"""
--------------------
BLOCK HYPERPARAMETERS
--------------------
"""

mlp_params = Hyperparams(
    name="mlp_z2",
    z_dim=2,
    z2_dim=2,
    z2_activation="relu",
    z2_num_layers=2,
    z2_hidden_dim=512,
    z2_dropout=0.0
)

cnn_params = Hyperparams(
    name="mlp_z2",
    z_dim=2,
    z2_dim=2,
    z2_activation="relu",
    z2_num_layers=2,
    z2_hidden_dim=512,
    z2_dropout=0.0
)

resnet_params = Hyperparams(
    name="mlp_z2",
    z_dim=2,
    z2_dim=2,
    z2_activation="relu",
    z2_num_layers=2,
    z2_hidden_dim=512,
    z2_dropout=0.0
)

"""
--------------------
CUSTOM BLOCK HYPERPARAMETERS
--------------------
"""
# add your custom block hyperparameters here
