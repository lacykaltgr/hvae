import torch
from hvae_backbone.utils import SerializableModule


def _model():
    from hvae_backbone.block import GenBlock, InputBlock, OutputBlock, SimpleBlock
    from hvae_backbone.sequence import hSequenceVAE
    from hvae_backbone.elements.layers import Conv2d, Slice, FixedStdDev
    from hvae_backbone.utils import SharedSerializableSequential as Shared
    from hvae_backbone.utils import OrderedModuleDict

    shared_net = Shared(Conv2d(40, 4, 3, 1, 'SAME'))

    _blocks = OrderedModuleDict(
        x=InputBlock(),
            sparse_mu_sigma=SimpleBlock(
                net=x_to_sparse,
                input_id="x",
            ),
            h_manifold=SimpleBlock(
                net=[Slice(40), shared_net],
                input_id="sparse_mu_sigma",
            ),
        h=GenBlock(
            prior_net=FixedStdDev(0.2),
            posterior_net=ProductOfExperts(),
            input_id="_h",
            condition=[("h_manifold", "_h_manifold"), "substract", FixedStdDev(0.4)],
            output_distribution="normal",
            fuse_prior="concat"
        ),
        z=GenBlock(
            prior_net=manifold_recon,
            posterior_net=z_posterior,
            input_id=[("_z_manifold", "h"), "add"],
            condition="sparse_mu_sigma",
            output_distribution="laplace",
            fuse_prior="concat",
        ),
            z_manifold=SimpleBlock(
                net=shared_net,
                input_id="z",
            ),
        x_hat=OutputBlock(
            net=[z_to_x, FixedStdDev(0.4)],
            input_id="z",
            output_distribution="normal"
        ),
    )

    _init = dict(
        _h_manifold=torch.zeros(size=(4, 10, 10)),
        _h=torch.zeros(size=(4, 10, 10)),
        _z=torch.zeros(size=(40, 10, 10)),
        _z_manifold=torch.zeros(size=(4, 10, 10)),
    )

    __model = hSequenceVAE(
        blocks=_blocks,
        init=_init,
    )

    return __model


class ProductOfExperts(SerializableModule):
    def __init__(self):
        super(ProductOfExperts, self).__init__()

    def forward(self, x):
        mu_0, sigma_0, mu_1, sigma_1 = x.chunk(4, dim=1)
        MU_num = mu_0 * sigma_1 ** -2 + mu_1 * sigma_0 ** -2
        MU_den = sigma_0 ** -2 + sigma_1 ** -2
        MU = MU_num / MU_den
        SIGMA = MU_den ** -1
        return torch.cat([MU, SIGMA], dim=1)


# --------------------------------------------------
# HYPERPAEAMETERS
# --------------------------------------------------
from hvae_backbone import Hyperparams

"""
--------------------
LOGGING HYPERPARAMETERS
--------------------
"""
log_params = Hyperparams(
    name='SMT-VAE',

    # TRAIN LOG
    # --------------------
    # Defines how often to save a model checkpoint and logs (tensorboard) to disk.
    checkpoint_interval_in_steps=150,
    eval_interval_in_steps=150,

    load_from_train=None,
    load_from_eval='path_to_directory/checkpoint.pth',
)

"""
--------------------
MODEL HYPERPARAMETERS
--------------------
"""

model_params = Hyperparams(
    model=_model,
    device='cpu',
    seed=420,

    # Latent layer distribution base can be in ('std', 'logstd').
    # Determines if the model should predict
    # std (with softplus) or logstd (std is computed with exp(logstd)).
    distribution_base='logstd',
    distribution_sigma_param="var",

    # Latent layer Gradient smoothing beta. ln(2) ~= 0.6931472.
    # Setting this parameter to 1. disables gradient smoothing (not recommended)
    gradient_smoothing_beta=0.6931472
)

"""
--------------------
DATA HYPERPARAMETERS
--------------------
"""
from data.forest_walk.forest_walk import ForestVideoDataset

data_params = Hyperparams(
    # Dataset source.
    # Can be one of ('mnist', 'cifar', 'imagenet', 'textures')
    dataset=ForestVideoDataset,
    params=dict(seq_len=5, n_frames=25600, frame_rate=2, patch_size=40, whiten=False, n_downsamples=2),

    # Image metadata
    shape=(5, 1, 40, 40),
)

"""
--------------------
TRAINING HYPERPARAMETERS
--------------------
"""
train_params = Hyperparams(
    # The total number of training updates
    total_train_steps=10_000,
    # training batch size
    batch_size=128,

    # Freeze spceific layers
    unfreeze_first=False,
    freeze_nets=[],
)

"""
--------------------
OPTIMIZER HYPERPARAMETERS
--------------------
"""
optimizer_params = Hyperparams(
    # Optimizer can be one of ('Radam', 'Adam', 'Adamax').
    # Adam and Radam should be avoided on datasets when the global norm is large!!
    type='Adam',
    # Learning rate decay scheme
    # can be one of ('cosine', 'constant', 'exponential', 'noam')
    learning_rate_scheme='constant',

    # Defines the initial learning rate value
    learning_rate=5e-4
    ,
    # Adam/Radam/Adamax parameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    # L2 weight decay
    l2_weight=0e-6,

    # noam/constant/cosine warmup (not much of an effect, done in VDVAE)
    warmup_steps=100.,
    # exponential or cosine
    #   Defines the number of steps over which the learning rate
    #   decays to the minimim value (after decay_start)
    decay_steps=750000,
    #   Defines the update at which the learning rate starts to decay
    decay_start=50000,
    #   Defines the minimum learning rate value
    min_learning_rate=2e-4,
    # exponential only
    #   Defines the decay rate of the exponential learning rate decay
    decay_rate=0.5,

    # Gradient
    #  clip_norm value should be defined for nats/dim loss.
    clip_gradient_norm=False,
    gradient_clip_norm_value=300.,

    # Whether to use gradient skipping.
    # This is usually unnecessary when using gradient smoothing but it
    # doesn't hurt to keep it for safety.
    gradient_skip=True,
    # Defines the threshold at which to skip the update.
    # Also defined for nats/dim loss.
    gradient_skip_threshold=1e10
)

"""
--------------------
LOSS HYPERPARAMETERS
--------------------
"""
loss_params = Hyperparams(
    reconstruction_loss="default",
    kldiv_loss="default",
    custom_loss=None,

    # ELBO beta warmup (from NVAE).
    # Doesn't make much of an effect
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
    use_gamma_schedule=False,
    gamma_max_steps=10000,
    scaled_gamma=True,
    gamma_n_groups=100,
)

"""
--------------------
EVALUATION HYPERPARAMETERS
--------------------
"""
eval_params = Hyperparams(
    # Defines how many validation samples to validate on every time we're going to write to tensorboard
    # Reduce this number of faster validation. Very small subsets can be non descriptive of the overall distribution
    n_samples_for_validation=5000,
    n_samples_for_reconstruction=3,

    # validation batch size
    batch_size=128,

    use_mean=False,
)

"""
--------------------
SYNTHESIS HYPERPARAMETERS
--------------------
"""
analysis_params = Hyperparams(
    # The synthesized mode can be a subset of
    # ('generation', 'decodability', 'white_noise_analysis', 'latent_step_analysis', 'mei')
    ops=['white_noise_analysis'],

    # inference batch size (all modes)
    batch_size=128,


    # White noise analysis mode
    # --------------------
    white_noise_analysis=dict(
        z=dict(
            n_samples=1000,
            sigma=0.1,
        )
    ),

    # Most Exciting Input (MEI) mode
    # --------------------
    mei=dict(
        operation_name=dict(
            # objective operation
            # return dict -> {'objective': ..., 'activation': ...}
            # or tensor -> activation
            objective=lambda computed: dict(
                objective=computed['x_hat'][0]
            ),
            # whether model should use mean or sample
            use_mean=False,

            # mei generation procedure
            # can either be 'pixel', 'distribution' or 'transform'
            type='pixel',

            # mei generation parameters
            config=dict()
        )
    ),


    # Decodability mode
    # --------------------

    decodability=dict(
        decode_from_block=dict(
            model=None,
            optimizer='Adam',
            loss="bce",
            epcohs=100,
            learning_rate=1e-3,
            batch_size=32,
        ),
    ),


    # Latent traversal mode
    # --------------------
    latent_step_analysis=dict(
        z=dict(
            diff=1,
            value=1,
        )
    ),

    # Generation_mode
    # --------------------
    generation=Hyperparams(
        # Number of generated batches per temperature from the temperature_settings list.
        n_generation_batches=1,
        # Temperatures for unconditional generation from the prior. We generate n_generation_batches for each element of
        # the temperature_settings list. This is implemented so that many temperatures can be tested in the same run for
        # speed. The temperature_settings elements can be one of: - A float: Example 0.8. Defines the temperature used
        # for all the latent variates of the model - A tuple of 3: Example ('linear', 0.6, 0.9). Defines a linearly
        # increasing temperature scheme from the deepest to shallowest top-down block. (different temperatures per latent
        # group) - A list of size len(down_strides): Each element of the list defines the temperature for their
        # respective top-down blocks.
        temperature_settings=[0.8, 0.85, 0.9, 1., ('linear', 0.7, 0.9), ('linear', 0.9, 0.7), ('linear', 0.8, 1.),
                              ('linear', 1., 0.8), ('linear', 0.8, 0.9)],
        # Temperature of the output layer (usually kept at 1. for extra realism)
        output_temperature=1.
    )
)


"""
--------------------
CUSTOM BLOCK HYPERPARAMETERS
--------------------
"""
# add your custom block hyperparameters here
x_to_sparse = Hyperparams(
    type="conv",
    in_filters=1,
    filters=[10, 40, 80],
    kernel_size=3,
    pools=[0, 1],
    unpool_strides=0,
    activation=torch.nn.Softplus(),
    activate_output=False
)

manifold_recon = Hyperparams(
    type="conv",
    in_filters=4,
    filters=[10, 40, 80],
    kernel_size=3,
    activation=torch.nn.Softplus(),
    activate_output=False
)

z_posterior = Hyperparams(
    type="conv",
    in_filters=160,
    filters=[160, 120, 80],
    kernel_size=3,
    activation=torch.nn.Softplus(),
    activate_output=False
)

z_to_x = Hyperparams(
    type="deconv",
    in_filters=40,
    filters=[40, 10, 4, 1],
    kernel_size=3,
    unpools=[0, 1],
    activation=torch.nn.Softplus(),
    activate_output=False
)
