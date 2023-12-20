from collections import OrderedDict
from src.utils import SerializableModule


def _model():
    from src.hvae.block import GenBlock, InputBlock, OutputBlock, TopGenBlock, SimpleBlock
    from src.hvae.sequence import hSequenceVAE
    from src.elements.layers import Conv2d, Slice, FixedStdDev
    from src.utils import SharedSerializableSequential as Shared

    shared_net = Shared(Conv2d(40, 4, 3, 1, 1))

    _blocks = OrderedDict(
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
            fuse_prior="concat"
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

    _init = OrderedDict(
        _h_manifold=torch.zeros(size=(1, 4, 10, 10)),
        _h=torch.zeros(size=(1, 4, 10, 10)),
        _z=torch.zeros(size=(1, 40, 10, 10)),
        _z_manifold=torch.zeros(size=(1, 4, 10, 10)),
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
from src.hparams import Hyperparams

"""
--------------------
LOGGING HYPERPARAMETERS
--------------------
"""
log_params = Hyperparams(
    dir='experiments/',
    name='SMT-VAE',

    # TRAIN LOG
    # --------------------
    # Defines how often to save a model checkpoint and logs (tensorboard) to disk.
    checkpoint_interval_in_steps=150,
    eval_interval_in_steps=150,

    load_from_train=None,
    dir_naming_scheme='timestamp',

    # EVAL
    # --------------------
    load_from_eval='path_to_directory/checkpoint.pth',

    # SYNTHESIS
    # --------------------
    load_from_analysis='path_to_directory/checkpoint.pth',
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
    gradient_smoothing_beta=0.6931472,

    # Num of mixtures in the MoL layer
    num_output_mixtures=3,
    # Defines the minimum logscale of the MoL layer (exp(-250 = 0) so it's disabled).
    # Look at section 6 of the Efficient-VDVAE paper.
    min_mol_logscale=-250.,
)

"""
--------------------
DATA HYPERPARAMETERS
--------------------
"""
from data.textures.textures import TexturesDataset

data_params = Hyperparams(
    # Dataset source.
    # Can be one of ('mnist', 'cifar', 'imagenet', 'textures')
    dataset=TexturesDataset,
    params=dict(type="natural", image_size=40, whitening="old"),

    # Image metadata
    shape=(1, 40, 40),
    # Image color depth in the dataset (bit-depth of each color channel)
    num_bits=8.,
)

"""
--------------------
TRAINING HYPERPARAMETERS
--------------------
"""
train_params = Hyperparams(
    # The total number of training updates
    total_train_steps=640000,
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
    learning_rate=.05e-3
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
    variation_schedule='None',

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
    # validation batch size
    batch_size=128,

    use_mean=True,

    # Threshold used to mark latent groups as "active".
    # Purely for debugging, shouldn't be taken seriously.
    latent_active_threshold=1e-4
)

"""
--------------------
SYNTHESIS HYPERPARAMETERS
--------------------
"""
analysis_params = Hyperparams(
    # The synthesized mode can be a subset of
    # ('reconstruction', 'generation', div_stats', 'decodability', 'white_noise_analysis', 'latent_step_analysis')
    # in development: 'mei', 'gabor'
    ops=['reconstruction'],

    # inference batch size (all modes)
    batch_size=32,

    # Latent traversal mode
    # --------------------
    reconstruction=Hyperparams(
        n_samples_for_reconstruction=3,
        # The quantile at which to prune the latent space
        # Example:
        # variate_masks_quantile = 0.03 means only 3% of the posteriors that encode the most information will be
        # preserved, all the others will be replaced with the prior. Encoding mode will always automatically prune the
        # latent space using this argument, so it's a good idea to run masked reconstruction (read below) to find a
        # suitable value of variate_masks_quantile before running encoding mode.
        mask_reconstruction=False,
        variate_masks_quantile=0.03,
    ),

    # Latent traversal mode
    # --------------------
    latent_step_analysis=Hyperparams(
        queries=dict(
            z=dict(
                diff=1,
                value=1,
                n_dims=70,
                n_cols=10,
            )
        )
    ),

    # White noise analysis mode
    # --------------------
    white_noise_analysis=Hyperparams(
        queries=dict(
            z=dict(
                n_samples=1000,
                sigma=1.,
                n_cols=10,
            )
        )
    ),

    # Most Exciting Input (MEI) mode
    # --------------------
    mei=Hyperparams(
        queries=dict(
        )

    ),
    gabor=Hyperparams(
        queries=dict(
        )
    ),

    # Div_stats mode
    # --------------------
    div_stats=Hyperparams(
        # Defines the ratio of the training data to compute the average KL per variate on (used for masked
        # reconstruction and encoding). Set to 1. to use the full training dataset.
        # But that' usually an overkill as 5%, 10% or 20% of the dataset tends to be representative enough.
        div_stats_subset_ratio=0.2
    ),

    # Decodability mode
    # --------------------

    decodability=Hyperparams(
        model=None,
        optimizer='Adam',
        loss="bce",
        epcohs=100,
        learning_rate=1e-3,
        batch_size=32,
        decode_from=['z', 'y'],
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
BLOCK HYPERPARAMETERS
--------------------
"""
import torch

# These are the default parameters,
# use this for reference when creating custom blocks.

mlp_params = Hyperparams(
    type='mlp',
    input_size=1000,
    hidden_sizes=[],
    output_size=1000,
    activation=torch.nn.ReLU(),
    residual=False,
    activate_output=True
)

cnn_params = Hyperparams(
    type="conv",
    n_layers=2,
    in_filters=3,
    bottleneck_ratio=0.5,
    output_ratio=1.,
    kernel_size=3,
    use_1x1=True,
    init_scaler=1.,
    pool_strides=False,
    unpool_strides=False,
    activation=None,
    residual=False,
)

pool_params = Hyperparams(
    type='pool',
    in_filters=3,
    filters=3,
    strides=2,
)

unpool_params = Hyperparams(
    type='unpool',
    in_filters=3,
    filters=3,
    strides=2,
)

"""
--------------------
CUSTOM BLOCK HYPERPARAMETERS
--------------------
"""
# add your custom block hyperparameters here
x_to_sparse = Hyperparams(
    type="conv",
    n_layers=0,
    in_filters=1,
    bottleneck_ratio=40,
    output_ratio=40,
    kernel_size=3,
    use_1x1=False,
    init_scaler=1.,
    pool_strides=False,
    unpool_strides=False,
    activation=torch.nn.ReLU(),
    residual=False,
)

manifold_recon = Hyperparams(
    type="conv",
    n_layers=0,
    in_filters=4,
    bottleneck_ratio=40,
    output_ratio=40,
    kernel_size=3,
    use_1x1=False,
    init_scaler=1.,
    pool_strides=False,
    unpool_strides=False,
    activation=torch.nn.ReLU(),
    residual=False,
)

z_posterior = Hyperparams(
    type="conv",
    n_layers=0,
    in_filters=80,
    bottleneck_ratio=40,
    output_ratio=80,
    kernel_size=3,
    use_1x1=False,
    init_scaler=1.,
    pool_strides=False,
    unpool_strides=False,
    activation=torch.nn.ReLU(),
    residual=False,
)

z_to_x = Hyperparams(
    type="conv",
    n_layers=0,
    in_filters=40,
    bottleneck_ratio=20,
    output_ratio=1,
    kernel_size=3,
    use_1x1=False,
    init_scaler=1.,
    pool_strides=False,
    unpool_strides=False,
    activation=None,
    residual=False,
)
