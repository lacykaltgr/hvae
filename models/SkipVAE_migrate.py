from collections import OrderedDict


def _model(migration):
    from src.hvae.block import GenBlock, InputBlock, OutputBlock, SimpleBlock
    from src.hvae.hvae import hVAE as hvae
    from src.elements.layers import FixedStdDev, Flatten, Unflatten

    _blocks = OrderedDict(
        x=InputBlock(
            net=Flatten(start_dim=1),  #0: batch-flatten, 1: sample-flatten
        ),
        hiddens=SimpleBlock(
            net=migration.get_net("mlp_shared_encoder", activate_output=True),
            input_id="x"
        ),
        y=GenBlock(
            input_id="y_prior",
            condition="hiddens",
            prior_net=None,
            posterior_net=migration.get_net("mlp_cluster_encoder", activate_output=False),
            output_distribution="laplace"
        ),
        z=GenBlock(
            prior_net=migration.get_net("mlp_latent_decoder", activate_output=False),
            posterior_net=migration.get_net("mlp_latent_encoder_concat_to_z", activate_output=False),
            input_id="y",
            condition=[("hiddens",
                        ["y",  migration.get_net("mlp_latent_encoder_y_to_concat", activate_output=True)]),
                       "concat"],
            output_distribution="normal",
        ),
        y_concat_z=SimpleBlock(
            net=migration.get_net("skip_mlp_data_decoder", activate_output=False),
            input_id="y",
        ),
        x_hat=OutputBlock(
            net=[migration.get_net("concat_mlp_data_decoder", activate_output=False),
                 Unflatten(1, data_params.shape),
                 FixedStdDev(0.4)],
            input_id=[("z", "y_concat_z"), "concat"],
            output_distribution="normal"
        ),
    )

    _prior=OrderedDict(
        y_prior=torch.cat((torch.zeros(1, 250), torch.ones(1, 250)), dim=1)
    )

    __model = hvae(
        blocks=_blocks,
        init=_prior
    )

    return __model


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
    name='SkipVAE_migrate',

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
    custom_loss=chainVAE_loss,

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




