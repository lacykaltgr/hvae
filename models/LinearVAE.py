import data
from src.block import InputBlock, OutputBlock, TopBlock
from src.hvae import hVAE as hvae


def _model():

    _blocks = dict(
        x=InputBlock(
            net=None,
        ),
        z=TopBlock(
            net=x_to_z_net,
            prior_trainable=True,
            condition="x",
        ),
        x_hat=OutputBlock(
            net=z_to_x_net,
            input_id="z"
        ),
    )

    __model = hvae(
        blocks=_blocks,
        device=model_params.device
    )

    return __model


# --------------------------------------------------
# HYPERPAEAMETERS
# --------------------------------------------------
from hparams import Hyperparams

"""
--------------------
MODEL HYPERPARAMETERS
--------------------
"""

model_params = Hyperparams(
    model=_model,
    device='cuda',

    # run.name: Mandatory argument, used to identify runs for save and restore
    name='LinearVAE',
    # run.seed: seed that fixes all randomness in the project
    seed=420,

    # Whether to initialize the prior latent layer as zeros (no effect)
    initialize_prior_weights_as_zero=False,

    # Latent layer distribution base can be in ('std', 'logstd').
    # Determines if the model should predict
    # std (with softplus) or logstd (std is computed with exp(logstd)).
    distribution_base='std',
    # Similarly for output layer
    output_distribution_base='std',
    num_output_mixtures=10,

    # Latent layer Gradient smoothing beta. ln(2) ~= 0.6931472.
    # Setting this parameter to 1. disables gradient smoothing (not recommended)
    gradient_smoothing_beta=0.6931472,
    # Similarly for output layer
    output_gradient_smoothing_beta=0.6931472,
)

"""
--------------------
DATA HYPERPARAMETERS
--------------------
"""
data_params = Hyperparams(
    # Dataset source.
    # Can be one of ('mnist', 'cifar', 'imagenet', 'textures')
    dataset=data.cifar10.CIFARDataset(),

    # Data paths. Not used for (mnist, cifar-10)
    train_data_path='../datasets/imagenet_32/train_data/',
    val_data_path='../datasets/imagenet_32/val_data/',
    synthesis_data_path='../datasets/imagenet_32/val_data/',

    # Image metadata
    # Image resolution of the dataset (High and Width, assumed square)
    target_res=32,
    # Image channels of the dataset (Number of color channels)
    channels=3,
    # Image color depth in the dataset (bit-depth of each color channel)
    num_bits=8.
)

"""
--------------------
TRAINING HYPERPARAMETERS
--------------------
"""
train_params = Hyperparams(
    # The total number of training updates
    total_train_steps=800000,
    # training batch size
    batch_size=32,

    # Exponential Moving Average
    ema_decay=0.9999,
    # Whether to resume the model training from its EMA weights
    # (highly experimental, not recommended)
    resume_from_ema=False,

    # Defines how often to save a model checkpoint and logs (tensorboard) to disk.
    checkpoint_and_eval_interval_in_steps=10000,
)

"""
--------------------
EVALUATION HYPERPARAMETERS
--------------------
"""
eval_params = Hyperparams(
    # Defines how many validation samples to validate on every time we're going to write to tensorboard
    # Reduce this number of faster validation. Very small subsets can be non descriptive of the overall distribution
    #TODO: implement
    n_samples_for_validation=5000,
    # validation batch size
    batch_size=32 * 2,

    # Threshold used to mark latent groups as "active".
    # Purely for debugging, shouldn't be taken seriously.
    latent_active_threshold=1e-4
)

"""
--------------------
SYNTHESIS HYPERPARAMETERS
--------------------
"""
synthesis_params = Hyperparams(
    # The synthesis mode can be one of ('reconstruction', 'generation', 'div_stats', 'encoding')
    synthesis_mode='reconstruction',

    # Whether to use the EMA weights for inference
    load_ema_weights=True,
    # reconstruction/encoding mode Defines the quantile at which to prune the latent space (section 7). Example:
    # variate_masks_quantile = 0.03 means only 3% of the posteriors that encode the most information will be
    # preserved, all the others will be replaced with the prior. Encoding mode will always automatically prune the
    # latent space using this argument, so it's a good idea to run masked reconstruction (read below) to find a
    # suitable value of variate_masks_quantile before running encoding mode.
    variate_masks_quantile=0.03,

    # Reconstruction mode
    # Whether to save the targets during reconstruction (for debugging)
    save_target_in_reconstruction=False,
    # Whether to prune the posteriors to variate_masks_quantile. If set to True, the reconstruction is run with only
    # variate_masks_quantile posteriors. All the other variates will be replaced with the prior. Used to compute the
    # NLL at different % of prune posteriors, and to determine an appropriate variate_masks_quantile that doesn't
    # hurt NLL.
    mask_reconstruction=False,

    # div_stats mode Defines the ratio of the training data to compute the average KL per variate on (used for masked
    # reconstruction and encoding). Set to 1. to use the full training dataset. But that' usually an overkill as 5%,
    # 10% or 20% of the dataset tends to be representative enough.
    div_stats_subset_ratio=0.2,

    # generation_mode
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
    output_temperature=1.,

    # inference batch size (all modes)
    # The inference batch size is global for all GPUs for JAX only. Pytorch does not support multi-GPU inference.
    batch_size=32
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
    # Learning rate decay scheme
    # can be one of ('cosine', 'constant', 'exponential', 'noam')
    learning_rate_scheme='cosine',


    # Defines the initial learning rate value
    learning_rate=1e-3,
    # Adam/Radam/Adamax parameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    # L2 weight decay
    l2_weight=1e-2,

    # noam/constant/cosine warmup (not much of an effect, done in VDVAE)
    warmup_steps=100.,
    # exponential or cosine
    # Defines the number of steps over which the learning rate
    # decays to the minimim value (after decay_start)
    decay_steps=750000,
    # Defines the update at which the learning rate starts to decay
    decay_start=50000,
    # Defines the minimum learning rate value
    min_learning_rate=2e-4,
    # exponential only
    # Defines the decay rate of the exponential learning rate decay
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
    use_gamma_schedule=False,
    gamma_max_steps=10000,
    scaled_gamma=True,
    gamma_n_groups=100,

    # Defines the minimum logscale of the MoL layer (exp(-250 = 0) so it's disabled).
    # Look at section 6 of the Efficient-VDVAE paper.
    min_mol_logscale=-250.,
)

"""
--------------------
BLOCK HYPERPARAMETERS
--------------------
"""
# These are the default parameters,
# use this for reference when creating custom blocks.

mlp_params = Hyperparams(
    type='mlp',
    input_size=1000,
    hidden_sizes=[],
    output_size=1000,
    activation=torch.nn.ReLU(),
    residual=False
)

cnn_params = Hyperparams(
    type="conv",
    in_filters=3,
    bottleneck_ratio=0.5,
    output_ratio=1.,
    kernel_size=3,
    use_1x1=True,
    init_scaler=1.,
    pool=False,
    unpool=False,
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
x_to_z_net = Hyperparams(
    type='mlp',
    input_size=784,
    hidden_sizes=[2000],
    output_size=1000,
    activation=torch.nn.ReLU(),
    residual=False
)

z_to_x_net = Hyperparams(
    type='mlp',
    input_size=784,
    hidden_sizes=[2000],
    output_size=1000,
    activation=torch.nn.ReLU(),
    residual=False
)
