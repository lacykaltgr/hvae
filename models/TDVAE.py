import torch.nn

from src.block import EncBlock, DecBlock, InputBlock, OutputBlock, TopBlock
from src.hvae import hVAE as hvae
import data


def _model():
    _device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")

    _blocks = dict(
        x=InputBlock(),
        hiddens=EncBlock(
            model=cnn_params,
            input="x", log_output=True),
        y=TopBlock(  # adds to KL loss
            model=mlp_params,
            input="hiddens"),
        z=DecBlock(mlp_params,
                   input="y",
                   condition="hiddens",
                   prior_net=z_prior_net_params,
                   posterior_net=z_posterior_net_params, ),
        x_hat=OutputBlock(),
    )

    __model = hvae(
        blocks=_blocks,
        name="hvae",
        device=_device
    )

    return __model


# --------------------------------------------------
# HYPERPAEAMETERS
# --------------------------------------------------
from hparams import Hyperparams

"""
--------------------
RUN HYPERPARAMETERS
--------------------
"""

run_params = Hyperparams(
    model=_model(),
    # Run section: Defines the most important randomness and hardware params
    device='cuda',
    # run.name: Mandatory argument, used to identify runs for save and restore
    name='cifar10_baseline',
    # run.seed: seed that fixes all randomness in the project
    seed=420,

    # Hardware
    # Global run config for GPUs and CPUs
    num_gpus=2,
    num_cpus=256,

    # JAX only: Defines how many checkpoints will be kept on disk (the latest N)
    max_allowed_checkpoints=5
)

"""
--------------------
DATA HYPERPARAMETERS
--------------------
"""
data_params = Hyperparams(
    # Data section: Defines the dataset parameters
    # To change a dataset to run the code on:
    #   - Change the data.dataset_source to reflect which dataset you're trying to run.
    #           This controls which data loading scripts to use and how to normalize
    #   - Change the paths. For all datasets but binarized_mnist and cifar-10, define where the data lives on disk.
    #   - Change the metadata: Define the image resolution, the number of channels and the color bit-depth of the data.

    # Dataset source. Can be one of ('binarized_mnist', 'cifar-10', 'imagenet', 'celebA', 'celebAHQ', 'ffhq')
    dataset_source='textures',

    # Data paths. Not used for (binarized_mnist, cifar-10)
    train_data_path='../datasets/imagenet_32/train_data/',
    val_data_path='../datasets/imagenet_32/val_data/',
    synthesis_data_path='../datasets/imagenet_32/val_data/',

    # Image metadata
    # Image resolution of the dataset (High and Width, assumed square)
    target_res=32,
    # Image channels of the dataset (Number of color channels)
    channels=3,
    # Image color depth in the dataset (bit-depth of each color channel)
    num_bits=8.,
    # Whether to do a random horizontal flip of images when loading the data (no applicable to MNIST)
    random_horizontal_flip=True,

    dataset_params={
        'batch_size': 128,
        'test_batch_size': 128,
        'train_every': 2,
        'test_every': 1,
        'crop_dim': 40,
        'path': '../datasets/fakelabeled_natural_commonfiltered_640000_40px.pkl',
        'offset': 0.0},
)

"""
--------------------
TRAINING HYPERPARAMETERS
--------------------
"""
train_params = Hyperparams(
    # Train section: Defines parameters only useful during training

    # The total number of training updates
    total_train_steps=800000,
    # training batch size (global for all devices)
    batch_size=32,

    # Exponential Moving Average
    ema_decay=0.9999,
    # Whether to resume the model training from its EMA weights (highly experimental, not recommended)
    resume_from_ema=False,

    # JAX only (controls how often to print the training metrics to terminal). Set to 1 for maximum verbosity, or higher for less frequent updates
    logging_interval_in_steps=1,

    # Defines how often to save a model checkpoint and logs (tensorboard) to disk.
    checkpoint_and_eval_interval_in_steps=10000,

    output_type='normal',
    output_sd=0.4,
    n_y=250,
    n_y_samples=1,
    n_y_samples_reconstr=1,
    beta_y_evo=beta_y,
    n_z=1800,
    beta_z_evo=1.0,
    lr_init=.05e-3,
    lr_factor=1.,
    lr_schedule=[1],
    n_steps=steps_per_epoch * 6750,
    report_interval=steps_per_epoch * 150,
    l2_lambda_w=0e-6,
    l2_lambda_b=0e-6,
    gradskip_threshold=1e10,
    gradclip_threshold=1e9,
    save_dir='log_TDVAE40',
    restore_from=None,
    tb_dir=None,
    activation=tf.math.softplus
)

"""
--------------------
EVALUATION HYPERPARAMETERS
--------------------
"""
eval_params = Hyperparams(
    # Val section: Defines parameters only useful during validation

    # Defines how many validation samples to validate on every time we're going to write to tensorboard
    # Reduce this number of faster validation. Very small subsets can be non descriptive of the overall distribution
    n_samples_for_validation=5000,
    # validation batch size (global for all devices)
    batch_size=32 * 2
)

"""
--------------------
SYNTHESIS HYPERPARAMETERS
--------------------
"""
synthesis_params = Hyperparams(
    # Synthesis section: Defines parameters only useful during synthesis/inference

    # The synthesis mode can be one of ('reconstruction', 'generation', 'div_stats', 'encoding')
    synthesis_mode='reconstruction',

    # Whether or not to use the EMA weights for inference
    load_ema_weights=True,

    # reconstruction/encoding mode
    # Defines the quantile at which to prune the latent space (section 7).
    # Example: variate_masks_quantile = 0.03 means only 3% of the posteriors that encode the most information will be preserved, all the others will be replaced with the prior.
    # Encoding mode will always automatically prune the latent space using this argument, so it's a good idea to run masked reconstruction (read below) to find a suitable value of
    # variate_masks_quantile before running encoding mode.
    variate_masks_quantile=0.03,

    # Reconstruction mode
    # Whether to save the targets during reconstruction (for debugging)
    save_target_in_reconstruction=False,
    # Whether to prune the posteriors to variate_masks_quantile. If set to True, the reconstruction is run with only variate_masks_quantile posteriors.
    # All the other variates will be replaced with the prior. Used to compute the NLL at different % of prune posteriors, and to determine an appropriate variate_masks_quantile
    # that doesn't hurt NLL.
    mask_reconstruction=False,

    # div_stats mode
    # Defines the ratio of the training data to compute the average KL per variate on (used for masked reconstruction and encoding).
    # Set to 1. to use the full training dataset. But that' usually an overkill as 5%, 10% or 20% of the dataset tends to be representative enough.
    div_stats_subset_ratio=0.2,

    # generation_mode
    # Number of generated batches per temperature from the temperature_settings list.
    n_generation_batches=1,
    # Temperatures for unconditional generation from the prior.
    # We generate n_generation_batches for each element of the temperature_settings list. This is implemented so that many temperatures can be tested in the same run for speed.
    # The temperature_settings elements can be one of:
    #     - A float: Example 0.8. Defines the temperature used for all the latent variates of the model
    #     - A tuple of 3: Example ('linear', 0.6, 0.9). Defines a linearly increasing temperature scheme from the deepest to shallowest top-down block. (different temperatures per latent
    #                             group)
    #     - A list of size len(down_strides): Each element of the list defines the temperature for their respective top-down blocks.
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
    type='mlp',
    n_hiddens=[2000],
    activation=torch.nn.ReLU
)

cnn_params = Hyperparams(
    type="conv",
    filters=[300, 20],
    strides=[1],
    kernel_size=3,
    activation=torch.nn.ReLU
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
shared_encoder_params = Hyperparams(
    type='mlp',
    n_hiddens=[2000],
    activation=torch.nn.ReLU
)

cluster_encoder_params = Hyperparams(
    type='mlp',
    n_hiddens=[1000, 500, 250],
    activation=torch.nn.ReLU
)

latent_encoder_params = Hyperparams(
    type='mlp',
    n_hiddens=[250, 500, 1000, 2000],
    activation=torch.nn.ReLU,
)

latent_decoder_params = Hyperparams(
    type='mlp',
    n_hiddens=[2000],
    activation=torch.nn.ReLU
)

data_decoder_params = Hyperparams(
    type='mlp',
    n_hiddens=[2000],
    activation=torch.nn.ReLU
)

z1_distr = Hyperparams(
    distr='laplace',
    sigma_nonlin='exp',
    sigma_param='var'
)

z2_distr = Hyperparams(
    distr='normal',
    sigma_nonlin='exp',
    sigma_param='var'
)
