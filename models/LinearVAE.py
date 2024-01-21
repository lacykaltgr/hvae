def _model():
    from hvae_backbone.block import InputBlock, OutputBlock, GenBlock
    from hvae_backbone.hvae import hVAE as hvae
    from hvae_backbone.elements.layers import Unflatten, Flatten, FixedStdDev
    from hvae_backbone.utils import OrderedModuleDict

    _blocks = OrderedModuleDict(
        x=InputBlock(
            net=Flatten(start_dim=1),
        ),
        z=GenBlock(
            prior_net=None,
            posterior_net=x_to_z_net,
            input_id="z_prior",
            condition="x",
            output_distribution="laplace",
        ),
        x_hat=OutputBlock(
            net=[z_to_x_net, FixedStdDev(0.4), Unflatten(1, (2, *data_params.shape[1:]))],
            input_id="z",
            output_distribution="normal"
        ),
    )

    prior_shape = (450, )
    _prior = dict(
        z_prior=torch.cat([torch.zeros(prior_shape), torch.ones(prior_shape)], 0).to("cuda")
    )

    __model = hvae(
        blocks=_blocks,
        init=_prior
    )

    return __model

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
    name='LinearVAE',

    # TRAIN LOG
    # --------------------
    # Defines how often to save a model checkpoint and logs (tensorboard) to disk.
    checkpoint_interval_in_steps=3000,
    eval_interval_in_steps=3000,

    load_from_train=None,
    load_from_eval='2023-09-23__16-12/checkpoints/checkpoint-750.pth',
)

"""
--------------------
MODEL HYPERPARAMETERS
--------------------
"""

model_params = Hyperparams(
    model=_model,
    device='cuda',
    seed=420,

    # Latent layer distribution base can be in ('std', 'logstd').
    # Determines if the model should predict
    # std (with softplus) or logstd (std is computed with exp(logstd)).
    distribution_base='logstd',
    distribution_sigma_param="var",

    # Latent layer Gradient smoothing beta. ln(2) ~= 0.6931472.
    # Setting this parameter to 1. disables gradient smoothing (not recommended)
    gradient_smoothing_beta=0.6931472,
)

"""
--------------------
DATA HYPERPARAMETERS
--------------------
"""
from data.textures.textures import TexturesDataset
data_params = Hyperparams(
    dataset=TexturesDataset,
    params=dict(type="natural", image_size=20, whitening="old"),

    # Image metadata
    shape=(1, 20, 20),
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
    learning_rate=0.001,

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
    vae_beta_anneal_start=24000,
    vae_beta_anneal_steps=30000,
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
    n_samples_for_validation=10000,
    n_samples_for_reconstruction=4,

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
    in_filters=1,
    filters=[25, 25, 40],
    kernel_size=3,
    pools=[0],
    activation=torch.nn.Softplus(),
    activate_output=False
)


"""
--------------------
CUSTOM BLOCK HYPERPARAMETERS
--------------------
"""
# add your custom block hyperparameters here

x_size = 20*20 # 400
z_size = 450


x_to_z_net = Hyperparams(
    type='mlp',
    input_size=x_size,
    hidden_sizes=[500, 500],
    output_size=2*z_size,
    activation=torch.nn.Softplus(),
    residual=False,
    activate_output=True
)

z_to_x_net = Hyperparams(
    type='mlp',
    input_size=z_size,
    hidden_sizes=[],
    output_size=x_size,
    activation=None,
    residual=False,
    activate_output=False
)
