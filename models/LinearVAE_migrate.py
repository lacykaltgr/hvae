import torch

def _model(migration):
    from hvae_backbone.block import InputBlock, SimpleBlock, GenBlock, OutputBlock, SimpleGenBlock
    from hvae_backbone.hvae import hVAE as hvae
    from hvae_backbone.elements.nets import BlockNet
    from hvae_backbone.elements.layers import Unflatten, Flatten, FixedStdDev, KeepShapeWithValue, EinsumLayer
    from hvae_backbone.utils import OrderedModuleDict

    _blocks = OrderedModuleDict(
        x=InputBlock(
            net=Flatten(start_dim=1),
        ),
        hiddens=SimpleBlock(
            net=migration.get_net("mlp_shared_encoder", activate_output=True),
            input_id="x",
        ),
        y=GenBlock(
            prior_net=None,
            posterior_net=[migration.get_net("mlp_cluster_encoder_final", activate_output=False),
                           KeepShapeWithValue(1)],
            input_id="y_prior",
            condition="hiddens",
            output_distribution="onehot_categorical",
        ),
        z=GenBlock(
            prior_net=BlockNet(
                y=InputBlock(),
                z_prior_mu=SimpleBlock(
                    input_id="y",
                    net=migration.get_net("latent_prior_mu", activate_output=False),
                ),
                z_prior_sigma=SimpleBlock(
                    input_id="y",
                    net=migration.get_net("latent_prior_sigma", activate_output=False),
                ),
                output=BlockNet.OutputBlock(
                    input_id=(["z_prior_mu", "z_prior_sigma"], "concat"),
                )
            ),
            posterior_net=None,
            input_id="y",
            condition=[("y",
                        ["hiddens", migration.get_net("mlp_latent_encoder_0", activate_output=False)]),
                       EinsumLayer(equation='ij,ik->ik')],
            output_distribution="laplace"
        ),
        x_hat=OutputBlock(
            net=[migration.get_net("mlp_latent_decoder", activate_output=False),
                 Unflatten(1,  data_params.shape),
                 FixedStdDev(0.4)],
            input_id="z",
            output_distribution="normal"
        ),
    )

    _prior = dict(
        y_prior=torch.ones(1, 2)
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
MIGRATION HYPERPARAMETERS
--------------------
"""
from migration.TDVAE_migration.migration_agent import TDVAEMigrationAgent
migration_params = Hyperparams(
    params=dict(
        path="migration/TDVAE_migration/weights_LinearVAE/eval_LinearVAE40/mycurl-30000000"
    ),
    migration_agent=TDVAEMigrationAgent
)

"""
--------------------
LOGGING HYPERPARAMETERS
--------------------
"""
log_params = Hyperparams(
    name='LinearVAE_migrate',

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
    gradient_smoothing_beta=0.6931472,
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
    n_samples_for_reconstruction=3,
    # validation batch size
    batch_size=128,

    use_mean=True,
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




