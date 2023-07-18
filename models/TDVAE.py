from src.block import EncBlock, DecBlock
from src.hvae import hVAE as hvae

import torch

_device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

_blocks = dict(
    cluster_encoder=EncBlock("mlp",
                             input="x", output="y"),
    latent_encoder=EncBlock("conv",
                            input=["x", "y"], output="z1",  log_output=True),
    latent_decoder=DecBlock("conv",
                            prior_net="conv",
                            posterior_net="conv",
                            encoder="mlp",
                            z_projection="conv",
                            zdim=16,
                            input="y", output="z"),
    data_decoder=DecBlock("mlp",
                          prior_net="conv",
                          posterior_net="conv",
                          encoder="mlp",
                          z_projection="conv",
                          zdim=16,
                          input="z", output="x")
)


# custom function
def _infer_latent(x, log, **ops):
    z1 = ops["cluster_encoder"](x)
    return z1


hvae_model = hvae(
    blocks=_blocks,
    name="hvae",
    infer_latent=_infer_latent,
    device=_device
)
