import numpy as np
import torch
from torch import tensor

from hvae_backbone.block import OutputBlock, InputPipeline
from hvae_backbone.utils import split_mu_sigma, SerializableSequential as Sequential
from hvae_backbone.elements.distributions import generate_distribution


class ContrastiveOutputBlock(OutputBlock):

    # only for 1D inputs
    def __init__(self, net, input_id, contrast_dims: int, output_distribution: str = 'normal'):
        super().__init__(net, input_id, output_distribution)
        self.contrast_dims = contrast_dims

    def _sample_uncond(self, y: tensor, t: float or int = None, use_mean=False) -> tensor:
        y_input = y[:, :-self.contrast_dims]
        contrast = y[:, -self.contrast_dims:]
        y_prior = self.prior_net(y_input)
        pm, pv = split_mu_sigma(y_prior)
        pm_shape = pm.shape
        pm_flattened = torch.flatten(pm, start_dim=1)
        pm = pm_flattened * contrast
        pm = pm.reshape(pm_shape)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        prior = generate_distribution(pm, pv, self.output_distribution)
        z = prior.sample() if not use_mean else prior.mean
        return z, (prior, None)

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["contrast_dims"] = self.contrast_dims
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        return ContrastiveOutputBlock(
            net=prior_net,
            input_id=InputPipeline.deserialize(serialized["input"]),
            contrast_dims=serialized["contrast_dims"],
            output_distribution=serialized["output_distribution"]
        )