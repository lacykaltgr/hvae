import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import draw_gaussian_diag_samples, gaussian_analytical_kl
from elements.model import get_model


class _Block(nn.Module):
    def __init__(self, model, input, output, log_output=False, **block_params):
        super(_Block, self).__init__()
        self.input = input
        self.output = output
        self.log_output = log_output
        self.block_params = block_params
        self.model = get_model(model)


class EncBlock(_Block):
    def __init__(self, model, input, output, log_output=False, **block_params):
        super(EncBlock, self).__init__(model, input, output, log_output, **block_params)

    def forward(self, inputs):
        return self.model(inputs)


class DecBlock(_Block):
    def __init__(self,
                 model,
                 prior_net,
                 posterior_net,
                 z_projection,
                 zdim,
                 input, output, log_output,
                 **block_params):
        super(DecBlock, self).__init__(model, input, output, log_output, **block_params)
        self.prior_net = get_model(prior_net)
        self.posterior_net = get_model(posterior_net)
        self.zdim = zdim
        self.z_projection = get_model(z_projection)

    def _sample(self, y, activations):
        qm, qv = self.posterior_net(torch.cat([y, activations], dim=1)).chunk(2, dim=1)
        y_prior = self.prior_net(y)
        pm, pv, kl_residual = y_prior[:, :self.zdim, ...], y_prior[:, self.zdim:self.zdim * 2, ...], y_prior[:,
                                                                                                     self.zdim * 2:,
                                                                                                     ...]
        y = y + kl_residual
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, y, kl

    def _sample_uncond(self, y, t=None, lvs=None):
        y_prior = self.prior_net(y)
        pm, pv, kl_residual = y_prior[:, :self.zdim, ...], y_prior[:, self.zdim:self.zdim * 2, ...], y_prior[:, self.zdim * 2:, ...]
        y = y + kl_residual
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, y

    def forward(self, x, acts, get_latents=False):
        z, x, kl = self._sample(x, acts)
        x = x + self.z_projection(z)
        x = self.model(x)
        if get_latents:
            return x, dict(z=z.detach(), kl=kl)
        return x, dict(kl=kl)

    def sample_from_prior(self, x, t=None, lvs=None):
        z, x = self._sample_uncond(x, t, lvs=lvs)
        x = x + self.z_projection(z)
        x = self.model(x)
        return x

