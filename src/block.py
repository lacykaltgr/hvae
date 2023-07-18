import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import generate_loc_scale_distr, draw_gaussian_diag_samples, gaussian_analytical_kl


class _Block(nn.Module):
    def __init__(self, model, input, output, log_output=False, **block_params):
        super(_Block, self).__init__()
        self.input = input
        self.output = output
        self.log_output = log_output
        self.block_params = block_params
        self.model = self._get_model(model)

    def forward(self, inputs):
        logits = self.model(inputs)
        return generate_loc_scale_distr(logits=logits, **self.region_params)

    def _get_model(self, model):
        if model is None:
            return None
        elif isinstance(model, str):
            # Load model from hparams file
            pass
        elif isinstance(model, nn.Module):
            # Load model from nn.Module
            return model
        elif isinstance(model, dict):
            # Load model from dictionary
            pass
        elif isinstance(model, list):
            # Load model from list
            pass
        else:
            raise NotImplementedError("Model type not supported.")


class EncBlock(_Block):
    def __init__(self, model, input, output, log_output, **block_params):
        super(EncBlock, self).__init__(model, input, output, log_output, **block_params)

    def forward(self, inputs):
        return self.model(inputs)


class DecBlock(_Block):
    def __init__(self,
                 model,
                 prior_net,
                 posterior_net,
                 encoder,
                 z_projection,
                 zdim,
                 input, output, log_output,
                 **block_params):
        super(DecBlock, self).__init__(model, input, output, log_output, **block_params)
        self.posterior_net = self._get_model(posterior_net)
        self.prior_net = self._get_model(prior_net)
        self.encoder = self._get_model(encoder)
        self.zdim = zdim
        self.z_projection = self.get_model(z_projection)

    def sample(self, y, activations):
        qm, qv = self.encoder(torch.cat([y, activations], dim=1)).chunk(2, dim=1)
        y_prior = self.prior(y)
        pm, pv, kl_residual = y_prior[:, :self.zdim, ...], y_prior[:, self.zdim:self.zdim * 2, ...], y_prior[:, self.zdim * 2:, ...]
        y = y + kl_residual
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, y, kl

    def sample_uncond(self, y, t=None, lvs=None):
        y_prior = self.prior(y)
        pm, pv, kl_residual = y_prior[:, :self.zdim, ...], y_prior[:, self.zdim:self.zdim * 2, ...], y_prior[:, self.zdim * 2:, ...]
        y = y + kl_residual
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, y

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, ys, activations, get_latents=False):
        x, acts = self.get_inputs(ys, activations)
        if self.mixin is not None:
            x = x + F.interpolate(ys[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl = self.sample(x, acts)
        x = x + self.z_projection(z)
        x = self.model(x)
        ys[self.base] = x
        if get_latents:
            return ys, dict(z=z.detach(), kl=kl)
        return ys, dict(kl=kl)

    def forward_uncond(self, ys, t=None, lvs=None):
        try:
            x = ys[self.base]
        except KeyError:
            ref = ys[list(ys.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(ys[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        ys[self.base] = x
        return ys


