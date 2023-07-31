import numpy as np
import torch
from torch import nn

from elements.losses import KLDivergence
from elements.layers import GaussianLatentLayer
from elements.models import get_model


class _Block(nn.Module):
    def __init__(self, input, log_output=False, **block_params):
        super(_Block, self).__init__()
        self.input = input
        self.output = self.input + "_out"
        self.log_output = log_output
        self.block_params = block_params

    def set_output(self, output):
        self.output = output


class ConcatBlock(_Block):
    def __init__(self, inputs, dimension, log_output=False, **block_params):
        super(ConcatBlock, self).__init__(inputs, log_output, **block_params)
        self.dimension = dimension

    def forward(self, computed):
        if not all([inp in computed for inp in self.inputs]):
            raise ValueError("Not all inputs found in computed")
        if len(self.inputs) != 2:
            raise ValueError("ConcatBlock only supports two inputs")
        x1, x2 = [computed[inp] for inp in self.inputs]
        return torch.cat([x1, x2], dim=self.dimension)


class InputBlock(_Block):
    def __init__(self, net, log_output=False, **block_params):
        super(InputBlock, self).__init__(input, log_output, **block_params)
        self.net = net

    def forward(self, inputs):
        computed = {self.input: inputs}
        if self.net is None:
            return inputs, computed
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class OutputBlock(_Block):
    def __init__(self, net, input, log_output=False, **block_params):
        super(OutputBlock, self).__init__(input, log_output, **block_params)
        self.net = net

    def forward(self, computed):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class EncBlock(_Block):
    def __init__(self, net, input, log_output=False, **block_params):
        super(EncBlock, self).__init__(input, log_output, **block_params)
        self.net = net

    def forward(self, computed):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class DecBlock(_Block):

    kl_divergence = KLDivergence()
    gaussian_diag_samples = GaussianLatentLayer()

    def __init__(self,
                 prior_net,
                 posterior_net,
                 input, condition, log_output,
                 **block_params):
        super(DecBlock, self).__init__(input, log_output, **block_params)
        self.prior_net = get_model(prior_net)
        self.posterior_net = get_model(posterior_net)
        self.condition = condition

    def _sample(self, y, cond):
        qm, qv = self.posterior_net(torch.cat([y, cond], dim=1)).chunk(2, dim=1)
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2)
        z = self.draw_gaussian_diag_samples(qm, qv)
        kl = self.kl_divergence(qm, pm, qv, pv)
        return z, kl

    def _sample_uncond(self, y, t=None):
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        z = draw_gaussian_diag_samples(pm, pv)
        return z

    def forward(self, computed):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        if self.condition not in computed:
            raise ValueError("Condition {} not found in computed".format(self.condition))
        x = computed[self.input]
        cond = computed[self.condition]
        z, kl = self._sample(x, cond)
        computed[self.output] = z
        return z, computed, kl

    def sample_from_prior(self, computed, t=None):
        x = computed[self.input]
        z = self._sample_uncond(x, t)
        return z, computed


class TopBlock(DecBlock):
    def __init__(self,
                 prior_net,
                 posterior_net,
                 prior_trainable,
                 input, log_output=False,
                 **block_params):
        super(TopBlock, self).__init__(prior_net, posterior_net, input, log_output, **block_params)
        if prior_trainable:
            H, W, C = None, None, None
            self.trainable_h = torch.nn.Parameter(  # for unconditional generation
                data=torch.empty(size=(1, C, H, W)), requires_grad=True)
            nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')

    def forward(self, computed):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        if self.condition not in computed:
            raise ValueError("Condition {} not found in computed".format(self.condition))
        x = computed[self.input]
        cond = computed[self.condition]
        z, kl = self._sample(x, cond)
        computed[self.output] = z
        return z, computed, kl

    def sample_from_prior(self, batch_size, t=None):
        y = torch.tile(self.trainable_h, (batch_size, 1, 1, 1))
        z = self._sample_uncond(y, t)
        computed = {
            "trainable_h": y,
            self.output: z}
        return z, computed


class DecBlockResidual(DecBlock):
    def __init__(self,
                 model,
                 prior_net,
                 posterior_net,
                 z_projection,
                 zdim,
                 input, log_output,
                 **block_params):
        super(DecBlockResidual, self).__init__(model, input, log_output, **block_params)
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
        pm, pv, kl_residual = y_prior[:, :self.zdim, ...], y_prior[:, self.zdim:self.zdim * 2, ...], y_prior[:,
                                                                                                     self.zdim * 2:,
                                                                                                     ...]
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
