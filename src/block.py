import numpy as np
import torch
from torch import nn

from elements.losses import KLDivergence
from elements.layers import GaussianLatentLayer
from elements.models import get_model


class _Block(nn.Module):
    def __init__(self, input=None):
        super(_Block, self).__init__()
        self.input = input
        self.output = self.input + "_out"

    def set_output(self, output):
        self.output = output


class ConcatBlock(_Block):
    def __init__(self, inputs, dimension=1):
        super(ConcatBlock, self).__init__(inputs)
        self.dimension = dimension

    def forward(self, computed):
        if not all([inp in computed for inp in self.inputs]):
            raise ValueError("Not all inputs found in computed")
        if len(self.inputs) != 2:
            raise ValueError("ConcatBlock only supports two inputs")
        x1, x2 = [computed[inp] for inp in self.inputs]
        return torch.cat([x1, x2], dim=self.dimension)


class InputBlock(_Block):
    def __init__(self, net):
        super(InputBlock, self).__init__()
        self.net = net

    def forward(self, inputs):
        computed = {self.input: inputs}
        if self.net is None:
            return inputs, computed
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class OutputBlock(_Block):
    def __init__(self, net, input):
        super(OutputBlock, self).__init__(input)
        self.net = net

    def forward(self, computed):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class EncBlock(_Block):
    def __init__(self, net, input):
        super(EncBlock, self).__init__(input)
        self.net = net

    def forward(self, computed):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class SimpleDecBlock(_Block):

    gaussian_diag_samples = GaussianLatentLayer()

    def __init__(self, net, input):
        super(SimpleDecBlock, self).__init__(input)
        self.prior_net = get_model(net)

    def _sample_uncond(self, y, t=None):
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        z = self.draw_gaussian_diag_samples(pm, pv)
        return z

    def forward(self, computed):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        x = computed[self.input]
        z = self._sample_uncond(x)
        computed[self.output] = z
        return z, computed

    def sample_from_prior(self, computed, t=None):
        x = computed[self.input]
        z = self._sample_uncond(x, t)
        computed[self.output] = z
        return z, computed


class DecBlock(SimpleDecBlock):

    kl_divergence = KLDivergence()

    def __init__(self,
                 prior_net,
                 posterior_net,
                 input, condition):
        super(DecBlock, self).__init__(prior_net, input)
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
        z = self.draw_gaussian_diag_samples(pm, pv)
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
        computed[self.output] = z
        return z, computed


class TopBlock(DecBlock):
    def __init__(self,
                 net,
                 prior_trainable,
                 condition):
        super(TopBlock, self).__init__(None, net, None, condition)
        H, W, C = None, None, None
        #TODO: itt fent hparamsból kell valami
        if prior_trainable:
            self.trainable_h = torch.nn.Parameter(  # for unconditional generation
                data=torch.empty(size=(1, C, H, W)), requires_grad=True)
            nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        else:
            # constant tensor with 0 values
            self.trainable_h = torch.zeros(size=(1, C, H, W), requires_grad=False)

    def forward(self, computed):
        if self.condition not in computed:
            raise ValueError("Condition {} not found in computed".format(self.condition))
        x = self.trainable_h
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
                 net,
                 prior_net,
                 posterior_net,
                 z_projection,
                 zdim,
                 input, condition):
        super(DecBlockResidual, self).__init__(prior_net, posterior_net, input, condition)
        self.prior_net = get_model(prior_net)
        self.posterior_net = get_model(posterior_net)
        self.zdim = zdim
        self.z_projection = get_model(z_projection)

    def _sample(self, y, cond):
        qm, qv = self.posterior_net(torch.cat([y, cond], dim=1)).chunk(2, dim=1)
        y_prior = self.prior_net(y)
        pm, pv, kl_residual = torch.chunk(y_prior, chunks=3)

        y = y + kl_residual
        z = self.draw_gaussian_diag_samples(qm, qv)
        kl = self.gaussian_analytical_kl(qm, pm, qv, pv)
        return z, y, kl

    def _sample_uncond(self, y, t=None):
        y_prior = self.prior_net(y)
        pm, pv, kl_residual = torch.chunk(y_prior, chunks=3)
        y = y + kl_residual
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        z = self.draw_gaussian_diag_samples(pm, pv)
        return z, y

    def forward(self, computed):
        x = computed[self.input]
        cond = computed[self.condition]
        z, x, kl = self._sample(x, cond)
        x = x + self.z_projection(z)
        x = self.net(x)
        computed[self.output] = x
        return x, computed, kl

    def sample_from_prior(self, computed, t=None):
        x = computed[self.input]
        z, x = self._sample_uncond(x, t)
        x = x + self.z_projection(z)
        x = self.net(x)
        computed[self.output] = x
        return x
