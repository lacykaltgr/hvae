import numpy as np
import torch
from torch import nn
from typing import List
from torch import tensor

from elements.losses import KLDivergence
from elements.layers import GaussianLatentLayer
from elements.models import get_model


class _Block(nn.Module):
    def __init__(self, input_id: str or List[str] = None):
        super(_Block, self).__init__()
        self.input = input_id
        self.output = self.input + "_out"

    def set_output(self, output: str) -> None:
        self.output = output


class ConcatBlock(_Block):
    def __init__(self, inputs: List[str], dimension: int = 1):
        super(ConcatBlock, self).__init__(inputs)
        self.dimension = dimension

    def forward(self, computed: dict) -> (tensor, dict):
        if not all([inp in computed for inp in self.inputs]):
            raise ValueError("Not all inputs found in computed")
        if len(self.inputs) != 2:
            raise ValueError("ConcatBlock only supports two inputs")
        assert len(self.inputs) == 2
        x, skip = [computed[inp] for inp in self.inputs]
        x_skip = torch.cat([x, skip], dim=self.dimension)
        computed[self.output] = x_skip
        return x_skip, computed


class InputBlock(_Block):
    def __init__(self, net):
        super(InputBlock, self).__init__()
        self.net = net

    def forward(self, inputs: tensor) -> (tensor, dict):
        computed = {self.input: inputs}
        if self.net is None:
            return inputs, computed
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class OutputBlock(_Block):
    def __init__(self, net, input_id: str):
        super(OutputBlock, self).__init__(input_id)
        self.net = net

    def forward(self, computed: dict) -> (tensor, dict):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class EncBlock(_Block):
    def __init__(self, net, input_id: str):
        super(EncBlock, self).__init__(input_id)
        self.net = net

    def forward(self, computed: dict) -> (tensor, dict):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return output, computed


class SimpleDecBlock(_Block):

    gaussian_diag_samples = GaussianLatentLayer()

    def __init__(self, net, input: str):
        super(SimpleDecBlock, self).__init__(input)
        self.prior_net = get_model(net)

    def _sample_uncond(self, y: tensor, t: float or int=None) -> tensor:
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        z = self.draw_gaussian_diag_samples(pm, pv)
        return z

    def forward(self, computed: dict) -> (tensor, dict, tuple):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        x = computed[self.input]
        z = self._sample_uncond(x)
        computed[self.output] = z
        return z, computed, None

    def sample_from_prior(self, computed: dict, t: float or int = None) -> (tensor, dict):
        x = computed[self.input]
        z = self._sample_uncond(x, t)
        computed[self.output] = z
        return z, computed


class DecBlock(SimpleDecBlock):

    kl_divergence = KLDivergence()

    def __init__(self,
                 prior_net,
                 posterior_net,
                 input_id: str, condition: str):
        super(DecBlock, self).__init__(prior_net, input_id)
        self.prior_net = get_model(prior_net)
        self.posterior_net = get_model(posterior_net)
        self.condition = condition

    def _sample(self, y: tensor, cond: tensor, variate_mask=None) -> (tensor, tuple):
        qm, qv = self.posterior_net(torch.cat([y, cond], dim=1)).chunk(2, dim=1)
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2)
        z = self.draw_gaussian_diag_samples(qm, qv)

        if variate_mask is not None:
            z_prior = self.draw_gaussian_diag_samples(pm, pv)
            z = self.prune(z, z_prior, variate_mask)

        kl = self.kl_divergence(qm, pm, qv, pv)
        return z, kl

    def _sample_uncond(self, y: tensor, t: float or int = None) -> tensor:
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        z = self.draw_gaussian_diag_samples(pm, pv)
        return z

    def forward(self, computed: dict, variate_mask=None) -> (tensor, dict, tuple):
        if self.input not in computed:
            raise ValueError("Input {} not found in computed".format(self.input))
        if self.condition not in computed:
            raise ValueError("Condition {} not found in computed".format(self.condition))
        x = computed[self.input]
        cond = computed[self.condition]
        z, kl = self._sample(x, cond, variate_mask)
        computed[self.output] = z
        return z, computed, kl

    def sample_from_prior(self, computed: dict, t: float or int = None) -> (tensor, dict):
        x = computed[self.input]
        z = self._sample_uncond(x, t)
        computed[self.output] = z
        return z, computed

    @staticmethod
    def prune(z, z_prior, variate_mask=None):
        variate_mask = torch.Tensor(variate_mask)[None, :, None, None].cuda()
        # Only used in inference mode to prune turned-off variates
        # Use posterior sample from meaningful variates, and prior sample from "turned-off" variates
        # The NLL should be similar to using z_post without masking if the mask is good (not very destructive)
        # variate_mask automatically broadcasts to [batch_size, H, W, n_variates]
        z = variate_mask * z + (1. - variate_mask) * z_prior
        return z


class TopBlock(DecBlock):
    def __init__(self, net,
                 prior_trainable: bool,
                 condition: str):

        super(TopBlock, self).__init__(None, net, 'trainable_h', condition)

        H, W, C = None, None, None
        #TODO: itt fent hparamsból kell valami
        if prior_trainable:
            self.trainable_h = torch.nn.Parameter(  # for unconditional generation
                data=torch.empty(size=(1, C, H, W)), requires_grad=True)
            nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        else:
            # constant tensor with 0 values
            self.trainable_h = torch.zeros(size=(1, C, H, W), requires_grad=False)

    def forward(self, computed: dict, variate_mask=None) -> (tensor, dict, tuple):
        if self.condition not in computed:
            raise ValueError("Condition {} not found in computed".format(self.condition))
        x = self.trainable_h
        cond = computed[self.condition]
        z, kl = self._sample(x, cond)
        computed[self.output] = z
        return z, computed, kl

    def sample_from_prior(self, batch_size: int, t: int or float = None) -> (tensor, dict):
        y = torch.tile(self.trainable_h, (batch_size, 1, 1, 1))
        z = self._sample_uncond(y, t)
        computed = {
            self.input: y,
            self.output: z}
        return z, computed


class ResidualDecBlock(DecBlock):
    def __init__(self, net,
                 prior_net,
                 posterior_net,
                 z_projection,
                 input, condition):
        super(ResidualDecBlock, self).__init__(prior_net, posterior_net, input, condition)
        self.net = get_model(net)
        self.prior_net = get_model(prior_net)
        self.posterior_net = get_model(posterior_net)
        self.z_projection = get_model(z_projection)

    def _sample(self, y: tensor, cond: tensor, variate_mask=None) -> (tensor, tensor, tuple):
        qm, qv = self.posterior_net(torch.cat([y, cond], dim=1)).chunk(2, dim=1)
        y_prior = self.prior_net(y)
        pm, pv, kl_residual = torch.chunk(y_prior, chunks=3)

        z = self.draw_gaussian_diag_samples(qm, qv)

        if variate_mask is not None:
            z_prior = self.draw_gaussian_diag_samples(pm, pv)
            z = self.prune(z, z_prior, variate_mask)

        y = y + kl_residual
        kl = self.gaussian_analytical_kl(qm, pm, qv, pv)
        return z, y, kl

    def _sample_uncond(self, y: tensor, t: float or int = None) -> (tensor, tensor):
        y_prior = self.prior_net(y)
        pm, pv, kl_residual = torch.chunk(y_prior, chunks=3)
        y = y + kl_residual
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        z = self.draw_gaussian_diag_samples(pm, pv)
        return z, y

    def forward(self, computed: dict, variate_mask=None) -> (tensor, dict, tuple):
        x = computed[self.input]
        cond = computed[self.condition]
        z, x, kl = self._sample(x, cond, variate_mask)
        x = x + self.z_projection(z)
        x = self.net(x)
        computed[self.output] = x
        return x, computed, kl

    def sample_from_prior(self, computed: dict, t: float or int = None) -> (tensor, dict):
        x = computed[self.input]
        z, x = self._sample_uncond(x, t)
        x = x + self.z_projection(z)
        x = self.net(x)
        computed[self.output] = x
        return x, computed
