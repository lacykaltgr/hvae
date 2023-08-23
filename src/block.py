from typing import List

import numpy as np
import torch
from torch import nn
from torch import tensor

from src.elements.distributions import generate_distribution
from src.elements.nets import get_net
from src.utils import SerializableModule, SerializableSequential as Sequential


class _Block(SerializableModule):
    def __init__(self, input_id: str or List[str] = None):
        super(_Block, self).__init__()
        self.input = input_id
        self.output = self.input + "_out" if self.input is not None else None

    def set_output(self, output: str) -> None:
        self.output = output

    def serialize(self) -> dict:
        return dict(
            input=self.input,
            output=self.output,
            type=self.__class__
        )


class SimpleBlock(_Block):
    def __init__(self, net, input_id: str):
        super(SimpleBlock, self).__init__(input_id)
        self.net: Sequential = get_net(net)

    def forward(self, computed: dict) -> dict:
        if self.input not in computed.keys():
            raise ValueError(f"Input {self.input} not found in computed")
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return computed

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["net"] = self.net.serialize()
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return SimpleBlock(net=net, input_id=serialized["input"])


class ConcatBlock(_Block):
    def __init__(self, inputs: List[str], dimension: int = 1):
        super(ConcatBlock, self).__init__(inputs)
        self.dimension = dimension

    def forward(self, computed: dict) -> dict:
        if not all([inp in computed for inp in self.inputs]):
            raise ValueError("Not all inputs found in computed")
        if len(self.inputs) != 2:
            raise ValueError("ConcatBlock only supports two inputs")
        assert len(self.inputs) == 2
        x, skip = [computed[inp] for inp in self.inputs]
        x_skip = torch.cat([x, skip], dim=self.dimension)
        computed[self.output] = x_skip
        return computed

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["dimension"] = self.dimension
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        return ConcatBlock(
            inputs=serialized["inputs"],
            dimension=serialized["dimension"]
        )


class InputBlock(SimpleBlock):
    def __init__(self, net):
        super(InputBlock, self).__init__(net, "input")

    def forward(self, inputs: tensor) -> dict:
        return {self.output: inputs} \
            if self.net is None else \
            {self.input: inputs,
             self.output: self.net(inputs)}

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return InputBlock(net=net)


class EncBlock(SimpleBlock):
    def __init__(self, net, input_id: str):
        super(EncBlock, self).__init__(net, input_id)

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return EncBlock(net=net, input_id=serialized["input"])


class SimpleDecBlock(_Block):

    def __init__(self, net,
                 input_id: str,
                 output_distribution: str = 'normal'):
        super(SimpleDecBlock, self).__init__(input_id)
        self.prior_net: Sequential = get_net(net)
        self.output_distribution: str = output_distribution

    def _sample_uncond(self, y: tensor, t: float or int = None) -> tensor:
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2, dim=1)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        prior = generate_distribution(pm, pv, self.output_distribution)
        z = prior.sample()
        return z, (prior, None)

    def forward(self, computed: dict) -> (dict, tuple):
        if self.input not in computed.keys():
            raise ValueError(f"Input {self.input} not found in computed")
        x = computed[self.input]
        z, distribution = self._sample_uncond(x)
        computed[self.output] = z
        return computed, distribution

    def sample_from_prior(self, computed: dict, t: float or int = None) -> dict:
        x = computed[self.input]
        z, _ = self._sample_uncond(x, t)
        computed[self.output] = z
        return computed

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["prior_net"] = self.prior_net.serialize()
        serialized["output_distribution"] = self.output_distribution
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        return SimpleDecBlock(
            net=prior_net,
            input_id=serialized["input"],
            output_distribution=serialized["output_distribution"]
        )


class OutputBlock(SimpleDecBlock):
    def __init__(self, net,
                 input_id: str,
                 output_distribution: str = 'normal'):
        super(OutputBlock, self).__init__(net, input_id, output_distribution)

    def forward(self, computed: dict) -> (tensor, dict, tuple):
        computed, distribution = super().forward(computed)
        output_sample = computed[self.output]
        return output_sample, computed, distribution

    def sample_from_prior(self, computed: dict, t: float or int = None) -> (tensor, dict):
        computed = super().sample_from_prior(computed, t)
        output_sample = computed[self.output]
        return output_sample, computed

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        return OutputBlock(
            net=prior_net,
            input_id=serialized["input"],
            output_distribution=serialized["output_distribution"]
        )


class DecBlock(SimpleDecBlock):

    def __init__(self,
                 prior_net,
                 posterior_net,
                 input_id: str, condition: str,
                 output_distribution: str = 'normal'):
        super(DecBlock, self).__init__(prior_net, input_id, output_distribution)
        self.prior_net: Sequential = get_net(prior_net)
        self.posterior_net: Sequential = get_net(posterior_net)
        self.condition = condition

    def _sample(self, y: tensor, cond: tensor, variate_mask=None) -> (tensor, tuple):
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2, dim=1)
        prior = generate_distribution(pm, pv, self.output_distribution)

        qm, qv = self.posterior_net(torch.cat([y, cond], dim=1)).chunk(2, dim=1)
        posterior = generate_distribution(qm, qv, self.output_distribution)
        z = posterior.sample()

        if variate_mask is not None:
            z_prior = prior.sample()
            z = self.prune(z, z_prior, variate_mask)

        return z, (prior, posterior)

    def _sample_uncond(self, y: tensor, t: float or int = None) -> tensor:
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2, dim=1)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)

        prior = generate_distribution(pm, pv, self.output_distribution)
        z = prior.sample()
        return z

    def forward(self, computed: dict, variate_mask=None) -> (dict, tuple):
        if self.input not in computed.keys():
            raise ValueError(f"Input {self.input} not found in computed")
        if self.condition not in computed.keys():
            raise ValueError(f"Condition {self.condition} not found in computed")
        x = computed[self.input]
        cond = computed[self.condition]
        z, distributions = self._sample(x, cond, variate_mask)
        computed[self.output] = z
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None) -> dict:
        x = computed[self.input]
        z = self._sample_uncond(x, t)
        computed[self.output] = z
        return computed

    @staticmethod
    def prune(z, z_prior, variate_mask=None):
        variate_mask = torch.Tensor(variate_mask)[None, :, None, None].cuda()
        # Only used in inference mode to prune turned-off variates
        # Use posterior sample from meaningful variates, and prior sample from "turned-off" variates
        # The NLL should be similar to using z_post without masking if the mask is good (not very destructive)
        # variate_mask automatically broadcasts to [batch_size, H, W, n_variates]
        z = variate_mask * z + (1. - variate_mask) * z_prior
        return z

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["prior_net"] = self.prior_net.serialize()
        serialized["posterior_net"] = self.posterior_net.serialize()
        serialized["condition"] = self.condition
        serialized["output_distribution"] = self.output_distribution
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        posterior_net = Sequential.deserialize(serialized["posterior_net"])
        return DecBlock(
            prior_net=prior_net,
            posterior_net=posterior_net,
            input_id=serialized["input"],
            condition=serialized["condition"],
            output_distribution=serialized["output_distribution"]
        )


class TopBlock(DecBlock):
    def __init__(self, net,
                 prior_shape: tuple,
                 prior_trainable: bool,
                 concat_prior: bool,
                 condition: str,
                 output_distribution: str = 'normal'):
        super(TopBlock, self).__init__(None, net, 'trainable_h', condition, output_distribution)
        self.concat_prior = concat_prior
        if prior_trainable:
            self.trainable_h = torch.nn.Parameter(  # for unconditional generation
                data=torch.empty(
                    size=prior_shape if len(prior_shape) > 1 else (1, *prior_shape)),
                requires_grad=True)
            nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        else:
            # constant tensor with 0 values
            self.trainable_h = torch.zeros(size=prior_shape, requires_grad=False)

    def _sample(self, y: tensor, cond: tensor, variate_mask=None) -> (tensor, tuple):
        y_prior = self.prior_net(y)
        pm, pv = torch.chunk(y_prior, chunks=2, dim=1)
        prior = generate_distribution(pm, pv, self.output_distribution)

        posterior_input = torch.cat([y, cond], dim=1) if self.concat_prior else cond
        qm, qv = self.posterior_net(posterior_input).chunk(2, dim=1)
        posterior = generate_distribution(qm, qv, self.output_distribution)
        z = posterior.sample()

        if variate_mask is not None:
            z_prior = prior.sample()
            z = self.prune(z, z_prior, variate_mask)

        return z, (prior, posterior)

    def forward(self, computed: dict, variate_mask=None) -> (tensor, dict, tuple):
        if self.condition not in computed.keys():
            raise ValueError(f"Condition {self.condition} not found in computed")
        cond = computed[self.condition]
        x = torch.tile(self.trainable_h, (cond.shape[0], 1))
        if cond.shape != x.shape and self.concat_prior:
            x = x.resize(cond.shape)
        z, distributions = self._sample(x, cond)
        computed[self.output] = z
        return computed, distributions

    def sample_from_prior(self, batch_size: int, t: int or float = None) -> (tensor, dict):
        y = torch.tile(self.trainable_h, (batch_size, 1))
        z = self._sample_uncond(y, t)
        computed = {
            self.input: y,
            self.output: z}
        return computed

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["trainable_h"] = self.trainable_h
        serialized["concat_prior"] = self.concat_prior
        serialized["prior_shape"] = self.trainable_h.shape
        serialized["prior_trainable"] = self.trainable_h.requires_grad
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["posterior_net"])
        return TopBlock(
            net=net,
            prior_shape=serialized["prior_shape"],
            prior_trainable=serialized["prior_trainable"],
            concat_prior=serialized["concat_prior"],
            condition=serialized["condition"],
            output_distribution=serialized["output_distribution"]
        )


class ResidualDecBlock(DecBlock):
    def __init__(self, net,
                 prior_net,
                 posterior_net,
                 z_projection,
                 input, condition,
                 output_distribution: str = 'normal'):
        super(ResidualDecBlock, self).__init__(prior_net, posterior_net, input, condition, output_distribution)
        self.net: Sequential = get_net(net)
        self.z_projection: Sequential = get_net(z_projection)

    def _sample(self, y: tensor, cond: tensor, variate_mask=None) -> (tensor, tensor, tuple):

        y_prior = self.prior_net(y)
        pm, pv, kl_residual = torch.chunk(y_prior, chunks=3, dim=1)
        prior = generate_distribution(pm, pv, self.output_distribution)

        qm, qv = self.posterior_net(torch.cat([y, cond], dim=1)).chunk(2, dim=1)
        posterior = generate_distribution(qm, qv, self.output_distribution)
        z = posterior.sample()

        if variate_mask is not None:
            z_prior = prior.sample()
            z = self.prune(z, z_prior, variate_mask)

        y = y + kl_residual
        return z, y, (prior, posterior)

    def _sample_uncond(self, y: tensor, t: float or int = None) -> (tensor, tensor):
        y_prior = self.prior_net(y)
        pm, pv, kl_residual = torch.chunk(y_prior, chunks=3, dim=1)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        prior = generate_distribution(pm, pv, self.output_distribution)
        z = prior.sample()
        y = y + kl_residual
        return z, y

    def forward(self, computed: dict, variate_mask=None) -> (dict, tuple):
        x = computed[self.input]
        cond = computed[self.condition]
        z, x, distributions = self._sample(x, cond, variate_mask)
        x = x + self.z_projection(z)
        x = self.net(x)
        computed[self.output] = x
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None) -> dict:
        x = computed[self.input]
        z, x = self._sample_uncond(x, t)
        x = x + self.z_projection(z)
        x = self.net(x)
        computed[self.output] = x
        return computed

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["net"] = self.net.serialize()
        serialized["z_projection"] = self.z_projection.serialize()
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        prior_net = Sequential.deserialize(serialized["prior_net"])
        posterior_net = Sequential.deserialize(serialized["posterior_net"])
        z_projection = Sequential.deserialize(serialized["z_projection"])
        return ResidualDecBlock(
            net=net,
            prior_net=prior_net,
            posterior_net=posterior_net,
            z_projection=z_projection,
            input=serialized["input"],
            condition=serialized["condition"],
            output_distribution=serialized["output_distribution"]
        )
