from typing import List
import numpy as np
import torch
from overrides import overrides
from torch import nn
from torch import tensor

from src.elements.distributions import generate_distribution
from src.elements.nets import get_net
from src.utils import SerializableModule, SerializableSequential as Sequential, split_mu_sigma

"""
Sampling and forward methods based on VDAVAE paper
"""


class _Block(SerializableModule):
    """
    Base class for all blocks
    """

    def __init__(self, input_id: str or List[str] = None):
        super(_Block, self).__init__()
        self.input = input_id
        self.output = self.input + "_out" if self.input is not None else None

    def forward(self, computed: dict, **kwargs) -> (dict, None):
        return dict(), None

    def sample_from_prior(self, computed: dict, t: float or int = None, **kwargs) -> (dict, None):
        return self.forward(computed)

    def freeze(self, net_name: str):
        for name, param in self.named_parameters():
            if net_name in name:
                param.requires_grad = False

    def set_output(self, output: str) -> None:
        self.output = output

    def serialize(self) -> dict:
        return dict(
            input=self.input,
            output=self.output,
            type=self.__class__
        )


class SimpleBlock(_Block):
    """
    Simple block that takes an input and returns an output
    No sampling is performed
    """

    def __init__(self, net, input_id: str):
        super(SimpleBlock, self).__init__(input_id)
        self.net: Sequential = get_net(net)

    def forward(self, computed: dict, **kwargs) -> (dict, None):
        if self.input not in computed.keys():
            raise ValueError(f"Input {self.input} not found in computed")
        inputs = computed[self.input]
        output = self.net(inputs)
        computed[self.output] = output
        return computed, None

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["net"] = self.net.serialize()
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return SimpleBlock(net=net, input_id=serialized["input"])


class ConcatBlock(_Block):
    """
    Concatenates two inputs along a given dimension
    """

    def __init__(self, inputs: List[str], dimension: int = 1):
        if len(inputs) != 2:
            raise ValueError("ConcatBlock only supports two inputs")
        super().__init__(None)
        self.inputs = inputs
        self.dimension = dimension

    def forward(self, computed: dict, **kwargs) -> (dict, None):
        if not all([inp in computed for inp in self.inputs]):
            raise ValueError("Not all inputs found in computed")
        x, skip = [computed[inp] for inp in self.inputs]
        x_skip = torch.cat([x, skip], dim=self.dimension)
        computed[self.output] = x_skip
        return computed, None

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["inputs"] = self.inputs
        serialized["dimension"] = self.dimension
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        return ConcatBlock(
            inputs=serialized["inputs"],
            dimension=serialized["dimension"]
        )


class DualInputBlock(_Block):
    """
    Takes 2 inputs
    """

    def __init__(self, inputs: List[str], net):
        super().__init__(None)
        assert len(inputs) == 2
        self.inputs = inputs
        self.net = get_net(net)

    def forward(self, computed: dict, **kwargs) -> (dict, None):
        if not all([inp in computed for inp in self.inputs]):
            raise ValueError("Not all inputs found in computed")
        input1, input2 = [computed[inp] for inp in self.inputs]
        output = self.net([input1, input2])
        computed[self.output] = output
        return computed, None

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["inputs"] = self.inputs
        serialized["net"] = self.net.serialize()
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return DualInputBlock(
            inputs=serialized["inputs"],
            net=net
        )


class InputBlock(SimpleBlock):
    """
    Block that takes an input
    and runs it through a preprocessing net if one is given
    """

    def __init__(self, net=None):
        super(InputBlock, self).__init__(net, "input")

    def forward(self, inputs: tensor, **kwargs) -> dict:
        computed = {self.output: inputs} \
            if self.net is None else \
            {self.input: inputs,
             self.output: self.net(inputs)}
        return computed

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return InputBlock(net=net)


class TopSimpleBlock(SimpleBlock):
    """
    Top block of the model
    Constant or trainable prior
    """

    def __init__(self, net,
                 prior_shape: tuple,
                 prior_trainable: bool,
                 prior_data=None):
        super(TopSimpleBlock, self).__init__(input_id='trainable_h', net=net)
        self.prior_shape = prior_shape
        self.prior_trainable = prior_trainable

        if prior_trainable:
            self.trainable_h = torch.nn.Parameter(  # for unconditional generation
                data=prior_data if prior_data is not None else
                torch.empty(size=prior_shape if len(prior_shape) > 1 else (1, *prior_shape)),
                requires_grad=True)
            nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        else:
            # constant tensor with 0 values
            self.trainable_h = torch.nn.Parameter(data=torch.zeros(size=prior_shape), requires_grad=False) \
                if prior_data is None else prior_data

    def forward(self, computed: dict, variate_mask=None, use_mean=False, **kwargs) -> (dict, None):
        x = torch.tile(self.trainable_h, (list(computed.values())[-1].shape[0], 1))
        z = self.net(x)
        computed[self.output] = z
        return computed, None

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["trainable_h"] = self.trainable_h.data
        serialized["prior_shape"] = self.trainable_h.shape
        serialized["prior_trainable"] = self.trainable_h.requires_grad
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return TopSimpleBlock(
            net=net,
            prior_shape=serialized["prior_shape"],
            prior_trainable=serialized["prior_trainable"],
            prior_data=serialized["trainable_h"]
        )


class SimpleGenBlock(_Block):
    """
    Takes an input and samples from a prior distribution
    """

    def __init__(self, net,
                 input_id: str,
                 output_distribution: str = 'normal'):
        super(SimpleGenBlock, self).__init__(input_id)
        self.prior_net: Sequential = get_net(net)
        self.output_distribution: str = output_distribution

    def _sample_uncond(self, y: tensor, t: float or int = None, use_mean=False) -> tensor:
        y_prior = self.prior_net(y)
        pm, pv = split_mu_sigma(y_prior)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        prior = generate_distribution(pm, pv, self.output_distribution)
        z = prior.sample() if not use_mean else prior.mean
        return z, (prior, None)

    def forward(self, computed: dict, use_mean=False, **kwargs) -> (dict, tuple):
        if self.input not in computed.keys():
            raise ValueError(f"Input {self.input} not found in computed")
        x = computed[self.input]
        z, distribution = self._sample_uncond(x, use_mean=use_mean)
        computed[self.output] = z
        return computed, distribution

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = computed[self.input]
        z, dist = self._sample_uncond(x, t, use_mean=use_mean)
        computed[self.output] = z
        return computed, dist

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["prior_net"] = self.prior_net.serialize()
        serialized["output_distribution"] = self.output_distribution
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        return SimpleGenBlock(
            net=prior_net,
            input_id=serialized["input"],
            output_distribution=serialized["output_distribution"]
        )


OutputBlock = SimpleGenBlock


class ContrastiveOutputBlock(OutputBlock):

    # only for 1D inputs

    def __init__(self, net, input_id: str, contrast_dims: int, output_distribution: str = 'normal'):
        super().__init__(net, input_id, output_distribution)
        self.contrast_dims = contrast_dims

    @overrides
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


class GenBlock(SimpleGenBlock):
    """
    Takes an input,
    samples from a prior distribution,
    (takes a condition,
    samples from a posterior distribution),
    and returns the sample
    """

    def __init__(self,
                 prior_net,
                 posterior_net,
                 input_id: str, condition: str,
                 concat_posterior: bool,
                 output_distribution: str = 'normal',
                 input_transform=None,
                 condition_transform=None):
        super(GenBlock, self).__init__(prior_net, input_id, output_distribution)
        self.prior_net: Sequential = get_net(prior_net)
        self.posterior_net: Sequential = get_net(posterior_net)
        self.input_transform: Sequential = get_net(input_transform)
        self.condition_transform = get_net(condition_transform)
        self.concat_posterior = concat_posterior
        self.condition = condition

    def _sample(self, y: tensor, cond: tensor, variate_mask=None, use_mean=False) -> (tensor, tuple):
        y_prior = self.prior_net(y)
        pm, pv = split_mu_sigma(y_prior)
        prior = generate_distribution(pm, pv, self.output_distribution)
        if self.input_transform is not None:
            y = self.input_transform(y)
        if self.condition_transform is not None:
            cond = self.condition_transform(cond)
        posterior_input = torch.cat([cond, y], dim=1) if self.concat_posterior else cond
        y_posterior = self.posterior_net(posterior_input)
        qm, qv = split_mu_sigma(y_posterior)
        posterior = generate_distribution(qm, qv, self.output_distribution)
        z = posterior.rsample() if not use_mean else posterior.mean

        if variate_mask is not None:
            z_prior = prior.rsample() if not use_mean else prior.mean
            z = self.prune(z, z_prior, variate_mask)

        return z, (prior, posterior)

    def _sample_uncond(self, y: tensor, t: float or int = None, use_mean=False) -> tensor:
        y_prior = self.prior_net(y)
        pm, pv = split_mu_sigma(y_prior)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)

        prior = generate_distribution(pm, pv, self.output_distribution)
        z = prior.sample() if not use_mean else prior.mean
        return z, (prior, None)

    def forward(self, computed: dict, variate_mask=None, use_mean=False, **kwargs) -> (dict, tuple):
        if self.input not in computed.keys():
            raise ValueError(f"Input {self.input} not found in computed")
        if self.condition not in computed.keys():
            raise ValueError(f"Condition {self.condition} not found in computed")
        x = computed[self.input]
        cond = computed[self.condition]
        z, distributions = self._sample(x, cond, variate_mask, use_mean=use_mean)
        computed[self.output] = z
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = computed[self.input]
        z, dist = self._sample_uncond(x, t, use_mean=use_mean)
        computed[self.output] = z
        return computed, dist

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
        serialized["input_transform"] = self.input_transform.serialize()
        serialized["condition_transform"] = self.condition_transform.serialize()
        serialized["condition"] = self.condition
        serialized["output_distribution"] = self.output_distribution
        serialized["concat_posterior"] = self.concat_posterior
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        posterior_net = Sequential.deserialize(serialized["posterior_net"])
        input_transform = Sequential.deserialize(serialized["input_transform"])
        condition_transform = Sequential.deserialize(serialized["condition_transform"])
        return GenBlock(
            prior_net=prior_net,
            posterior_net=posterior_net,
            input_transform=input_transform,
            condition_transform=condition_transform,
            input_id=serialized["input"],
            condition=serialized["condition"],
            output_distribution=serialized["output_distribution"],
            concat_posterior=serialized["concat_posterior"]
        )


class TopGenBlock(GenBlock):
    """
    Top block of the model
    Constant or trainable prior
    Posterior is conditioned on the condition
    """

    def __init__(self, net,
                 prior_trainable: bool,
                 condition: str,
                 output_distribution: str = 'normal',
                 concat_posterior: bool = False,
                 prior_data=None,
                 prior_shape: tuple = None):
        super(TopGenBlock, self).__init__(prior_net=None, posterior_net=net,
                                          input_id='trainable_h', condition=condition,
                                          output_distribution=output_distribution,
                                          concat_posterior=concat_posterior)
        self.prior_shape = prior_shape
        self.prior_trainable = prior_trainable

        if prior_trainable:
            self.trainable_h = torch.nn.Parameter(  # for unconditional generation
                data=prior_data if prior_data is not None else
                torch.empty(size=prior_shape if len(prior_shape) > 1 else (1, *prior_shape)),
                requires_grad=True)
            nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        else:
            # constant tensor with 0 values
            self.trainable_h = torch.nn.Parameter(data=torch.concat(
                [
                    torch.zeros(size=prior_shape),
                    torch.ones(size=prior_shape)
                ], 0),
                requires_grad=False) \
                if prior_data is None else prior_data

    def _sample(self, y: tensor, cond: tensor, variate_mask=None, use_mean=False) -> (tensor, tuple):
        y_prior = self.prior_net(y)
        pm, pv = split_mu_sigma(y_prior)
        prior = generate_distribution(pm, pv, self.output_distribution)
        posterior_input = torch.cat([cond, y], dim=1) if self.concat_posterior else cond
        y_posterior = self.posterior_net(posterior_input)
        qm, qv = split_mu_sigma(y_posterior)
        posterior = generate_distribution(qm, qv, self.output_distribution)
        z = posterior.rsample() if not use_mean else posterior.mean

        if variate_mask is not None:
            z_prior = prior.rsample() if not use_mean else prior.mean
            z = self.prune(z, z_prior, variate_mask)

        return z, (prior, posterior)

    def forward(self, computed: dict, variate_mask=None, use_mean=False, **kwargs) -> (tensor, dict, tuple):
        if self.condition not in computed.keys():
            raise ValueError(f"Condition {self.condition} not found in computed")
        cond = computed[self.condition]
        x = torch.tile(self.trainable_h, (cond.shape[0], 1)).to(cond.device)
        if cond.shape != x.shape and self.concat_posterior:
            x = x.resize(cond.shape)
        z, distributions = self._sample(x, cond, use_mean=use_mean)
        computed[self.output] = z
        return computed, distributions

    def sample_from_prior(self, batch_size: int, t: int or float = None, use_mean=False, **kwargs) -> (dict, tuple):
        y = torch.tile(self.trainable_h, (batch_size, 1))
        z, dist = self._sample_uncond(y, t, use_mean=use_mean)
        computed = {
            self.input: y,
            self.output: z}
        return computed, dist

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["trainable_h"] = self.trainable_h.data
        serialized["concat_prior"] = self.concat_posterior
        serialized["prior_shape"] = self.trainable_h.shape
        serialized["prior_trainable"] = self.trainable_h.requires_grad
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["posterior_net"])
        return TopGenBlock(
            net=net,
            prior_shape=serialized["prior_shape"],
            prior_trainable=serialized["prior_trainable"],
            concat_posterior=serialized["concat_prior"],
            condition=serialized["condition"],
            output_distribution=serialized["output_distribution"],
            prior_data=serialized["trainable_h"]
        )


class ResidualGenBlock(GenBlock):
    """
    Architecture from VDVAE paper
    """

    def __init__(self, net,
                 prior_net,
                 posterior_net,
                 z_projection,
                 input, condition,
                 concat_posterior: bool,
                 prior_layer=None,
                 posterior_layer=None,
                 output_distribution: str = 'normal'):
        super(ResidualGenBlock, self).__init__(
            prior_net, posterior_net, input, condition, concat_posterior, output_distribution)
        self.net: Sequential = get_net(net)
        self.z_projection: Sequential = get_net(z_projection)
        self.prior_layer: Sequential = get_net(prior_layer)
        self.posterior_layer: Sequential = get_net(posterior_layer)

    def _sample(self, y: tensor, cond: tensor, variate_mask=None, use_mean=False) -> (tensor, tensor, tuple):

        y_prior = self.prior_net(y)
        kl_residual, y_prior = split_mu_sigma(y_prior, chunks=2)
        y_prior = self.prior_layer(y_prior)
        pm, pv = split_mu_sigma(y_prior)
        prior = generate_distribution(pm, pv, self.output_distribution)

        y_posterior = self.posterior_net(torch.cat([y, cond], dim=1))  # y, cond fordított sorrendben mint máshol
        y_posterior = self.posterior_layer(y_posterior)
        qm, qv = split_mu_sigma(y_posterior)
        posterior = generate_distribution(qm, qv, self.output_distribution)
        z = posterior.rsample() if not use_mean else posterior.mean

        if variate_mask is not None:
            z_prior = prior.rsample() if not use_mean else prior.mean
            z = self.prune(z, z_prior, variate_mask)

        y = y + kl_residual
        return z, y, (prior, posterior)

    def _sample_uncond(self, y: tensor, t: float or int = None, use_mean=False) -> (tensor, tensor):
        y_prior = self.prior_net(y)
        kl_residual, y_prior = split_mu_sigma(y_prior, chunks=2)
        y_prior = self.prior_layer(y_prior)
        pm, pv = split_mu_sigma(y_prior)
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        prior = generate_distribution(pm, pv, self.output_distribution)
        z = prior.sample() if not use_mean else prior.mean
        y = y + kl_residual
        return z, y, (prior, None)

    def forward(self, computed: dict, variate_mask=None, use_mean=False, **kwargs) -> (dict, tuple):
        x = computed[self.input]
        cond = computed[self.condition]
        z, y, distributions = self._sample(x, cond, variate_mask, use_mean)
        y = y + self.z_projection(z)
        y = self.net(y)
        computed[self.output] = y
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = computed[self.input]
        z, y, dist = self._sample_uncond(x, t, use_mean=use_mean)
        y = y + self.z_projection(z)
        y = self.net(y)
        computed[self.output] = y
        return computed, dist

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
        return ResidualGenBlock(
            net=net,
            prior_net=prior_net,
            posterior_net=posterior_net,
            z_projection=z_projection,
            input=serialized["input"],
            condition=serialized["condition"],
            concat_posterior=serialized["concat_posterior"],
            output_distribution=serialized["output_distribution"]
        )
