import numpy as np
import torch
from overrides import overrides
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

    def __init__(self, input_id: str or list or tuple = None):
        super(_Block, self).__init__()
        self.input = InputPipeline(input_id)
        self.output = None

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
            input=self.input.serialize(),
            output=self.output,
            type=self.__class__
        )


class InputPipeline(SerializableModule):
    """
    Helper class for preprocessing pipeline
    """
    def __init__(self, input_pipeline: str or tuple or list):
        super(InputPipeline, self).__init__()
        self.inputs = input_pipeline

    def forward(self, computed):
        return self._load(computed, self.inputs)

    def serialize(self):
        return self._serialize(self.inputs)

    def _serialize(self, item):
        if isinstance(item, str):
            return item
        elif isinstance(item, list):
            return [i.serialize() if isinstance(i, SerializableModule)
                    else self._serialize(i) for i in item]
        elif isinstance(item, tuple):
            return tuple([self._serialize(i) for i in item])

    @staticmethod
    def deserialize(serialized):
        if isinstance(serialized, str):
            return serialized
        elif isinstance(serialized, list):
            return [i["type"].deserialize(i) if isinstance(i, dict) and "type" in i.keys()
                    else InputPipeline.deserialize(i) for i in serialized]
        elif isinstance(serialized, tuple):
            return tuple([InputPipeline.deserialize(i) for i in serialized])

    @staticmethod
    def _load(computed: dict, inputs):
        def _validate_get(_inputs):
            if not isinstance(_inputs, str):
                raise ValueError(f"Input {_inputs} must be a string")
            if _inputs not in computed.keys():
                raise ValueError(f"Input {_inputs} not found in computed")
            return computed[_inputs]

        # single input
        if isinstance(inputs, str):
            return _validate_get(inputs)

        # multiple inputs
        elif isinstance(inputs, tuple):
            return tuple([InputPipeline._load(computed, i) for i in inputs])

        # list of operations
        elif isinstance(inputs, list):
            if len(inputs) < 2:
                raise ValueError(f"Preprocessing pipeline must have at least 2 elements, got {len(inputs)}"
                                 f"Provide the inputs in [inputs, operation1, operation2, ...] format")
            if not isinstance(inputs[0], (str, tuple)):
                raise ValueError(f"First element of the preprocessing pipeline "
                                 f"must be the input id or tuple of input ids, got {inputs[0]}")
            input_tensors = InputPipeline._load(computed, inputs[0])
            for op in inputs[1:]:
                if callable(op):
                    input_tensors = op(input_tensors)
                elif isinstance(op, str):
                    if op == "concat":
                        input_tensors = torch.cat(input_tensors, dim=1)
                    elif op == "substract":
                        input_tensors = input_tensors[0] - input_tensors[1]
                    elif op == "add":
                        input_tensors = input_tensors[0] + input_tensors[1]
            return input_tensors


class SimpleBlock(_Block):
    """
    Simple block that takes an input and returns an output
    No sampling is performed
    """

    def __init__(self, net, input_id):
        super(SimpleBlock, self).__init__(input_id)
        self.net: Sequential = get_net(net)

    def forward(self, computed: dict, **kwargs) -> (dict, None):
        inputs = self.input(computed)
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
        return SimpleBlock(net=net, input_id=InputPipeline.deserialize(serialized["input"]))


class InputBlock(SimpleBlock):
    """
    Block that takes an input
    and runs it through a preprocessing net if one is given
    """

    def __init__(self, net=None):
        super(InputBlock, self).__init__(net, "input")

    def forward(self, inputs: dict, **kwargs) -> tuple:
        if isinstance(inputs, dict):
            computed = inputs
        elif isinstance(inputs, torch.Tensor):
            computed = {self.output: self.net(inputs)}
        else:
            raise ValueError(f"Input must be a tensor or a dict got {type(inputs)}")
        distributions = dict()
        return computed, distributions

    @staticmethod
    def deserialize(serialized: dict):
        net = Sequential.deserialize(serialized["net"])
        return InputBlock(net=net)


class SimpleGenBlock(_Block):
    """
    Takes an input and samples from a prior distribution
    """

    def __init__(self, net, input_id, output_distribution: str = 'normal'):
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
        x = self.input(computed)
        z, distribution = self._sample_uncond(x, use_mean=use_mean)
        computed[self.output] = z
        return computed, distribution

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
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
            input_id=InputPipeline.deserialize(serialized["input"]),
            output_distribution=serialized["output_distribution"]
        )


class OutputBlock(SimpleGenBlock):
    def __init__(self, net, input_id, output_distribution: str = 'normal'):
        super(OutputBlock, self).__init__(net, input_id, output_distribution)

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        return OutputBlock(
            net=prior_net,
            input_id=InputPipeline.deserialize(serialized["input"]),
            output_distribution=serialized["output_distribution"]
        )


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
                 input_id, condition,
                 output_distribution: str = 'normal',
                 fuse_prior: str = None
                ):
        super(GenBlock, self).__init__(prior_net, input_id, output_distribution)
        self.prior_net: Sequential = get_net(prior_net)
        self.posterior_net: Sequential = get_net(posterior_net)
        self.condition = InputPipeline(condition)
        self.fuse_prior = fuse_prior

    def _sample(self, y: tensor, cond: tensor, variate_mask=None, use_mean=False) -> (tensor, tuple):
        y_prior = self.prior_net(y)
        pm, pv = split_mu_sigma(y_prior)
        prior = generate_distribution(pm, pv, self.output_distribution)

        if self.fuse_prior is not None:
            cond = self.fuse(y_prior, cond, self.fuse_prior)

        y_posterior = self.posterior_net(cond)
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
        x = self.input(computed)
        cond = self.condition(computed)
        z, distributions = self._sample(x, cond, variate_mask, use_mean=use_mean)
        computed[self.output] = z
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        z, dist = self._sample_uncond(x, t, use_mean=use_mean)
        computed[self.output] = z
        return computed, dist

    @staticmethod
    def prune(z, z_prior, variate_mask=None):
        variate_mask = torch.Tensor(variate_mask)
        # Only used in inference mode to prune turned-off variates
        # Use posterior sample from meaningful variates, and prior sample from "turned-off" variates
        # The NLL should be similar to using z_post without masking if the mask is good (not very destructive)
        # variate_mask automatically broadcasts to [batch_size, H, W, n_variates]
        z = variate_mask * z + (1. - variate_mask) * z_prior
        return z

    @staticmethod
    def fuse(prior, cond, method):
        if method == "concat":
            return torch.cat([cond, prior], dim=1)
        elif method == "add":
            return cond + prior
        elif method == "substract":
            return cond - prior
        #elif callable(method):
        #    return method(cond, prior)
        else:
            raise ValueError(f"Unknown method {method} for fusing prior and condition")

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["prior_net"] = self.prior_net.serialize()
        serialized["posterior_net"] = self.posterior_net.serialize()
        serialized["condition"] = self.condition.serialize()
        serialized["output_distribution"] = self.output_distribution
        serialized["fuse_prior"] = self.fuse_prior
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = Sequential.deserialize(serialized["prior_net"])
        posterior_net = Sequential.deserialize(serialized["posterior_net"])
        return GenBlock(
            prior_net=prior_net,
            posterior_net=posterior_net,
            input_id=InputPipeline.deserialize(serialized["input"]),
            condition=InputPipeline.deserialize(serialized["condition"]),
            output_distribution=serialized["output_distribution"],
            fuse_prior=serialized["fuse_prior"]
        )


"""
------------------------
CUSTOM BLOCKS
------------------------
"""


class ContrastiveOutputBlock(OutputBlock):

    # only for 1D inputs
    def __init__(self, net, input_id, contrast_dims: int, output_distribution: str = 'normal'):
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


class ResidualGenBlock(GenBlock):
    """
    Architecture from VDVAE paper
    """

    def __init__(self, net,
                 prior_net,
                 posterior_net,
                 z_projection,
                 input_id, condition,
                 concat_posterior: bool,
                 prior_layer=None,
                 posterior_layer=None,
                 output_distribution: str = 'normal'):
        super(ResidualGenBlock, self).__init__(
            prior_net, posterior_net, input_id, condition, concat_posterior, output_distribution)
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
        x = self.input(computed)
        cond = self.condition(computed)
        z, y, distributions = self._sample(x, cond, variate_mask, use_mean)
        y = y + self.z_projection(z)
        y = self.net(y)
        computed[self.output] = y
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
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
            input_id=InputPipeline.deserialize(serialized["input_id"]),
            condition=InputPipeline.deserialize(serialized["condition"]),
            concat_posterior=serialized["concat_posterior"],
            output_distribution=serialized["output_distribution"]
        )
