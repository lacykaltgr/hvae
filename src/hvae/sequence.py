from src.hvae.hvae import hVAE
from torch import tensor
import torch
from collections import OrderedDict
import torch.distributions as dist


class hSequenceVAE(hVAE):
    def __init__(self, blocks: OrderedDict, init: dict = None):
        super(hSequenceVAE, self).__init__(blocks, init=init)

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        computed, distributions = self.generator.sample_from_prior(batch_size, temperatures)
        computed, output_distribution = self.output_block.sample_from_prior(computed)
        computed['output'] = computed[self.output_block.output]
        distributions['output'] = output_distribution
        return computed, distributions

    def forward(self, x: tensor, variate_masks=None, use_mean=False) -> (dict, dict):
        seq_len = x.shape[1]
        computed = dict()
        distributions = dict()

        for i in range(seq_len):
            observation_computed = self.input_block(x[:, i])
            if i == 0:
                observation_computed = self._init_prior(observation_computed, x.shape[0])
                observation_computed['outputs'] = []
                distributions['outputs'] = []
            computed.update(observation_computed)
            computed, distributions = self.encoder(computed, use_mean=use_mean)
            computed, distributions = self.generator(computed, distributions, variate_masks, use_mean=use_mean)
            computed, output_distribution = self.output_block(computed, use_mean=use_mean)
            computed['outputs'].append(computed[self.output_block.output])
            distributions['outputs'].append(dist.Independent(output_distribution, 1))

            computed = {f'_{key}': value for key, value in computed.items()},
            distributions = {f'_{key}': value for key, value in distributions.items()}

        computed['output'] = torch.stack(list(computed['outputs']), dim=1)
        #distributions['output'] =
        return computed, distributions

    def visualize_graph(self) -> None:
        raise NotImplementedError()

