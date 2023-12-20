from src.hvae.hvae import hVAE

from torch import tensor
from torch import nn
from collections import OrderedDict
from src.utils import OrderedModuleDict


class SequenceEncoder(nn.Module):
    def __init__(self, encoder_blocks: OrderedModuleDict):
        super(SequenceEncoder, self).__init__()
        self.blocks: OrderedModuleDict = encoder_blocks

    def forward(self, x: tensor, to_compute: str = None, use_mean: bool = False) -> (tensor, dict):
        _computed = x  # (batch_size, seq_len, c, h, w)
        # initialize states

        seq_len = x.shape[1]
        distributions = []
        for i in range(seq_len):
            distributions.append(dict())
            for block in self.blocks.values():
                computed, dists = block(_computed, use_mean=use_mean)
                _computed = {**{f'_{key}': value for key, value in _computed.items()},
                             **{f'_{key}': value for key, value in computed.items()}}
                if dists:
                    distributions[i][block.output] = dists
                if to_compute is not None and to_compute in computed:
                    return computed, distributions
        return computed, distributions


class SequenceGenerator(nn.Module):
    def __init__(self, blocks: OrderedModuleDict):
        super(SequenceGenerator, self).__init__()
        self.blocks: OrderedModuleDict = blocks

    def forward(self, computed: dict, distributions: dict, variate_masks: list = None,
                to_compute: str = None, use_mean: bool = False) -> (dict, dict):
        if variate_masks is None:
            variate_masks = [None] * len(self.blocks)
        assert len(variate_masks) == len(self.blocks)

        for block, variate_mask in zip(self.blocks.values(), variate_masks):
            if block.output in computed.keys():
                continue
            output = block(computed, use_mean=use_mean, variate_mask=variate_mask)
            computed, dists = output
            if dists:
                distributions[block.output] = dists
            if to_compute is not None and to_compute in computed:
                return computed, distributions
        return computed, distributions

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        distributions = dict()
        with torch.no_grad():
            for i, block in enumerate(self.blocks.values()):
                computed, dist = block.sample_from_prior(batch_size if i == 0 else computed, temperatures[i])
                distributions[block.output] = dist
        return computed, distributions


class hSequenceVAE(hVAE):
    def __init__(self, blocks: OrderedDict):
        super(hSequenceVAE, self).__init__(blocks)

    def compute_function(self, block_name='output') -> callable:
        if block_name == 'output':
            block_name = self.output_block.output

        def compute(x: tensor or dict, use_mean=False) -> (dict, dict):
            if isinstance(x, dict):
                computed = x
            else:
                computed = self.input_block(x)
            computed, distributions = self.encoder(computed, to_compute=block_name, use_mean=use_mean)
            if block_name in computed.keys():
                return computed, distributions
            computed, distributions = self.generator(computed, distributions, to_compute=block_name, use_mean=use_mean)
            if block_name in computed.keys():
                return computed, distributions
            computed, distribution = self.output_block(computed, use_mean=use_mean)
            computed['output'] = computed[self.output_block.output]
            distributions['output'] = distribution
            return computed, distributions
        return compute

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        computed, distributions = self.generator.sample_from_prior(batch_size, temperatures)
        computed, output_distribution = self.output_block.sample_from_prior(computed)
        computed['output'] = computed[self.output_block.output]
        distributions['output'] = output_distribution
        return computed, distributions

    def forward(self, x: tensor, variate_masks=None, use_mean=False) -> (dict, dict):
        computed = self.input_block(x)
        computed, distributions = self.encoder(computed, use_mean=use_mean)
        computed, distributions = self.generator(computed, distributions, variate_masks, use_mean=use_mean)
        computed, output_distribution = self.output_block(computed, use_mean=use_mean)
        computed['output'] = computed[self.output_block.output]
        distributions['output'] = output_distribution
        return computed, distributions

    def visualize_graph(self) -> None:
        raise NotImplementedError()

