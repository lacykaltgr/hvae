import torch
from torch import nn
from torch import tensor

from src.utils import OrderedModuleDict
from src.utils import handle_shared_modules
from src.hvae.block import GenBlock, InputBlock, OutputBlock
from src.hvae.analysis_tools import model_summary


class hVAE(nn.Module):
    def __init__(self, blocks: OrderedModuleDict, init: dict = None):
        super(hVAE, self).__init__()

        assert isinstance(blocks[0], InputBlock), "First block must be an InputBlock"
        assert isinstance(blocks[-1], OutputBlock), "Last block must be an OutputBlock"

        for key, block in zip(blocks.keys(), blocks.values()):
            block.set_output(key)

        self.blocks = blocks
        self.prior = init

    def forward(self, inputs, stop_at=None, use_mean: bool = False) -> (dict, dict):
        computed, distributions = None, None
        for i, block in enumerate(self.blocks.values()):
            if i == 0:
                computed, distributions = block(inputs, use_mean=use_mean)
                batch_size = computed[self.blocks[0].output].shape[0]
                computed = self._init_prior(computed, batch_size)
                continue
            if block.output in computed.keys():
                continue

            computed, dists = block(computed, use_mean=use_mean)
            if dists is not None:
                distributions[block.output] = dists
            if stop_at is not None and stop_at in computed.keys():
                return computed, distributions

        computed['output'] = computed[self.blocks[-1].output]
        distributions['output'] = distributions.pop(self.blocks[-1].output)[0]
        return computed, distributions

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        computed = self._init_prior(dict(), batch_size)
        distributions = dict()
        in_generator = False
        with torch.no_grad():
            for i, block in enumerate(self.blocks.values()):
                if isinstance(block, GenBlock):
                    in_generator = True
                if in_generator:
                    computed, dist = block.sample_from_prior(computed, temperatures[i])
                    if dist:
                        distributions[block.output] = dist
        return computed, distributions

    def _init_prior(self, computed, batch_size) -> dict:
        for key, value in self.prior.items():
            dims = [1] * len(value.shape)
            batched_prior = torch.tile(value, (batch_size, *dims))
            computed[key] = batched_prior
        return computed

    def summary(self):
        return model_summary(self)

    def freeze(self, nets: list):
        for net in nets:
            assert len(net) == 2
            block_name, net_name = net

            if block_name == "encoder":
                assert net_name == "*"
                for block in self.encoder.blocks.values():
                    block.freeze("*")
            elif block_name == "generator" or block_name == "decoder":
                assert net_name == "*"
                for block in self.generator.blocks.values():
                    block.freeze("*")
            elif block_name in self.encoder.blocks.keys():
                self.encoder.blocks[block_name].freeze(net_name)
            elif block_name in self.generator.blocks.keys():
                self.generator.blocks[block_name].freeze(net_name)
            else:
                raise ValueError(f"Unknown net {block_name} {net_name}")

    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True

    # TODO
    def visualize_graph(self) -> None:
        raise NotImplementedError()

    def serialize(self):
        blocks = list()
        for block in self.blocks.values():
            serialized = block.serialize()
            blocks.append(serialized)
        serialized = dict(
            blocks=blocks,
            prior=self.prior
        )
        return serialized

    @staticmethod
    def deserialize(serialized):
        blocks = OrderedModuleDict()
        shared = dict()
        for block in serialized["blocks"]:
            deserialized = block["type"].deserialize(block)
            deserialized, shared = handle_shared_modules(deserialized, shared)
            blocks[block["output"]] = deserialized
        return hVAE(blocks, serialized["prior"])

    @staticmethod
    def load(path):
        from src.checkpoint import Checkpoint
        checkpoint = Checkpoint.load(path)
        return checkpoint.get_model()
