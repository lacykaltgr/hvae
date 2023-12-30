import logging

import torch
from torch import nn
from torch import tensor
from collections import OrderedDict
from src.utils import OrderedModuleDict
from src.utils import handle_shared_modules

from src.hvae.block import GenBlock, InputBlock, OutputBlock
from src.hvae.model import train, reconstruct, generate, evaluate
from src.hvae.analysis_tools import model_summary, compute_per_dimension_divergence_stats


class Encoder(nn.Module):
    def __init__(self, encoder_blocks: OrderedModuleDict):
        super(Encoder, self).__init__()
        self.blocks: OrderedModuleDict = encoder_blocks

    def forward(self, computed: tensor, distributions: dict, to_compute: str = None, use_mean: bool = False) -> (tensor, dict):
        for block in self.blocks.values():
            output = block(computed, use_mean=use_mean)
            computed, dists = output
            if dists:
                distributions[block.output] = dists
            if to_compute is not None and to_compute in computed:
                return computed, distributions
        return computed, distributions


class Generator(nn.Module):
    def __init__(self, blocks: OrderedModuleDict):
        super(Generator, self).__init__()
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


class hVAE(nn.Module):
    def __init__(self, blocks: OrderedDict, init: dict = None):
        super(hVAE, self).__init__()

        encoder_blocks = OrderedModuleDict()
        generator_blocks = OrderedModuleDict()

        in_generator = False
        for i, (output, block) in enumerate(blocks.items()):
            block.set_output(output)
            if i == 0:
                assert isinstance(block, InputBlock)
                self.input_block = block
                continue
            if i == len(blocks) - 1:
                assert isinstance(block, OutputBlock)
                self.output_block = block
                continue
            if isinstance(block, GenBlock):
                in_generator = True
            if in_generator:
                generator_blocks.update({output: block})
            else:
                encoder_blocks.update({output: block})

        self.encoder: Encoder = Encoder(encoder_blocks)
        self.generator: Generator = Generator(generator_blocks)
        self.prior = init

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

    def summary(self):
        return model_summary(self)

    def reconstruct(self, dataset, artifacts_folder=None, latents_folder=None):
        return reconstruct(self, dataset, artifacts_folder, latents_folder)

    def generate(self):
        logger = logging.Logger("generate")
        return generate(self, logger=logger)

    def kldiv_stats(self, dataset):
        return compute_per_dimension_divergence_stats(self, dataset)

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

    def train_model(self, optimizer, schedule,
                    train_loader, val_loader, checkpoint,
                    writer_train, writer_val, checkpoint_path, logger=None):
        if logger is None:
            from src.utils import setup_logger
            logger = setup_logger(checkpoint_path)
        train(self, optimizer, schedule,
              train_loader, val_loader, checkpoint['global_step'],
              writer_train, writer_val, checkpoint_path, logger)

    def test_model(self, test_loader):
        return evaluate(self, test_loader)

    def _init_prior(self, computed, batch_size) -> dict:
        for key, value in self.prior.items():
            dims = [1] * len(value.shape)
            batched_prior = torch.tile(value, (batch_size, *dims))
            computed[key] = batched_prior
        return computed

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        computed, distributions = self.generator.sample_from_prior(batch_size, temperatures)
        computed, output_distribution = self.output_block.sample_from_prior(computed)
        computed['output'] = computed[self.output_block.output]
        distributions['output'] = output_distribution
        return computed, distributions

    def forward(self, x: tensor, variate_masks=None, use_mean=False) -> (dict, dict):
        computed = self.input_block(x)
        computed = self._init_prior(computed, x.shape[0])
        distributions = dict()
        computed, distributions = self.encoder(computed, distributions, use_mean=use_mean)
        computed, distributions = self.generator(computed, distributions, variate_masks, use_mean=use_mean)
        computed, output_distribution = self.output_block(computed, use_mean=use_mean)
        computed['output'] = computed[self.output_block.output]
        distributions['output'] = output_distribution
        return computed, distributions

    # TODO
    def visualize_graph(self) -> None:

        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        encoder_edges = list()
        decoder_edges = list()
        nodes = list()
        position = dict()
        pos = 0
        for _, block in self.encoder.blocks:
            if isinstance(block.input, (list, tuple)):
                for inp in block.input:
                    encoder_edges.append((inp, block.output))
                    if inp not in nodes:
                        nodes.append(inp)
                        position[inp] = (0, pos)
                        pos = pos + 1
            else:
                encoder_edges.append((block.input, block.output))
                if block.input not in nodes:
                    nodes.append(block.input)
                    position[inp] = (0, pos)
                    pos = pos + 1
        for block in self.generator.decoder_blocks:
            if isinstance(block.input, (list, tuple)):
                for inp in block.input:
                    decoder_edges.append((inp, block.output))
            else:
                decoder_edges.append((block.input, block.output))

        nx.draw(G, nodelist=nodes, pos=pos, edgelist=encoder_edges, edge_color="r", width=2, node_size=2000,
                with_labels=True, node_color="lightblue", ax=plt.gca(), arrowstyle="->", connectionstyle="arc3,rad=0.2")
        nx.draw(G, nodelist=nodes, pos=pos, edgelist=decoder_edges, edge_color="b", width=2, node_size=2000,
                with_labels=True, node_color="lightblue", ax=plt.gca(), arrowstyle="->")
        plt.show()

    def serialize(self):
        blocks = list()
        blocks.append(self.input_block.serialize())
        for block in self.encoder.blocks.values():
            blocks.append(block.serialize())
        for block in self.generator.blocks.values():
            blocks.append(block.serialize())
        blocks.append(self.output_block.serialize())
        serialized = dict(
            blocks=blocks,
            prior=self.prior
        )
        return serialized

    @staticmethod
    def deserialize(serialized):
        blocks = OrderedDict()
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
