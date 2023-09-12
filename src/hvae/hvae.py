import logging

import torch
from torch import nn
from torch import tensor
from collections import OrderedDict

from src.hvae.block import GenBlock, InputBlock, OutputBlock, TopSimpleBlock, SimpleBlock, TopGenBlock
from src.hvae.model import train, reconstruct, generate, compute_per_dimension_divergence_stats, evaluate, model_summary


class Encoder(nn.Module):
    def __init__(self, encoder_blocks: nn.ModuleDict):
        super(Encoder, self).__init__()
        self.blocks: nn.ModuleDict = encoder_blocks

    def forward(self, x: tensor, to_compute: str = None) -> (tensor, dict):
        computed = x
        distributions = []
        for block in self.blocks.values():
            output = block(computed)
            if isinstance(output, tuple):
                computed, dists = output
                distributions.append(dists)
            else:
                computed = output
            if to_compute is not None and to_compute in computed:
                return computed
        return computed, distributions


class Generator(nn.Module):
    def __init__(self, blocks: nn.ModuleDict):
        super(Generator, self).__init__()
        self.blocks: nn.ModuleDict = blocks

    def forward(self, computed: dict, distributions: list, variate_masks: list = None, to_compute: str = None) \
            -> (tensor, dict, list):

        if variate_masks is None:
            variate_masks = [None] * len(self.blocks)
        assert len(variate_masks) == len(self.blocks)

        for block, variate_mask in zip(self.blocks.values(), variate_masks):
            if block.output in computed.keys():
                continue
            args = dict(computed=computed, variate_mask=variate_mask) \
                if isinstance(block, GenBlock) else dict(computed=computed)
            output = block(**args)
            if isinstance(output, tuple):
                computed, dists = output
                distributions.append(dists)
            else:
                computed = output
            if to_compute is not None and to_compute in computed:
                return computed, distributions
        return computed, distributions

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        with torch.no_grad():
            for i, block in enumerate(self.blocks.values()):
                if not isinstance(block, SimpleBlock):
                    computed = block.sample_from_prior(batch_size if i == 0 else computed, temperatures[i])
                else:
                    computed = block(computed)
        return computed


class hVAE(nn.Module):
    def __init__(self, blocks: OrderedDict):
        super(hVAE, self).__init__()

        self.input_block, output = next(((block, output) for output, block in blocks.items()
                                         if isinstance(block, InputBlock)), None)
        self.input_block.set_output(output)
        encoder_blocks = nn.ModuleDict()
        generator_blocks = nn.ModuleDict()
        self.output_block, output = next(((block, output) for output, block in blocks.items()
                                          if isinstance(block, OutputBlock)), None)
        self.output_block.set_output(output)

        in_generator = False
        for output, block in blocks.items():
            block.set_output(output)
            if isinstance(block, (TopGenBlock, TopSimpleBlock)):
                in_generator = True
            if not isinstance(block, (InputBlock, OutputBlock)):
                if in_generator:
                    generator_blocks.update({output: block})
                else:
                    encoder_blocks.update({output: block})

        self.encoder: Encoder = Encoder(encoder_blocks)
        self.generator: Generator = Generator(generator_blocks)

    def compute_function(self, block_name) -> (tensor, dict):
        def compute(x: tensor or dict) -> (tensor, dict):
            if isinstance(x, dict):
                computed = x
            else:
                computed = self.input_block(x)
            computed, distributions = self.encoder(computed, to_compute=block_name)
            if block_name in computed.keys():
                return computed[block_name]
            computed, _ = self.generator(computed, distributions, to_compute=block_name)
            if block_name in computed.keys():
                return computed[block_name]
            output_sample, computed, _ = self.output_block(computed)
            return output_sample

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

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        computed = self.generator.sample_from_prior(batch_size, temperatures)
        output_sample, computed = self.output_block.sample_from_prior(computed)
        return output_sample, computed

    def forward(self, x: tensor, variate_masks=None) -> (tensor, dict, list):
        computed = self.input_block(x)
        computed, distributions = self.encoder(computed)
        computed, distributions = self.generator(computed, distributions, variate_masks)
        output_sample, computed, output_distribution = self.output_block(computed)
        distributions.append(output_distribution)
        return output_sample, computed, distributions

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
        return blocks

    @staticmethod
    def deserialize(serialized_blocks):
        blocks = OrderedDict()
        for block in serialized_blocks:
            blocks[block["output"]] = block["type"].deserialize(block)
        return hVAE(blocks)

    @staticmethod
    def load(path):
        from src.checkpoint import Checkpoint
        checkpoint = Checkpoint.load(path)
        return checkpoint.get_model()
