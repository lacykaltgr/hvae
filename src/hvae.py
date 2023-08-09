import torch
from torch import nn
from torch import tensor

from src.block import DecBlock, EncBlock, InputBlock, OutputBlock, ConcatBlock
#from src.model import train, reconstruct, generate, compute_per_dimension_divergence_stats, sample, evaluate


class Encoder(nn.Module):
    def __init__(self, encoder_blocks: nn.ModuleList, device: str = "cuda"):
        super(Encoder, self).__init__()
        self.encoder_blocks: nn.ModuleList = encoder_blocks
        self.device = device

    def forward(self, x: tensor) -> (tensor, dict):
        computed = x
        output = None
        for block in self.encoder_blocks:
            output, computed = block(computed)
        return output, computed


class Decoder(nn.Module):
    def __init__(self, decoder_blocks: nn.ModuleList, device: str = "cuda"):
        super(Decoder, self).__init__()
        self._decoder_blocks: nn.ModuleList = decoder_blocks
        self._device = device

    def forward(self, computed: dict, variate_masks: list = None) -> (tensor, dict, list):
        distributions = []
        output = None

        if variate_masks is None:
            variate_masks = [None] * len(self._decoder_blocks)
        assert len(variate_masks) == len(self._decoder_blocks)

        for block, variate_mask in zip(self._decoder_blocks, variate_masks):
            args = dict(computed=computed, variate_mask=variate_mask) \
                if isinstance(block, DecBlock) else dict(computed=computed)
            output, computed, dists = block(**args)
            if dists is not None:
                distributions.append(dists)
        return output, computed, distributions

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        with torch.no_grad():
            for i, block in enumerate(self._decoder_blocks):
                output, computed = block.sample_from_prior(batch_size if i == 0 else computed, temperatures[i])
        return output, computed


class hVAE(nn.Module):
    def __init__(self, blocks: dict, device: str = "cuda"):
        super(hVAE, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for block in blocks:
            blocks[block].set_output(block)
            if isinstance(blocks[block], (InputBlock, EncBlock, ConcatBlock)):
                self.encoder.append(blocks[block])
            elif isinstance(blocks[block], (DecBlock, ConcatBlock, OutputBlock)):
                self.decoder.append(blocks[block])
            else:
                raise ValueError(f"Unknown block type {type(blocks[block])}")

        self.encoder: nn.Module = Encoder(self.encoder, device)
        self.decoder: nn.Module = Decoder(self.decoder, device)
        self.device = device

    def compute(self, block_name) -> (tensor, dict):
        pass

    def summary(self):
        self.to_string()

    #def reconstruct(self, dataset, artifacts_folder=None, latents_folder=None):
    #    return reconstruct(self, dataset, artifacts_folder, latents_folder)

    #def generate(self):
    #    return generate(self)

    #def kldiv_stats(self, dataset):
    #    return compute_per_dimension_divergence_stats(self, dataset)

    #def train_model(self, optimizer, schedule,
    #                train_loader, val_loader, checkpoint,
    #                writer_train, writer_val, checkpoint_path, logger=None):
    #    if logger is None:
    #        from utils import setup_logger
    #        logger = setup_logger(checkpoint_path)
    #    train(self, optimizer, schedule,
    #          train_loader, val_loader, checkpoint['global_step'],
    #          writer_train, writer_val, checkpoint_path, logger)

    #def test_model(self, test_loader):
    #    return evaluate(self, test_loader)

    #def sample(self, logits) -> tensor:
    #    samples = sample(logits)
    #    return samples

    def sample_from_prior(self, batch_size: int, temperatures: list) -> (tensor, dict):
        output, computed = self.decoder.sample_from_prior(batch_size, temperatures)
        return output, computed

    def forward(self, x: tensor, variate_masks=None) -> (tensor, dict, list):
        _, computed = self.encoder(x)
        output, computed, distributions = self.decoder(computed, variate_masks)
        return output, computed, distributions

    #TODO
    def visualize_graph(self) -> None:

        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        encoder_edges = list()
        decoder_edges = list()
        nodes = list()
        position = dict()
        pos = 0
        for _, block in self.encoder.encoder_blocks:
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
        for block in self.decoder.decoder_blocks:
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

    def save(self, path):
        #TODO
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path) -> None:
        #TODO
        self.load_state_dict(torch.load(path))
