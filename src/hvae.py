import copy
from torch import nn
from model import train, reconstruct, generate, encode, compute_per_dimension_divergence_stats
import torch
from block import DecBlock, EncBlock, InputBlock, OutputBlock, ConcatBlock


class Encoder(nn.Module):
    def __init__(self, encoder_blocks):
        super(Encoder, self).__init__()
        self.encoder_blocks = encoder_blocks

    def forward(self, x):
        computed = dict()
        output = None
        for block in self.encoder_blocks:
            output, computed = block(computed)
        return output, computed


class Decoder(nn.Module):
    def __init__(self, decoder_blocks, out_net):
        super(Decoder, self).__init__()
        self._decoder_blocks = decoder_blocks
        self.out_net = out_net

    def forward(self, computed):
        kl_divs = []
        output = None
        for block in self._decoder_blocks:
            output, computed, kl_div = block(computed)
            kl_divs.append(kl_div)
        return output, computed, kl_divs

    def sample_from_prior(self, batch_size, temperatures):
        with torch.no_grad():
            for i, block in enumerate(self._decoder_blocks):
                output, computed = block.sample_from_prior(batch_size if i == 0 else computed, temperatures[i])
        return output, computed


class hVAE(nn.Module):
    def __init__(self, blocks, device):
        super(hVAE, self).__init__()

        for block in blocks:
            blocks[block].set_output(block)
        encoder_blocks = filter(lambda block: isinstance(blocks[block], (InputBlock, EncBlock, ConcatBlock)), blocks)
        decoder_blocks = filter(lambda block: isinstance(blocks[block], (DecBlock, ConcatBlock, OutputBlock)), blocks)

        self.encoder = Encoder(encoder_blocks)
        self.decoder = Decoder(decoder_blocks)
        self.ema_model = copy.deepcopy(self)

        self.device = device

    def compute(self, block_name):
        pass

    def reconstruct(self, dataset, artifacts_folder=None, latents_folder=None):
        return reconstruct(self, dataset, artifacts_folder, latents_folder)

    def generate(self):
        return generate(self)

    def encode(self, dataset, latents_folder=None):
        return encode(self, dataset, latents_folder)

    def kldiv_stats(self, dataset):
        return compute_per_dimension_divergence_stats(self, dataset)

    def train_model(self, optimizer, schedule,
                    train_loader, val_loader, checkpoint,
                    writer_train, writer_val, checkpoint_path):
        train(self, optimizer, schedule,
              train_loader, val_loader, checkpoint['global_step'],
              writer_train, writer_val, checkpoint_path)

    def sample(self, logits):
        from model import sample_from_mol
        samples = sample_from_mol(logits)
        return samples

    def sample_from_prior(self, batch_size, temperatures):
        output, computed = self.decoder.sample_from_prior(batch_size, temperatures)
        return output, computed

    def forward(self, x):
        _, computed = self.encoder(x)
        output, computed, kl_divs = self.decoder(computed)
        return output, computed, kl_divs

    def update_ema(self, ema_rate):
        for p1, p2 in zip(self.parameters(), self.ema_model.parameters()):
            # Beta * previous ema weights + (1 - Beta) * current non ema weight
            p2.data.mul_(ema_rate)
            p2.data.add_(p1.data * (1 - ema_rate))

    def visualize_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        encoder_edges = list()
        decoder_edges = list()
        nodes = list()
        position = dict()
        pos = 0
        for block in self.encoder.encoder_blocks:
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
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path) -> None:
        self.load_state_dict(torch.load(path))
