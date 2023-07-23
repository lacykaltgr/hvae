import copy
from torch import nn
from model import train, reconstruct, generate, encode, compute_per_dimension_divergence_stats
import torch
from block import DecBlock, EncBlock


def propogate(x, blocks, method_name, params, **kwparams):
    computed = dict(x=x)
    output = None
    while len(blocks) > 0:
        for block in blocks:
            inputs = block.inputs.split(",")
            outputs = block.outputs.split(",")
            if all([inp in computed for inp in inputs]):
                method = getattr(block, method_name)
                input_list = [computed[inp] for inp in inputs]
                output = method(*input_list, *params, **kwparams)
                for outp, outp_value in zip(outputs, output):
                    computed[outp] = output
                blocks.remove(block)
                break
    return output, computed


class Encoder(nn.Module):
    def __init__(self, encoder_blocks):
        super(Encoder, self).__init__()
        self.encoder_blocks = encoder_blocks

    def forward(self, x):
        return propogate(x, self.encoder_blocks, "forward")


class Decoder(nn.Module):
    def __init__(self, decoder_blocks, out_net):
        super(Decoder, self).__init__()
        self._decoder_blocks = decoder_blocks
        self.out_net = out_net
        self.bias_xs = nn.ParameterList()  # for unconditional generation

    def forward(self, activations, get_latents=False):
        xs = torch.tile(self.trainable_h, (activations[0].size()[0], 1, 1, 1))
        result, _ = propogate(xs, self._decoder_blocks, "forward", [activations], get_latents=get_latents)
        xs, stats = result
        return xs, stats

    def sample_from_prior(self, batch_size, temperatures):
        with torch.no_grad():
            y = torch.tile(self.trainable_h, (batch_size, 1, 1, 1))
            y, _ = propogate(y, self._decoder_blocks, "sample_from_prior", [temperatures])
        return y


class hVAE(nn.Module):
    def __init__(self, blocks, name, device, **custom_ops):
        super(hVAE, self).__init__(name=name)

        encoder_blocks = list()
        decoder_blocks = list()
        for block in blocks:
            if isinstance(block, EncBlock):
                encoder_blocks.append(block)
            elif isinstance(block, DecBlock):
                decoder_blocks.append(block)
            else:
                raise ValueError(f'Unknown block type {type(block)}')
        self.encoder = Encoder(encoder_blocks)
        self.decoder = Decoder(decoder_blocks)
        self.ema_model = copy.deepcopy(self)

        self.device = device

        for operation in custom_ops:
            setattr(self, operation,
                    lambda x: custom_ops[operation](x, None, blocks)) \
                if callable(custom_ops[operation]) else None

    def reconstruct(self, dataset,artifacts_folder=None, latents_folder=None):
        return reconstruct(dataset, self, artifacts_folder, latents_folder)

    def generate(self):
        return generate(self)

    def encode(self, dataset, latents_folder=None):
        return encode(dataset, self, latents_folder)

    def kldiv_stats(self, dataset):
        return compute_per_dimension_divergence_stats(dataset, self)

    def train_model(self, optimizer, schedule,
                    train_loader, val_loader, checkpoint,
                    writer_train, writer_val, checkpoint_path, local_rank):
        train(self, self.ema_model, optimizer, schedule,
              train_loader, val_loader, checkpoint['global_step'],
              writer_train, writer_val, checkpoint_path, self.device, local_rank)

    def sample(self, logits):
        from model import sample_from_mol
        return sample_from_mol(logits)

    def sample_from_prior(self, batch_size, temperatures):
        return self.decoder.sample_from_prior(batch_size, temperatures)

    def forward(self, x):
        activations = self.encoder(x)
        px_z, stats = self.decoder(activations)
        return px_z, stats

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
