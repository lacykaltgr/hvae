import copy

from model import train
from hparams import *
from block import DecBlock, EncBlock
from elements.optimizers import get_optimizer
from elements.schedules import get_schedule


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
        self.decoder_blocks = decoder_blocks
        self.out_net = out_net
        self.bias_xs = nn.ParameterList()  # for unconditional generation

    def forward(self, activations, get_latents=False):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        xs, computed = propogate(xs, self.decoder_blocks, "forward", [activations], get_latents=get_latents)
        return xs, stats

    def forward_uncond(self, n, t=None):
        xs = {a.shape[2]: a.repeat(n, 1, 1, 1) for a in self.bias_xs}
        xs = propogate(xs, self.decoder_blocks, "forward_uncond", [t])
        return xs


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

    def reconstruct(self, x):
        latents = self.encoder(x)
        reconstructions, _ = self.decoder(latents)
        return self.decoder.out_net.sample(reconstructions)

    def train_model(self, optimizer, schedule,
                    train_loader, val_loader, checkpoint,
                    writer_train, writer_val, checkpoint_path, local_rank):
        train(self, self.ema_model, optimizer, schedule,
              train_loader, val_loader, checkpoint['global_step'],
              writer_train, writer_val, checkpoint_path, self.device, local_rank)

    def forward(self, x, x_target):
        activations = self.encoder(x)
        px_z, stats = self.decoder(activations)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = (distortion_per_pixel + rate_per_pixel).mean()
        return dict(elbo=elbo, distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean())

    def forward_get_latents(self, x):
        activations = self.encoder(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

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
