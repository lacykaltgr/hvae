import torch
import os
import json
import tensorflow as tf

from migration.ChainVAE_migration.utils.models import VAE_model, shared_NN
from hvae_backbone.elements.nets import MLPNet

class ChainVAEMigrationAgent:

    def __init__(self, path):
        self.directory = path
        configs_path = os.path.join(self.directory, "configs.json")
        if os.path.isfile(configs_path):
            print("Loading experiment configs...")
            with open(configs_path) as configs_json:
                self.config = json.load(configs_json)

        self.model_path = os.path.join(self.directory, "weights")
        self.model_configs=self.config["model_configs"]
        self.model = VAE_model(self.model_configs)
        self.model.load_weights(self.model_path).expect_partial()

        input = tf.ones((1, self.model_configs["q_z1_x_configs"]["input_shape"]))
        self.model(input)

    def get_net(self, net_name):
        config = self.model_configs[net_name + "_configs"]
        neurons = config["neurons"]
        n_hidden_layers = config["hidden_layers"]
        activation = config["activation"]
        distribution = config["distribution"]
        input_size = config["input_shape"]
        output_shape = config["output_shape"]

        if isinstance(neurons, list):
            if len(neurons) != n_hidden_layers:
                raise ValueError("Number of layers is not consistent: hidden_layers != len(neurons)")
            hidden_layers = neurons
        else:
            hidden_layers = [neurons] * n_hidden_layers

        if distribution == "normal" or distribution == "laplace":
            output_size = 2 * output_shape
        elif distribution == "observation_normal":
            output_size = torch.prod(torch.tensor(output_shape))
        else:
            output_size = output_shape

        if activation == "softplus":
            activation = torch.nn.Softplus()

        net: shared_NN = self.model.__getattribute__(net_name + "_model")

        migrated_net = MLPNet(
            input_size,
            hidden_layers,
            output_size,
            activation=activation,
            activate_output=False
        )

        weight_i = 2
        with torch.no_grad():
            for layer in migrated_net.mlp_layers.modules():
                if isinstance(layer, torch.nn.Linear):
                    if weight_i == len(net.weights)-1:
                        weight_i = 0
                    layer.weight.copy_(torch.tensor(net.weights[weight_i].numpy()).mT)
                    layer.bias.copy_(torch.tensor(net.weights[weight_i+1].numpy()))
                    weight_i += 2
        return migrated_net

    def get_optimizer(self):
        return None

    def get_schedule(self):
        return None

    def get_global_step(self):
        return -1








