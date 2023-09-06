import torch
import os
import json
import tensorflow as tf

from migration.ChainVAE_migration.utils.models import VAE_model, nested_model_summary
from src.elements.nets import MLPNet

class ChainVAEMigrationAgent:

    def __init__(self, path):
        self.directory = path
        configs_path = os.path.join(self.directory, "configs.json")
        if os.path.isfile(configs_path):
            print ("Loading experiment configs...")
            with open(configs_path) as configs_json:
                self.config = json.load(configs_json)

        self.model_path = os.path.join(self.directory, "weights")
        self.model_configs=self.config["model_configs"]
        self.model = VAE_model(self.model_configs)
        self.model.load_weights(self.model_path).expect_partial()

        input = tf.ones((1, self.model_configs["q_z1_x_configs"]["input_shape"]))
        self.model(input)
        print(self.model.q_z2_z1_model.weights[3].shape)
        print("cső")

        #nested_model_summary(self.model)

    def get_net(self, config):
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
            output_size = torch.prod(output_shape)
        else:
            output_size = output_shape

        net = MLPNet(
            input_size,
            hidden_layers,
            output_size,
            activation=activation,
            activate_output=False
        )

        with torch.no_grad():
            for layer in net.mlp_layers.modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.copy_(torch.tensor(layers[layer_i]['w']).mT)
                    layer.bias.copy_(torch.tensor(layers[layer_i]['b']))
        return net




    def get_optimizer(self):
        return None

    def get_schedule(self):
        return None








