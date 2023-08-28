import tensorflow as tf
import torch
from src.elements.nets import MLPNet
from checkpoint import Checkpoint


class TDVAEMigrationAgent:
    def __init__(self, path):
        checkpoint = tf.train.load_checkpoint(path)
        variables = tf.train.list_variables(path)
        modules = dict()
        for v in variables:
            name, shape = v
            path_list = name.split('/')
            if len(path_list) < 4:
                if name == 'global_step':
                    self.global_step = checkpoint.get_tensor(name)
                elif name == 'beta1_power':
                    self.beta1_power = checkpoint.get_tensor(name)
                elif name == 'beta2_power':
                    self.beta2_power = checkpoint.get_tensor(name)
            module_name = path_list[-3]
            module_type = path_list[-3].split('_')[0]
            layer_name = path_list[-2]
            param_type = path_list[-1]
            if module_name not in modules.keys():
                modules[module_name] = dict()
            modules[module_name]['type'] = module_type
            if layer_name not in modules[module_name].keys():
                modules[module_name][layer_name] = dict()
            modules[module_name][layer_name][param_type] = checkpoint.get_tensor(name)
            modules[module_name][layer_name]['shape'] = shape
        self.modules = modules

    def __getitem__(self, item):
        module = self.modules[item]

        if module['type'] != 'mlp':
            raise NotImplementedError(f'Module type {module["type"]} not implemented.')

        input_size = None
        hidden_sizes = []
        output_size = None
        for layer_name, layer in module.items():
            for param_type, param in layer.items():
                if param_type == 'w':
                    if input_size is None:
                        input_size = param['shape'][0]
                    else:
                        hidden_sizes.append(param['shape'][0])
                    output_size = param['shape'][1]

        net = MLPNet(input_size, hidden_sizes, output_size)

        layers = list(module.values())
        layer_i = 0
        with torch.no_grad():
            for layer in net.mlp_layers.modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.copy_(torch.tensor(layers[layer_i]['w']))
                    layer.bias.copy_(torch.tensor(layers[layer_i]['b']))
                    layer_i += 1
        return net

    def get_global_step(self):
        return self.global_step

    def get_optimizer(self, optimizer):
        return self.optimizer

    def get_schedule(self, schedule):
        return self.schedule








