import os

import numpy as np
from sklearn.utils import shuffle

from hparams import *
from src.elements.dataset import _DataSet


class ImageNetDataSet(_DataSet):
    def __init__(self):
        super(ImageNetDataSet, self).__init__()

    def load(self):
        train_data = self.read_imagenet_images(data_params.train_data_path)
        val_data = self.read_imagenet_images(data_params.val_data_path)
        test_data = self.read_imagenet_images(data_params.test_data_path)
        return train_data, val_data, test_data

    def read_imagenet_images(self, path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
        data = np.concatenate([np.load(f)['data'] for f in files], axis=0) # [samples, C * H * W]
        data = data.reshape(
            [data.shape[0], data_params.channels, data_params.target_res, data_params.target_res])  # [samples, C, H, W]
        data = data.transpose([0, 2, 3, 1])  # [samples, H, W, C]
        assert data.shape[1] == data.shape[2] == data_params.target_res
        assert data.shape[3] == data_params.channels

        data = shuffle(data)
        print('Number of Images:', len(data))
        print('Path: ', path)
        return data





