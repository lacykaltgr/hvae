import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms
from enum import Enum
from hparams import *
import numpy as np
from PIL import Image


class DataSetState(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


class Normalize(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype

        img = np.floor(img / np.uint8(2 ** (8 - hparams.data.num_bits))) * 2 ** (8 - hparams.data.num_bits)
        img = img.astype(img_dtype)

        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinMax(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Tensor
        """
        img = np.asarray(img)

        shift = scale = (2 ** 8 - 1) / 2
        img = (img - shift) / scale  # Images are between [-1, 1]
        return torch.tensor(img).permute(2, 0, 1).contiguous().float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


default_transform = transforms.Compose([
    Normalize(),
    MinMax(),
])


class _DataSet(TorchDataset):

    def __init__(self):
        self.mode = DataSetState.TEST

        self.train_data, self.val_data, self.test_data = self.load()

        self.train_transform = default_transform
        self.test_transform = default_transform
        self.val_transform = default_transform

    def train(self):
        self.mode = DataSetState.TRAIN

    def eval(self):
        self.mode = DataSetState.VAL

    def test(self):
        self.mode = DataSetState.TEST

    def mode(self):
        return self.mode

    def __getitem__(self, idx):
        if self.mode == DataSetState.TRAIN:
            item = self.train_data[idx]
            return self.train_transform(item)
        elif self.mode == DataSetState.VAL:
            item = self.val_data[idx]
            return self.val_transform(item)
        elif self.mode == DataSetState.TEST:
            item = self.test_data[idx]
            return self.test_transform(item)
        else:
            raise ValueError("Invalid mode")

    def __len__(self):
        if self.mode == DataSetState.TRAIN:
            return len(self.train_data)
        elif self.mode == DataSetState.VAL:
            return len(self.val_data)
        elif self.mode == DataSetState.TEST:
            return len(self.test_data)

    def train_loader(self):
        return torch.utils.data.DataLoader(dataset=self.train_data,
                                           batch_size=train_params.batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=2,
                                           drop_last=True, prefetch_factor=3)

    def val_loader(self):
        return torch.utils.data.DataLoader(dataset=self.val_data,
                                           batch_size=eval_params.batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=2,
                                           drop_last=True, prefetch_factor=3)

    def test_loader(self):
        return torch.utils.data.DataLoader(dataset=self.val_data,
                                           batch_size=eval_params.batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=2,
                                           drop_last=True, prefetch_factor=3)

    def load(self):
        return None, None, None
