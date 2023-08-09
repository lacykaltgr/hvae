from enum import Enum

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from hparams import get_hparams


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
        params = get_hparams()

        img = np.asarray(img)
        img_dtype = img.dtype

        img = np.floor(img / np.uint8(2 ** (8 - params.data_params.num_bits))) * 2 ** (8 - params.data_params.num_bits)
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
    transforms.ToTensor(),
    #Normalize(),
    #MinMax(),
])


class _DataSet(TorchDataset):

    def __init__(self,
                 train_transform=default_transform,
                 val_transform=default_transform,
                 test_transform=default_transform):
        self.mode = DataSetState.TEST
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_transform = val_transform

        train_data, val_data, test_data = self.load()

        self.train_set = self._FunctionalDataset(train_data, self.train_transform)
        self.val_set = self._FunctionalDataset(val_data, self.val_transform)
        self.test_set = self._FunctionalDataset(test_data, self.test_transform)

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
            item = self.train_set[idx]
            return self.train_transform(item)
        elif self.mode == DataSetState.VAL:
            item = self.val_set[idx]
            return self.val_transform(item)
        elif self.mode == DataSetState.TEST:
            item = self.test_set[idx]
            return self.test_transform(item)
        else:
            raise ValueError("Invalid mode")

    def __len__(self):
        if self.mode == DataSetState.TRAIN:
            return len(self.train_set)
        elif self.mode == DataSetState.VAL:
            return len(self.val_set)
        elif self.mode == DataSetState.TEST:
            return len(self.test_set)

    def get_train_loader(self):
        train_params = get_hparams().train_params
        return torch.utils.data.DataLoader(dataset=self.train_set,
                                           batch_size=train_params.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=True)

    def get_val_loader(self):
        eval_params = get_hparams().eval_params
        return torch.utils.data.DataLoader(dataset=self.val_set,
                                           batch_size=eval_params.batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           drop_last=True)

    def get_test_loader(self):
        eval_params = get_hparams().test_params
        return torch.utils.data.DataLoader(dataset=self.test_set,
                                           batch_size=eval_params.batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           drop_last=True)

    def load(self):
        return None, None, None

    class _FunctionalDataset(TorchDataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform

        def __getitem__(self, idx):
            item = self.data[idx]
            #if self.transform:
            #    item = self.transform(item)
            return torch.tensor(item).to(torch.float32)

        def __len__(self):
            return self.data.shape[0]