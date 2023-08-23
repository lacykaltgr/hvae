import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from hparams import get_hparams


def default_transform():
    params = get_hparams()
    return lambda x: transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(params.data_params.shape),
        #Normalize(),
        #MinMax(),
    ])(x)


class Normalize(object):
    def __call__(self, img):
        """
        Normalize image to be between [0, 1]

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
        Normalize image to be between [-1, 1]

        :param img: PIL): Image
        :return: Tensor
        """
        img = np.asarray(img)

        shift = scale = (2 ** 8 - 1) / 2
        img = (img - shift) / scale  # Images are between [-1, 1]
        return torch.tensor(img).permute(2, 0, 1).contiguous().float()

    def __repr__(self):
        return self.__class__.__name__ + '()'
