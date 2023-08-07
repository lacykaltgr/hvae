import numpy as np
import tensorflow as tf
import torch
from sklearn.utils import shuffle

from src.elements.dataset import _DataSet


class MNISTDataSet(_DataSet):
    def __init__(self):
        super(MNISTDataSet).__init__()

    def load(self):
        # Ignore labels
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

        x_train = x_train[:, np.newaxis, :, :]  # (60000, 1, 28, 28)
        x_test = x_test[:, np.newaxis, :, :]  # (10000, 1,  28, 28)
        x_train, x_test = x_train / 255., x_test / 255.

        # make images of size 32x32
        x_train = np.pad(x_train, pad_width=((0, 0), (0, 0), (2, 2), (2, 2)))  # (60000, 1, 32, 32)
        x_test = np.pad(x_test, pad_width=((0, 0), (0, 0), (2, 2), (2, 2)))  # (60000, 1, 32, 32)

        x_train = shuffle(x_train)
        x_test = shuffle(x_test,
                         random_state=101)  # Fix this seed to not overlap val and test between train and inference runs

        x_val = x_test[:len(x_test) // 2]  # 5000
        x_test = x_test[len(x_test) // 2:]  # 5000

        x_val, x_test = torch.tensor(x_val), torch.tensor(x_test)
        x_val = torch.Tensor(x_val.size()).bernoulli_(x_val)  # fix binarization
        x_test = torch.Tensor(x_test.size()).bernoulli_(x_test)

        return x_train, x_val, x_test