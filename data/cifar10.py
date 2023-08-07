import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from src.elements.dataset import _DataSet


class CIFARDataset(_DataSet):
    def __init__(self):
        super(CIFARDataset).__init__()

    def load(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        x_train = shuffle(x_train)
        x_test = shuffle(x_test, random_state=101)  # Fix this seed to not overlap val and test between train and inference runs
        x_val = x_test[:len(x_test) // 2]  # 5000
        x_test = x_test[len(x_test) // 2:]  # 5000
        return x_train.astype(np.uint8), x_val.astype(np.uint8), x_test.astype(np.uint8)





