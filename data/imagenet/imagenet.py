import os
import numpy as np

from hvae_backbone.elements.dataset import DataSet


class ImageNetDataSet(DataSet):
    def __init__(self, root_path):
        self.root_path = root_path
        super(ImageNetDataSet, self).__init__()

    def load(self):
        train_data, val_data, test_data = None, None, None
        train_labels, val_labels, test_labels = None, None, None

        # TRAIN DATA
        for idx in range(1, 10):
            file_path = os.path.join(self.root_path, f'train_data_batch_{idx}')
            x, y = load_databatch(file_path)
            if idx == 1:
                train_data = x
                train_labels = y
            else:
                train_data = np.concatenate((train_data, x), axis=0)
                train_labels = np.concatenate((train_labels, y), axis=0)

        # VAL DATA
        file_path = os.path.join(self.root_path, f'val_data')
        x, y = load_databatch(file_path)
        val_data = x

        # TEST DATA
        file_path = os.path.join(self.root_path, f'test_data')
        x, y = load_databatch(file_path)
        test_data = x

        return train_data, val_data, test_data


def load_databatch(filpath, img_size=32):
    from hvae_backbone.utils import unpickle
    d = unpickle(filpath)
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    y = [i-1 for i in y]

    x -= mean_image
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return x, y



