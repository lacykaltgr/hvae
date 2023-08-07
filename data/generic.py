import os

from PIL import Image

from hparams import *
from src.elements.dataset import _DataSet, DataSetState


class GenericDataSet(_DataSet):
    def __init__(self):
        super(GenericDataSet).__init__()
        self.train_files = self.create_filenames_list(data_params.train_data_path)
        self.val_files = self.create_filenames_list(data_params.val_data_path)
        self.test_files = self.create_filenames_list(data_params.test_data_path)

    def __getitem__(self, idx):
        if self.mode == DataSetState.TRAIN:
            file = self.train_files[idx]
            item = self.read_resize_image(file)
            return self.train_transform(item)
        elif self.mode == DataSetState.VAL:
            file = self.val_files[idx]
            item = self.read_resize_image(file)
            return self.val_transform(item)
        elif self.mode == DataSetState.TEST:
            file = self.test_files[idx]
            item = self.read_resize_image(file)
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

    def create_filenames_list(self, path):
        filenames = sorted(os.listdir(path))
        files = [os.path.join(path, f) for f in filenames]
        print(path, len(files))
        return files

    def read_resize_image(self, image_file):
        return Image.open(image_file).convert("RGB").resize\
            ((data_params.target_res, data_params.target_res), resample=Image.BILINEAR)


