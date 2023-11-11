from abc import ABC, abstractmethod

import torch
from src.elements.data_preprocessing import default_transform
from torch.utils.data import Dataset as TorchDataset
from src.hparams import get_hparams


class _DataSet(ABC):

    """
    Abstract class for datasets
    Inherit from this class to create a new dataset in /data
    """

    def __init__(self,
                 train_transform=default_transform,
                 val_transform=default_transform,
                 test_transform=default_transform,
                with_labels=False):
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_transform = val_transform
        
        data = self.load(with_labels=with_labels)
        
        if not with_labels:
            train_data, val_data, test_data = data
            self.train_set = self._FunctionalDataset(train_data, transform=self.train_transform)
            self.val_set = self._FunctionalDataset(val_data, transform=self.val_transform)
            self.test_set = self._FunctionalDataset(test_data, transform=self.test_transform)
        else:
            train_data, val_data, test_data,train_labels, val_labels, test_labels = data
            self.train_set = self._FunctionalDataset(train_data, train_labels, self.train_transform)
            self.val_set = self._FunctionalDataset(val_data,val_labels, self.val_transform)
            self.test_set = self._FunctionalDataset(test_data, test_labels, self.test_transform)
            
            

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
        eval_params = get_hparams().eval_params
        if not with_labels:
            return torch.utils.data.DataLoader(dataset=self.test_set,
                                               batch_size=eval_params.batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)

    @abstractmethod
    def load(self):
        """
        Load data from disk
        :return: train, val, test
        """
        pass

    class _FunctionalDataset(TorchDataset):

        """
        Dataset class for functional datasets (train, validation, test)
        """
        def __init__(self, data, labels=None, transform=None):
            self.data = data
            self.transform = transform
            self.labels = labels

        def __getitem__(self, idx):
            params = get_hparams()
            item = self.data[idx]
            #if self.transform:
            #    item = self.transform(item)
            item = torch.tensor(item).view(params.data_params.shape).to(torch.float32)
            if not self.labels is not None:
                return item
    
            label = self.labels[idx]
            label = torch.tensor(label)
            return item, label
            

        def __len__(self):
            return self.data.shape[0]
        
    