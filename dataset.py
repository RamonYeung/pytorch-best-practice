import re
import json
import numpy as np
import pandas as pd
import torch
from itertools import chain
from collections import Counter
from abc import ABC, abstractmethod, ABCMeta
from utils import personal_display_settings

# Global Display Settings
personal_display_settings()


class NLPDataset(metaclass=ABCMeta):
    """
    Abstract Base Class (ABC) for all NLP Datasets
    ... when you have enough RAM to load the entire dataset into!
    """
    @abstractmethod
    def __init__(self):
        """
        Initilization of one dataset should be calling the following methods
        in specific order:

        (1) load_data():
            - load train, dev and test files
            - return pandas:DataFrame for self.train, self.dev and self.test

        (2) build_vocab():
            - generate word2id (dict) and id2word (list)

        (3) generate_tensors():
            - return tensors for torch.utils.data.dataset.TensorDataset

        (4) report():
            - print / log some useful dataset info
        """
        pass

    @abstractmethod
    def load_data(self):
        """
        Whatever messsssy data preprocessing pipeline.

        Usage:
        =================================================================================
            self.train = load_data('train')
            self.test = load_data('test')
            self.dev = load_data('dev')
        =================================================================================
        :return: data in clean text
        """
        pass

    @abstractmethod
    def build_vocab(self):
        """
        Build vocabulary from self.train

        Usage:
        =================================================================================
            self.id2word, self.id2word = build_vocab()
        =================================================================================
        :return: id2word :list, word2id :dict
        """
        pass

    @abstractmethod
    def generate_tensors(self):
        """
        Note that generate tensors on-the-fly will be much slower. Since NLP datasets
        are often small enough, we directly generate all the tensors and store them
        into memory.

        Alternatives:
            Inheriting torch.utils.data.Dataset and generate tensors on-the-fly.
        =================================================================================
            class MyDataset(torch.utils.data.Dataset):
                def __init__(self):
                    pass

                def __getitem__(self):
                    pass

                def __len__(self):
                    # return something larger than 0, otherwise SegmentFault.
                    pass
        =================================================================================

        Usage:
        =================================================================================
            self.train_x, self.train_y = self.generate_tensors(self.train)
            self.test_x, self.test_y = self.generate_tensors(self.test)
            self.dev_x, self.dev_y = self.generate_tensors(self.dev)
        =================================================================================

        Outside the class:
            1. 'use torch.utils.data.TensorDataset(*tensors)' to wrap tensors
            as iterable dataset (x and y)
            E.g.
        =================================================================================
            train_dataset = TensorDataset(NLPDataset.train_x, NLPDataset.train_y)
        =================================================================================

            2. use 'torch.utils.data.DataLoader()' to loop forever.
            E.g.
        =================================================================================
            train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            for batch_i, (train_x, train_y) in enumerate(train_dataloader):
                # ...
        ==================================================================================
        :return: tensors or tuple of tensors.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Print of logging everything about your dataset.
        ==================================================================================
            print('=====Dataset Info=====')
            print(f'seq_max_length: {self.seq_max_length}')
            print(f'#train_data_points: {len(self.train_x)}')
            print(f'#test_data_points: {len(self.test_x)}')
            print(f'#dev_data_points: {len(self.dev_x)}')
            print('======================')
            print(self.train.tail(10))
        ==================================================================================
        :return: None
        """
        pass
