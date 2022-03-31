import torch
from model import Model
from data import RapppidDataModule
import os

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DefaultConfig(object):
    def __init__(self):
        self.max_epochs = 100
        self.lr = 0.01  # initial learning rate
        self.weight_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim=25

        self.log_path = 'result/log.txt'
        self.save_model_path_train = 'result/save_train/'
        self.save_model_path_val = 'result/save_val/'
        self.save_train_inforamtion = 'result/train_result.txt'
        self.save_val_inforamtion = 'result/val_result.txt'
        self.save_test_inforamtion = 'result/test.txt'
        if not os.path.exists(self.save_model_path_train):
            os.mkdir(self.save_model_path_train)
        if not os.path.exists(self.save_model_path_val):
            os.mkdir(self.save_model_path_val)

        self.train_path = '../model2/data/rapppid/comparatives/string_c1/train_pairs.pkl.gz'
        self.val_path = '../model2/data/rapppid/comparatives/string_c1/val_pairs.pkl.gz'
        self.test_path = '../model2/data/rapppid/comparatives/string_c1/test_pairs.pkl.gz'
        self.seqs_path = '../model2/data/rapppid/comparatives/string_c1/seqs.pkl.gz'
        self.batch_size = 64
        self.model = Model(input_dim=self.input_dim)
        self.data = RapppidDataModule(self.batch_size, self.train_path, self.val_path, self.test_path,
                                      self.seqs_path)

        self.criterion = nn.CrossEntropyLoss()
        self.steps_per_epoch = len(self.data.dataset_train) // self.batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', patience=3)

        informatinon = ""
        for name, value in vars(self).items():
            informatinon += '%s=%s' % (name, value) + '   '

        print(informatinon)
        with open(self.log_path, 'a+') as file:
            file.write(informatinon)
