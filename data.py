import gzip
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from util import ac_code_of,ld_code_of


class RapppidDataset(Dataset):
    def __init__(self, rows, seqs, trunc_len=1000):
        super().__init__()

        self.trunc_len = trunc_len
        self.rows = rows
        self.seqs = seqs
        self.aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                    'Y', 'V', 'O', 'U']
        self.protein_sequence = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                 'W',
                                 'Y', 'V', 'O', 'U']
        self.protein_sequence_dict = {}
        id = 1
        for i in range(len(self.protein_sequence)):
                    self.protein_sequence_dict[
                        self.protein_sequence[i] ] = id
                    id += 1
        assert id - 1 == len(self.protein_sequence)

    def __getitem__(self, idx):
        p1, p2, label = self.rows[idx]
        p1_seq = "".join([self.aas[r] for r in self.seqs[p1][:self.trunc_len]])
    


        p1_seq = np.array(self.to_gap_D(p1_seq, 1))

        p2_seq = "".join([self.aas[r] for r in self.seqs[p2][:self.trunc_len]])

        p2_seq = np.array(self.to_gap_D(p2_seq, 1))

        p1_pad_len = self.trunc_len - len(p1_seq)
        p2_pad_len = self.trunc_len - len(p2_seq)

        p1_seq = np.pad(p1_seq, (0, p1_pad_len), 'constant')
        p2_seq = np.pad(p2_seq, (0, p2_pad_len), 'constant')

        p1_seq = torch.tensor(p1_seq).long()
        p2_seq = torch.tensor(p2_seq).long()
        label = torch.tensor(label).long()

        return (p1_seq, p2_seq, label)

    def __len__(self):
        return len(self.rows)

    def to_gap_D(self, seq, gap):
        return [self.protein_sequence_dict[seq[i:i + gap]] for i in range(len(seq)) if not i + gap > len(seq)]


class RapppidDataModule():

    def __init__(self, batch_size: int, train_path: str, val_path: str, test_path: str, seqs_path: str):
        super().__init__()

        self.batch_size = batch_size
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.seqs_path = seqs_path

        self.dataset_train = None
        self.dataset_test = None

        self.train = []
        self.test = []
        self.seqs = []

        with gzip.open(self.seqs_path) as f:
            self.seqs = pickle.load(f)

        with gzip.open(self.test_path) as f:
            self.test_pairs = pickle.load(f)

        with gzip.open(self.val_path) as f:
            self.val_pairs = pickle.load(f)

        with gzip.open(self.train_path) as f:
            self.train_pairs = pickle.load(f)

        self.dataset_train = RapppidDataset(self.train_pairs, self.seqs)
        self.dataset_val = RapppidDataset(self.val_pairs, self.seqs)
        self.dataset_test = RapppidDataset(self.test_pairs, self.seqs)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False)
