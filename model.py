import torch
import torch.nn as nn
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNNLayer1(nn.Module):
    def __init__(self, input_dim, in_len=1000, pool_size=3, cnn_hidden=1):
        super(CNNLayer1, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_hidden, kernel_size=3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=2)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / pool_size) - 1, 256)

    def forward(self, x):
        x1 = x.transpose(1, 2)
        x1 = self.conv1d(x1)
        x1 = self.bn1(x1)
        x1 = self.maxpool1d(x1)
        x1 = x1.transpose(1, 2)
        x1, _ = self.biGRU(x1)
        x1 = self.global_avgpool1d(x1)
        x1 = x1.squeeze()
        x1 = self.fc1(x1)
        return x1
class CNNLayer2(nn.Module):
    def __init__(self, input_dim, hid_dim=64, kernel_size=3, init_weights=True):
        super(CNNLayer2, self).__init__()
        self.maxpol = nn.MaxPool1d(kernel_size)
        self.cov1 = nn.Conv1d(in_channels=input_dim, out_channels=hid_dim, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.cov2 = nn.Conv1d(in_channels=hid_dim, out_channels=hid_dim * 2, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm1d(hid_dim * 2)
        self.cov3 = nn.Conv1d(in_channels=hid_dim * 2, out_channels=hid_dim * 2 * 2, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm1d(hid_dim * 2 * 2)
        self.cov4 = nn.Conv1d(in_channels=hid_dim * 2 * 2, out_channels=hid_dim * 2 * 2 * 2, kernel_size=3, bias=False)
        self.bn4 = nn.BatchNorm1d(hid_dim * 2 * 2 * 2)
        self.liner = nn.Linear(512, 256)

        if init_weights:
            self.initialize_weights()
        return

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.bn1(self.cov1(x))
        x = self.maxpol(x)
        x = self.bn2(self.cov2(x))
        x = self.maxpol(x)
        x = self.bn3(self.cov3(x))
        x = self.maxpol(x)
        x = self.bn4(self.cov4(x))
        x = torch.max(x, dim=2).values
        x = self.liner(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal(m.weight)


class Model(nn.Module):
    def __init__(self, input_dim,
                 in_len=1000, pool_size=3, cnn_hidden=1,
                 hid_dim=64, kernel_size=3,
                 init_weights=True):
        super(Model, self).__init__()
        self.feature_ex1 = CNNLayer1(input_dim=input_dim)
        self.feature_ex2 = CNNLayer2(input_dim=input_dim)
        self.lin1 = nn.Linear(256 * 4, 512)
        self.relu = nn.ReLU()
        self.leau = nn.LeakyReLU(0.3)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 2)
        self.drop_out = nn.Dropout(0.1)
        self.embedding = nn.Embedding(num_embeddings=30, embedding_dim=input_dim)

    def forward(self, seq1, seq2):
        seq1 = self.embedding(seq1)
        seq2 = self.embedding(seq2)

        seq1_f1 = self.feature_ex1(seq1)
        seq1_f2 = self.feature_ex2(seq1)
        seq2_f1 = self.feature_ex1(seq2)
        seq2_f2 = self.feature_ex2(seq2)

        all_fea = torch.cat((seq1_f1, seq1_f2, seq2_f1, seq2_f2), dim=1).to(device)
        all_fea = self.lin1(all_fea)
        all_fea = self.drop_out(all_fea)
        all_fea = self.leau(all_fea)
        all_fea = self.lin2(all_fea)
        all_fea = self.leau(all_fea)
        all_fea = self.lin3(all_fea)
        return all_fea
model=Model(input_dim=25)
#网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(model.feature_ex1))