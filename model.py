# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x).transpose(-2, -1)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K) / (Q.size(-1) ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention, V)
        return attended + x


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.attention1 = SelfAttention(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.21, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.attention1(x)
        x = F.dropout(x, p=0.31, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=0.11, training=self.training)
        x = self.fc4(x)

        return x




