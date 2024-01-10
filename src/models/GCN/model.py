"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout_prob):
        super(GCN, self).__init__()

        self.gc_layers = nn.ModuleList()

        # self.bns = nn.ModuleList()

        self.activation = activation

        # input layer & hidden layers
        for i in range(n_layers - 1):
            if i == 0:
                self.gc_layers.append(GraphConv(in_feats, n_hidden, activation=None))
            else:
                self.gc_layers.append(GraphConv(n_hidden, n_hidden, activation=None))
            # self.bns.append(nn.BatchNorm1d(n_hidden))
        
        # output layer
        self.gc_layers.append(GraphConv(n_hidden, n_classes))

        # dropout
        self.dropout_prob = dropout_prob

    def forward(self, g, x):
        for i, conv in enumerate(self.gc_layers[:-1]):
            x = conv(g, x)
            # x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        logits = self.gc_layers[-1](g, x)

        # ** remove log_softmax
        return logits

