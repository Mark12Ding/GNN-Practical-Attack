import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, n_units, dropout, activation):
        super(GCN, self).__init__()
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.gc1 = GraphConv(in_feats, n_units, activation=self.activation)
        self.gc2 = GraphConv(n_units, out_feats)
        self.dropout = dropout

    def forward(self, data):
        x = data.features
        x = self.gc1(data.g, x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(data.g, x)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_units,
                 out_feats,
                 num_heads,
                 activation,
                 dropout,
                 negative_slope=0.2):
        super(GAT, self).__init__()
        self.gat_layers = nn.ModuleList()
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.gat_layers.append(GATConv(in_feats, n_units, num_heads, dropout, dropout, negative_slope, False, self.activation))
        self.gat_layers.append(GATConv(n_units * num_heads, out_feats, 1, dropout, dropout, negative_slope, False, None))

    def forward(self, data):
        h = data.features
        h = self.gat_layers[0](data.g, h).flatten(1)
        logits = self.gat_layers[1](data.g, h).mean(1)
        return F.log_softmax(logits, dim=1)

class JKNetMaxpool(nn.Module):
    def __init__(self, in_feats, out_feats, n_layers, n_units, dropout,
                 activation):
        super(JKNetMaxpool, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(in_feats, n_units, activation=self.activation))
        self.dropout = dropout
        for i in range(1, self.n_layers):
            self.layers.append(
                GraphConv(n_units, n_units, activation=self.activation))
        self.layers.append(GraphConv(n_units, out_feats))

    def forward(self, data):
        h = data.features
        layer_outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = layer(data.g, h)
            layer_outputs.append(h)
        h = th.stack(layer_outputs, dim=0)
        h = th.max(h, dim=0)[0]
        h = self.layers[-1](data.g, h)
        return F.log_softmax(h, dim=1)


class JKNetConCat(nn.Module):
    def __init__(self, in_feats, out_feats, n_layers, n_units, dropout,
                 activation):
        super(JKNetConCat, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(in_feats, n_units, activation=self.activation))
        self.dropout = dropout
        for i in range(1, self.n_layers):
            self.layers.append(
                GraphConv(n_units, n_units, activation=self.activation))
        self.layers.append(GraphConv(n_layers * n_units, out_feats))

    def forward(self, data):
        h = data.features
        layer_outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = layer(data.g, h)
            layer_outputs.append(h)
        h = th.cat(layer_outputs, dim=1)
        h = self.layers[-1](data.g, h)
        return F.log_softmax(h, dim=1)
