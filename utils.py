from dgl.data import citation_graph as citegrh
import networkx as nx
import torch as th
import numpy as np
from easydict import EasyDict
from torch.nn.functional import normalize
from dgl import DGLGraph
from networkx.generators.random_graphs import fast_gnp_random_graph
from numpy.random import multivariate_normal

def synthetic_data(num_node=3000, num_feature=10, num_class=2, num_important=4):
    gnp = nx.barabasi_albert_graph(num_node, 2)
    gnp.remove_edges_from(nx.selfloop_edges(gnp))
    g = DGLGraph(gnp)
    g.add_edges(g.nodes(), g.nodes())
    data = EasyDict()
    data.graph = gnp
    data.num_labels = num_class
    data.g = g
    data.adj = g.adjacency_matrix(transpose=None).to_dense()
    means = np.zeros(num_node)
    degree = np.zeros((num_node, num_node))
    for i in range(num_node):
        degree[i,i] = data.adj[i].sum()**-0.5
    lap_matrix = np.identity(num_node) - np.matmul(np.matmul(degree, data.adj.numpy()), degree)
    cov = np.linalg.inv(lap_matrix + np.identity(num_node))
    data.features = th.from_numpy(multivariate_normal(means, cov, num_feature).transpose())
    data.features = data.features.float().abs()
    g.ndata['x'] = data.features
    W = th.randn(num_feature) * 0.1
    W[range(num_important)] = th.Tensor([10,-10,10,-10])
    data.Prob = normalize(th.FloatTensor(data.adj), p=1, dim=1)
    logits = th.sigmoid(th.matmul(th.matmul(normalize(data.adj, p=1, dim=1), data.features), W)) 
    labels = th.zeros(num_node)
    labels[logits>0.5] = 1      
    data.labels = labels.long()
    data.size = num_node
    return data

def load_data(dataset="cora"):
    assert dataset in ["cora", "pubmed", "citeseer", "synthetic"]
    if dataset == "cora":
        data = citegrh.load_cora()
    elif dataset == "pubmed":
        data = citegrh.load_pubmed()
    elif dataset == "citeseer":
        data = citegrh.load_citeseer()
    else:
        data = synthetic_data()
    data.features = th.FloatTensor(data.features)
    data.labels = th.LongTensor(data.labels)
    data.size = data.labels.shape[0]
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    data.g = g
    data.adj = g.adjacency_matrix(transpose=None).to_dense()
    data.Prob = normalize(th.FloatTensor(data.adj), p=1, dim=1)
    print("============Successfully Load %s===============" % dataset)
    return data


def split_data(data, NumTrain, NumTest, NumVal):
    idx_test = np.random.choice(data.size, NumTest, replace=False)
    without_test = np.array([i for i in range(data.size) if i not in idx_test])
    idx_train = without_test[np.random.choice(len(without_test),
                                              NumTrain,
                                              replace=False)]
    idx_val = np.array([
        i for i in range(data.size) if i not in idx_test if i not in idx_train
    ])
    idx_val = idx_val[np.random.choice(len(idx_val), NumVal, replace=False)]
    return idx_train, idx_val, idx_test
