import torch as th
import numpy as np


def getScore(K, data):
    Random = data.Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, data.Prob)
    return Random.sum(dim=0)


def getScoreGreedy(K, data, bar, num, beta):
    Random = data.Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, data.Prob)
    W = th.zeros(data.size, data.size)
    for i in range(data.size):
        value, index = th.topk(Random[i], beta)
        for j, ind in zip(value, index):
            if j != 0:
                W[i, ind] = 1
    SCORE = W.sum(dim=0)
    ind = []
    l = [i for i in range(data.size) if data.g.out_degree(i) <= bar]
    for _ in range(num):
        cand = [(SCORE[i], i) for i in l]
        best = max(cand)[1]
        for neighbor in data.g.out_edges(best)[1]:
            if neighbor in l:
                l.remove(neighbor)
        ind.append(best)
        for i in l:
            W[:, i] -= (W[:, best] > 0) * 1.0
        SCORE = th.sum(W > 0, dim=0)
    return np.array(ind)


def getThrehold(g, size, threshold, num):
    degree = g.out_degrees(range(size))
    Cand_degree = sorted([(degree[i], i) for i in range(size)], reverse=True)
    threshold = int(size * threshold)
    bar, _ = Cand_degree[threshold]
    Baseline_Degree = []
    index = [j for i, j in Cand_degree if i == bar]
    if len(index) >= num:
        Baseline_Degree = np.array(index)[np.random.choice(len(index),
                                                           num,
                                                           replace=False)]
    else:
        while 1:
            bar -= 1
            index_ = [j for i, j in Cand_degree if i == bar]
            if len(index) + len(index_) >= num:
                break
            for i in index_:
                index.append(i)
        for i in np.array(index_)[np.random.choice(len(index_),
                                                   num - len(index),
                                                   replace=False)]:
            index.append(i)
        Baseline_Degree = np.array(index)
    random = [j for i, j in Cand_degree if i <= bar]
    Baseline_Random = np.array(random)[np.random.choice(len(random),
                                                        num,
                                                        replace=False)]
    return bar, Baseline_Degree, Baseline_Random


def getIndex(g, Cand, bar, num):
    ind = []
    for j, i in Cand:
        if g.out_degree(i) <= bar:
            ind.append(i)
        if len(ind) == num:
            break
    return np.array(ind)
