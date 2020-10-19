import argparse
import random

from torch.autograd.gradcheck import zero_gradients
import torch as th
import torch.nn.functional as F

from utils import load_data, split_data
from model import GCN, JKNetConCat, JKNetMaxpool, GAT
from attack import getScore, getScoreGreedy, getThrehold, getIndex


import numpy as np
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.centrality import betweenness_centrality as betweenness
from copy import deepcopy 

parser = argparse.ArgumentParser()

# General configs.
parser.add_argument("--dataset",
                    default="citeseer",
                    help="[cora, pubmed, citeseer, synthetic]")
parser.add_argument("--model",
                    default="GCN",
                    help="[GCN, GAT, JKNetConCat, JKNetMaxpool]")
parser.add_argument("--result_path", default="results")
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help="Early stopping patience.")
parser.add_argument("--seed", type=int, default=42, help="Random Seed")
parser.add_argument("--epochs",
                    type=int,
                    default=200,
                    help="Number of epochs to train.")
parser.add_argument("--verbose", type=int, default=0, help="Verbose.")
parser.add_argument("--train",
                    type=float,
                    default=0.6,
                    help="Train data portion.")
parser.add_argument("--test",
                    type=float,
                    default=0.2,
                    help="Test data portion.")
parser.add_argument("--validation",
                    type=float,
                    default=0.2,
                    help="Validation data portion.")

# Common hyper-parameters.
parser.add_argument("--lr",
                    type=float,
                    default=5e-3,
                    help="Initial learning rate.")
parser.add_argument("--weight_decay",
                    type=float,
                    default=5e-4,
                    help="Weight decay (L2 loss on parameters).")
parser.add_argument("--hidden",
                    type=int,
                    default=32,
                    help="Number of hidden units.")
parser.add_argument("--num_heads",
                    type=int,
                    default=8,
                    help="Number of attention heads.")
parser.add_argument("--hidden_layers",
                    type=int,
                    default=6,
                    help="Number of hidden layers.")
parser.add_argument("--dropout",
                    type=float,
                    default=0.5,
                    help="Dropout rate (1 - keep probability).")
parser.add_argument("--activation", default="relu")

# Attack setting
parser.add_argument("--num_node",
                    type=int,
                    default=33,
                    help="Number of target nodes.")
parser.add_argument("--num_features",
                    type=int,
                    default=74,
                    help="Number of modified features.")
parser.add_argument("--threshold",
                    type=float,
                    default=0.1,
                    help="Threshold percentage of degree.")
parser.add_argument("--norm_length",
                    type=float,
                    default=1,
                    help="Variable lambda in the paper.")
parser.add_argument("--beta",
                    type=int,
                    default=30,
                    help="Variable l in the paper.")
parser.add_argument("--steps",
                    type=int,
                    default=4,
                    help="Steps of Random Walk")

args = parser.parse_args()  
print("Random Seed:%d" % args.seed)
print("Threshold:%.2f" % args.threshold)
random.seed(args.seed)
np.random.seed(args.seed)
th.manual_seed(args.seed)
data = load_data(dataset=args.dataset)
print("Attack Setting:")
print(
    "Number of victim nodes:{}\nNumber of modified features:{}\nDegree threshold:{}\nPerturbation strength:{}\nSteps:{}"
    .format(args.num_node, args.num_features, args.threshold, args.norm_length,
            args.steps))

model_args = {
    "in_feats": data.features.shape[1],
    "out_feats": data.num_labels,
    "n_units": args.hidden,
    "dropout": args.dropout,
    "activation": args.activation
}

def init_model():
    if args.model == "GCN":
        model = GCN(**model_args)
    elif args.model == "GAT":
        model_args["num_heads"] = 8
        model_args["n_units"] = 8
        model_args["dropout"] = 0.6
        model_args["activation"] = "elu"
        model = GAT(**model_args)
    else:
        model_args["n_layers"] = args.hidden_layers
        if args.model == "JKNetConCat":
            model = JKNetConCat(**model_args)
        elif args.model == "JKNetMaxpool":
            model = JKNetMaxpool(**model_args)
        else:
            print("Model should be GCN, GAT, JKNetConCat or JKNetMaxpool.")
            assert False

    optimizer = th.optim.Adam(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    return model, optimizer


def evaluate(model, data, mask):
    model.eval()
    with th.no_grad():
        logits = model(data)
        logits = logits[mask]
        _, indices = th.max(logits, dim=1)
        labels = data.labels[mask]
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train():
    model.train()
    logits = model(data)
    loss = F.nll_loss(logits[idx_train], data.labels[idx_train])
    val_loss = F.nll_loss(logits[idx_val], data.labels[idx_val]).item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = evaluate(model, data, idx_train)
    val_acc = evaluate(model, data, idx_val)
    test_acc = evaluate(model, data, idx_test)
    return val_loss, [train_acc, val_acc, test_acc]


def Train():
    patience = args.patience
    best_val_loss = np.inf
    selected_accs = None
    for epoch in range(1, args.epochs):
        if patience < 0:
            print("Early stopping happen at epoch %d." % epoch)
            break
        val_loss, accs = train()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            selected_accs = accs
            patience = args.patience
            if args.verbose:
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, *accs))
        else: 
            patience -= 1
    log = 'Training finished. Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(*accs))


def grad_attack(norm_length):
    data.features.requires_grad_(True)
    model.eval()
    logits = model(data)
    loss = F.nll_loss(logits[idx_train], data.labels[idx_train])
    optimizer.zero_grad()
    zero_gradients(data.features)
    loss.backward(retain_graph=True)
    grad = data.features.grad.detach().clone()
    signs, indexs = pick_feature(grad, args.num_features)
    data.features.requires_grad_(False)
    result = th.zeros(7, 2)
    result[0][0] = evaluate(model, data, idx_test)
    for i, targets in enumerate([
            Baseline_Degree, Baseline_Pagerank, Baseline_Between,
            Baseline_Random, GC_RWCS, RWCS
    ]):
        for target in targets:
            for index in indexs:
                data.features[target][index] += norm_length * signs[index]
        result[i][0] = evaluate(model, data, idx_test)
        model.eval()
        with th.no_grad():
            logits = model(data)[idx_test]
            result[i][1] = F.nll_loss(logits, data.labels[idx_test])
        for target in targets:
            for index in indexs:
                data.features[target][index] -= norm_length * signs[index]
    result[-1,0] = evaluate(model, data, idx_test)
    model.eval()
    with th.no_grad():
        logits = model(data)[idx_test]
        result[-1,1] = F.nll_loss(logits, data.labels[idx_test])
    return result

def black_attack(norm_length=10):
    result = th.zeros(7, 2)
    for i, targets in enumerate([
            Baseline_Degree, Baseline_Pagerank, Baseline_Between,
            Baseline_Random, GC_RWCS, RWCS
    ]):  
        data = deepcopy(data_backup)
        for target in targets:
            positive = data.features[target][0] + data.features[target][2]
            negative = data.features[target][1] + data.features[target][3]
            if positive > negative:
                data.features[target][0] -= norm_length 
                data.features[target][1] += norm_length 
            else:
                data.features[target][0] += norm_length 
                data.features[target][1] -= norm_length 
        result[i][0] = evaluate(model, data, idx_test)
        model.eval()
        with th.no_grad():
            logits = model(data)[idx_test]
            result[i][1] = F.nll_loss(logits, data.labels[idx_test])    
        data = deepcopy(data_backup)
        result[-1,0] = evaluate(model, data, idx_test)
        model.eval()
        with th.no_grad():
            logits = model(data)[idx_test]
            result[-1,1] = F.nll_loss(logits, data.labels[idx_test])
    return result

def pick_feature(grad, k):
    score = grad.sum(dim=0)
    _, indexs = th.topk(score.abs(), k)
    signs = th.zeros(data.features.shape[1])
    for i in indexs:
        signs[i] = score[i].sign()
    return signs, indexs


assert args.train + args.test + args.validation <= 1
NumTrain = int(data.size * args.train)
NumTest = int(data.size * args.test)
NumVal = int(data.size * args.validation)


nxg = nx.Graph(data.g.to_networkx())
page = pagerank(nxg)
between = betweenness(nxg)
PAGERANK = sorted([(page[i], i) for i in range(data.size)], reverse=True)
BETWEEN = sorted([(between[i], i) for i in range(data.size)], reverse=True)
Important_score = getScore(args.steps, data)
Important_list = sorted([(Important_score[i], i) for i in range(data.size)],
                        reverse=True)
bar, Baseline_Degree, Baseline_Random = getThrehold(data.g, data.size,
                                                    args.threshold,
                                                    args.num_node)
Baseline_Pagerank = getIndex(data.g, PAGERANK, bar, args.num_node)
Baseline_Between = getIndex(data.g, BETWEEN, bar, args.num_node)
RWCS = getIndex(data.g, Important_list, bar, args.num_node)
GC_RWCS = getScoreGreedy(args.steps, data, bar, args.num_node, args.beta)
model, optimizer = init_model()
idx_train, idx_val, idx_test = split_data(data, NumTrain, NumTest, NumVal)
print("Attack model:\n", model)
print(optimizer)
print("Num_Train : %d\nNum_valiation : %d\nNum_Test : %d\n" %
      (len(idx_train), len(idx_val), len(idx_test)))
Train()


print("===================Node chosen(threshold:%.2f)=================" %
      args.threshold)
print("Baseline_Degree:\n", Baseline_Degree, "Those degree:\n",
      data.g.out_degrees(Baseline_Degree))
print("Baseline_Pagerank:\n", Baseline_Pagerank, "Those degree:\n",
      data.g.out_degrees(Baseline_Pagerank))
print("Baseline_Between:\n", Baseline_Between, "Those degree:\n",
      data.g.out_degrees(Baseline_Between))
print("Baseline_Random:\n", Baseline_Random, "Those degree:\n",
      data.g.out_degrees(Baseline_Random))
print("GC-RWCS:\n", GC_RWCS, "Those degree:\n", data.g.out_degrees(GC_RWCS))
print("RWCS:\n", RWCS, "Those degree:\n", data.g.out_degrees(RWCS))
data_backup = deepcopy(data)   
if args.dataset == "synthetic":
    result = black_attack(args.norm_length)
else:   
    result = grad_attack(args.norm_length)
for index, method in enumerate([
        "Baseline_Degree", "Baseline_Pagerank", "Baseline_Between",
        "Baseline_Random", "GC-RWCS", "RWCS", "None"
]):
    print("{} : Accuracy : {:.4f}, Loss : {:.4f}".format(
        method, result[index][0].item(), result[index][1].item()))
