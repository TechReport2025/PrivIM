import argparse
import random
import sys

import numpy as np

import torch
from graph_util import get_adj_mat
from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss
from load_graph import load_train, bfs_split, load_val, load_train_2000, bfs_split_two_parts, bfs_split_dp, rate_split, rate_split_dp, random_walking_with_restart
from util import get_model
from train import train

import time
import logging
import pickle
import logging
from pathlib import Path
import pandas as pd
from itertools import product
from furl import furl
from baselines.heuristics import d_greedy, d_closure, greedy, bfs
from torch.utils.data import Dataset, DataLoader


# ==================== LOG SETTING ====================
DATE = time.strftime('%m-%d', time.localtime())
TIME = time.strftime('%H.%M.%S', time.localtime())
Path(f"log/{DATE}").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"log/{DATE}/debug-{DATE}_{TIME}.log"),
        logging.StreamHandler()
    ]
)
# ==================== END OF LOG SETTING ====================


# ==================== PARAMETERS ====================
TRAIN = True  # TRAIN MODE
HIDDEN_FEATS = [32, 32, 32, 32, 32, 32]
C_LOSS = 1
LOSS_FUNCTION = KSetMaxCoverAdjLoss(C_LOSS)
ROUND = 2

class GraphDataset(Dataset):
    def __init__(self, edges):
        self.edges = edges

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        return {
            'idx': idx,
            'edge': self.edges[idx]
        }

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_upper_bound(args):
    setting = args.settings[0]
    if setting == "PrivIM":
        upper_bound = args.score_upper_bound

    elif setting == "PrivIM+":
        upper_bound = args.score_upper_bound

    return upper_bound

def get_dir_name(args):
    setting = args.settings[0]
    model_name = args.model_name[0]
    dataset = args.dataset[0]
    if model_name == "GRAT3" and setting == "PrivIM+" or setting == "PrivIM":
        dir_name = f"{dataset}_dp/{setting}"
    return dir_name

def get_gnns_name(args):
    model_name = args.model_name[0]
    setting = args.settings[0]
    dataset = args.dataset[0]
    if model_name == "GRAT3":
        gnns_name = f"{dataset}_dp/GRAT"
    return gnns_name


def train_models(debug=False):
# 
    # Prepare training instances
    seed_everything(args.seed[0])
    idx = 0
    for idx in range(args.round):
        print(f"Round: {idx}")

        raw_train_graphs, raw_train_dglgraphs = load_train(input_dim=HIDDEN_FEATS[0], dataset=args.dataset[0])
        # train_graphs, train_dglgraphs, num_layers, min_subgraphs_per_layer = rate_split_dp(raw_train_graphs, raw_train_dglgraphs,
        #             num_subgraphs=args.num_subgraph, min_node_threshold=args.subgraph_size, K=args.K,
        #             r=args.r, k_training=args.k_train, args=args)

        train_graphs, train_dglgraphs = random_walking_with_restart(raw_train_graphs,
                                                                    raw_train_dglgraphs, args=args)

        upper_bound = calculate_upper_bound(args=args)
        dir_name = get_dir_name(args=args)
        gnns_name = get_gnns_name(args=args)
        # train_graphs, train_dglgraphs= bfs_split(raw_train_graphs, raw_train_dglgraphs,
        #                                           num_subgraphs=args.num_subgraph,
        #                                           min_node_threshold=args.subgraph_size,
        #                                           K=args.K,
        #                                           r=args.r, k_training=args.k_train,
        #                                           args=args)
        #
        # train_graphs, train_dglgraphs = rate_split(raw_train_graphs, raw_train_dglgraphs,
        #                                             num_subgraphs=args.num_subgraph,
        #                                             min_node_threshold=args.subgraph_size,
        #                                             K=args.K,
        #                                             r=args.r, k_training=args.k_train,
        #                                             args=args)
        #
        # num_layers = 999
        # min_subgraphs_per_layer = 999

        # 计算一下一个点平均度数
        degree_list = []
        for i in range(len(train_graphs)):
            degree_list.append(train_graphs[i].ecount() / train_graphs[i].vcount())
        print(f"Average Degree：{np.mean(degree_list)}, Upper Bound: {upper_bound}")

        # train_graphs, train_dglgraphs = raw_train_graphs, raw_train_dglgraphs

        # Train models
        for (k, d) in zip([args.k_train], args.d):
            print(f"d: {d}, k: {k}")
            adj_matrices = []
            greedy_perfs = []
            train_closed_graphs = []

            i = 0
            # 这部分是为了计算greedy的baseline
            for train_graph in train_graphs:
                i = i + 1
                # calculate graph closure
                train_closed_graph = bfs(train_graph, d=d)
                train_closed_graphs.append(train_closed_graph)

                # calculate the adj matrix of each graph
                adj_matrices.append(get_adj_mat(train_closed_graph))

                # solve by greedy as baseline
                _, n_covered = greedy(train_closed_graph, k=k)
                greedy_perfs.append(n_covered)

                print(f"Graph:{i}/{len(train_graphs)}, Greedy influence: {n_covered}/{train_graph.vcount()}={n_covered/train_graph.vcount():.2f}.")

            for model_name, seed in product(args.model_name, args.seed):
                print(f"Parameters: model: {model_name}, d: {d}, k: {k}, seed: {seed}")
                torch.manual_seed(seed)  # reproducibility
                param_file = furl("params").add({
                    "model": model_name,
                    "d": d,
                    "seed": seed,
                })

                net = get_model(model_name, *HIDDEN_FEATS)
                if debug:
                    logging.debug("Model loaded. Model info:")
                    logging.debug(net)

                # 把图转成list形式
                edges = []
                closed_edges = []

                for i in range(len(train_graphs)):
                    raw_edges = train_graphs[i].get_edgelist()
                    raw_closed_edges = train_closed_graphs[i].get_edgelist()

                    start_points = [edge[0] for edge in raw_edges]
                    end_points = [edge[1] for edge in raw_edges]
                    closed_start_points = [edge[0] for edge in raw_closed_edges]
                    closed_end_points = [edge[1] for edge in raw_closed_edges]
                    edges.append([start_points, end_points])
                    closed_edges.append([closed_start_points, closed_end_points])

                dataset = GraphDataset(edges)

                loss_list, _, _ = train(
                    net=net,
                    dataset=dataset,
                    loss_function=LOSS_FUNCTION,
                    n_epoch=args.n_epoch,
                    batchsize=args.batchsize,
                    lr=args.lr,
                    k_train=args.k_train,
                    adj_matrices=adj_matrices,
                    greedy_perfs=greedy_perfs,
                    closed_edges=closed_edges,
                    dglgraphs=train_dglgraphs,
                    graphs=train_graphs,
                    closed_graphs=train_closed_graphs,
                    model_name=model_name,
                    d=d,
                    seed=seed,
                    subgraph_size=args.subgraph_size,
                    num_subgraph=args.num_subgraph,
                    score_upper_bound=upper_bound,
                    rate = args.rate,
                    settings = args.settings[0],
                    round = idx,
                    with_dp = args.with_dp,
                    dataset_name = args.dataset[0],
                    target_epsilon = args.target_epsilon,
                    gamma = args.gamma,
                    dir_name = dir_name,
                    gnns_name = gnns_name,
                    K = args.K,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=[0])
    parser.add_argument("--d",default=[1])
    parser.add_argument("--k_train", default=10, type=int)
    parser.add_argument("--batchsize", default=16, type=int)
    parser.add_argument("--lr", default=0.005)
    parser.add_argument("--n_epoch", default=20)
    parser.add_argument("--subgraph_size", default=40, type=int)
    parser.add_argument("--score_upper_bound", default=10, type=int)
    parser.add_argument("--target_epsilon", default=3, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--round", default=1)
    parser.add_argument("--settings", nargs='+', default=["PrivIM+"], help="[PrivIM+, PrivIM, HP, Random]")
    parser.add_argument("--model_name", nargs='+', default=["GRAT3"], help="[GRAT3, DGCN4, GAT4, GIN4]")
    parser.add_argument("--dataset", nargs='+', default=["lastfm"],help="[email, bitcoin, lastfm, hepph, facebook, gowalla]")
    parser.add_argument("--with_dp", default=True)

    parser.add_argument("--sample_mode", default="bfs")
    parser.add_argument("--rate", default=0)
    parser.add_argument("--theta", default=3)
    parser.add_argument("--method", default='ours')
    parser.add_argument("--baseline", default='ours')

    args = parser.parse_args()

    train_models(debug=False)
