import time
import sys

from torch import optim

from util import get_model, get_memory
from load_graph import load_train, load_val, bfs_split
from algo import greedy
import torch
from copy import deepcopy
from tqdm import tqdm
import logging
from pytorchtools import EarlyStopping
import numpy as np
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
import os
sys.path.append(r"D:\demo\Awesome-Differential-Privacy-and-Meachine-Learning-20230731")  # Change the path

from optimizer.dp_optimizer import DPSGD, DPAdam

# ==================== PARAMETERS ====================
PATH_TO_PARAMS = "params/"
USE_CUDA_TRAIN = True
HIDDEN_FEATS = [32, 32, 32, 32, 32, 32]
# ==================== END OF PARAMETERS ====================


# ==================== HELPER FUNCTIONS ====================
def model_file(model_name, path_to_params=PATH_TO_PARAMS):
    time_str = time.strftime("%m-%d", time.localtime())
    return f"{path_to_params}{time_str}{model_name}.pkl"


def get_influence(graph, seeds):
    covered = set()
    for seed in seeds:
        covered.add(int(seed))
        for u in graph.successors(seed):  # Add all the covered seeds
            covered.add(int(u))
    return len(covered)


def get_k_top(net, dglgraph, test_k):
    """Get top k nodes based on the net
    """
    dglgraph_ = deepcopy(dglgraph)
    seeds = set()

    cum_n_covered = 0
    for i in range(test_k):
        # bar.set_description('GetKTop')
        out = net(dglgraph_, dglgraph_.ndata['feat']).squeeze(1)
        _, top_node = torch.max(out, 0)  # index of the best-scored node
        seeds.add(top_node)
        srcs = []  # the src node of the i_th edge
        dsts = []  # the dst node of the i_th edge
        covered = dglgraph_.predecessors(top_node)
        parents = dglgraph_.successors(top_node)

        # add edges (top_node, succ)
        srcs.extend([top_node]*len(parents))
        dsts.extend(parents)

        # add edges (pred, top_node)
        srcs.extend(covered)
        dsts.extend([top_node]*len(covered))

        for node in covered:
            parents = dglgraph_.successors(node)
            srcs.extend([node]*len(parents))
            dsts.extend(parents)
        dglgraph_.remove_edges(dglgraph_.edge_ids(srcs, dsts))

        n_covered = len(covered)
        cum_n_covered += n_covered
        logging.info(f"Node {i+1}: {n_covered}/{cum_n_covered} points are covered.")

    return list(seeds)


def get_x_top(net, dglgraph, test_k, x):  # get top 1/x points.
    d = deepcopy(dglgraph)
    seeds = []

    for _ in range((test_k + x - 1) // x):
        # bar.set_description(' Get'+str(x)+'Top')
        out = net(d, d.ndata['feat']).squeeze(1)
        y = x if test_k >= x else test_k
        test_k -= x
        _, topNodes = torch.topk(out, x)
        seeds.extend(topNodes.tolist())
        srcs = []
        dsts = []
        nodes = []
        nodes.extend(topNodes)
        for topNode in topNodes:
            nodes.extend(d.predecessors(topNode))
        nodes = list(set(nodes))
        for node in nodes:
            for succ in d.successors(node):
                srcs.append(node)
                dsts.append(succ)
        d.remove_edges(d.edge_ids(srcs, dsts))

    seeds = list(set(seeds))
    return seeds


def get_x_top_plus(net, graph, dglgraph, test_k, x):
    graph_ = deepcopy(graph)
    graph_.to_directed()
    seeds = []
    if test_k % x == 0:
        bar = tqdm(range(test_k//x))
        # bar = list(range(test_k//x))
    else:
        bar = tqdm(range(test_k//x+1))
        # bar = list(range(test_k//x+1))
    for i in bar:
        bar.set_description(' Get'+str(x)+'TopPlus')

        out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)

        y = x if test_k >= x else test_k
        test_k -= x

        _, topNodes = torch.topk(out, x)

        # 删边可以有重复，但不能删除不存在的边
        nodes = topNodes.tolist()
        topNodesNum = len(nodes)
        seeds.extend(nodes)
        for i in range(topNodesNum):
            nodes.extend(graph_.successors(nodes[i]))
        nodes = list(set(nodes))

        edges = []
        for node in nodes:
            for pred in graph_.predecessors(node):
                edges.append((pred, node))

        if edges != []:
            srcs, dsts = zip(*edges)
            dglgraph.remove_edges(dglgraph.edge_ids(dsts, srcs))
        graph_.delete_edges(edges)
        dglgraph.ndata['degree'] = torch.tensor(graph.degree()).float()
    seeds = list(set(seeds))
    return seeds


def get_x_top_plus_plus(net, graph, dglgraph, test_k, x):
    dglgraph_ = deepcopy(dglgraph)
    graph_ = deepcopy(graph)
    graph_.to_directed()
    seeds = set()
    if test_k % x == 0:
        bar = tqdm(range(test_k//x))
    else:
        bar = tqdm(range(test_k//x+1))
    for i in bar:
        bar.set_description(' GetXTopPlusPlus')
        out = net(dglgraph_, dglgraph_.ndata['feat']).squeeze(1)
        y = x if test_k >= x else test_k
        test_k -= x
        r = 10  # 查找范围
        _, indices = torch.topk(out, r*x)
        count = 0
        i = 0
        topNodes = []
        while count < x and i < r*x:
            idx = int(indices[i])
            if idx in seeds:
                pass
            else:
                seeds.add(idx)
                topNodes.append(idx)
                count += 1
            i += 1
        # 删边可以有重复，但不能删除不存在的边
        nodes = []
        nodes.extend(topNodes)
        for topNode in topNodes:
            nodes.extend(graph_.successors(topNode))
        nodes = list(set(nodes))
        edges = []
        for node in nodes:
            for pred in graph_.predecessors(node):
                edges.append((pred, node))
        if edges != []:
            srcs, dsts = zip(*edges)
            dglgraph_.remove_edges(dglgraph_.edge_ids(dsts, srcs))
        graph_.delete_edges(edges)
    seeds = list(seeds)
    return seeds
# ==================== END OF HELPER FUNCTIONS ====================

def compute_base_sensitivity(num_message_passing_steps, max_degree, batch_size, num_hops, upper_bound, dp_method):
    """Returns the base sensitivity which is multiplied to the clipping threshold.
    Args:
    """
    if dp_method == "DPGNN4GC":
        max_terms_per_edge = batch_size
    elif dp_method == 'DP_ipd_0617':
        max_terms_per_node = min(upper_bound, batch_size)
        max_terms_per_edge = max_terms_per_node
    return float(2 * max_terms_per_edge)

def account_privacy(num_message_passing_steps,
                    max_node_degree,
                    num_hops,
                    step_num,
                    batch_size,
                    train_num,
                    sigma,
                    orders,
                    target_delta=1e-5,
                    upper_bound=None,
                    ):
    from privacy_analysis.RDP.compute_multiterm_rdp import compute_multiterm_rdp
    from privacy_analysis.RDP.rdp_convert_dp import compute_eps

    # max_terms_per_edge = compute_max_terms_per_edge(num_message_passing_steps,
    #                                                 max_node_degree,
    #                                                 num_hops)
    max_terms_per_node = upper_bound
    max_terms_per_node = min(max_terms_per_node, batch_size)
    # assert max_terms_per_node <= len(train_dataset), "#affected terms must <= #samples"
    rdp_every_epoch = compute_multiterm_rdp(orders, step_num, sigma, train_num,
                                            max_terms_per_node, batch_size)
    # rdp_every_epoch_org = compute_rdp(args.batch_size / len(train_dataset), args.sigma, 1 * epoch, orders)
    epsilon, best_alpha = compute_eps(orders, rdp_every_epoch, target_delta)
    return epsilon, best_alpha


def get_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        step_num: int,
        orders: list,
        num_message_passing_steps,
        max_node_degree,
        num_hops,
        batch_size,
        train_num,
        upper_bound,
        epsilon_tolerance: float = 0.01,
        dp_method="DPGNN4GC",
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    sigma_low, sigma_high = 0, 100
    eps_high, best_alpha = account_privacy(num_message_passing_steps=num_message_passing_steps,
                                           max_node_degree=max_node_degree,
                                           num_hops=num_hops,
                                           step_num=step_num,
                                           batch_size=batch_size,
                                           train_num=train_num,
                                           sigma=sigma_high,
                                           target_delta=target_delta,
                                           orders=orders,
                                           upper_bound=upper_bound)

    if eps_high > target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到100")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        if dp_method == "DPGNN4GC":
            eps, best_alpha = account_privacy_dpsgd(step_num=step_num, target_delta=args.target_delta,
                                                    sigma=args.sigma, orders=orders)
        else:
            eps, best_alpha = account_privacy(num_message_passing_steps=num_message_passing_steps,
                                              max_node_degree=max_node_degree,
                                              num_hops=num_hops,
                                              step_num=step_num,
                                              batch_size=batch_size,
                                              train_num=train_num,
                                              sigma=sigma,
                                              target_delta=target_delta,
                                              orders=orders,
                                              upper_bound=upper_bound)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


def my_collate_fn(batch):
    return{'idx': [item['idx'] for item in batch],
           'edge': [item['edge'] for item in batch]}

def cal_target_delta(dataset_name):
    if dataset_name == "email":
        target_delta_value = 2e-3
    elif dataset_name == "bitcoin":
        target_delta_value = 4e-4
    elif dataset_name == "lastfm":
        target_delta_value = 1e-4
    elif dataset_name == "hepph":
        target_delta_value = 1e-4
    elif dataset_name == "facebook":
        target_delta_value = 9e-5
    elif dataset_name == "gowalla":
        target_delta_value = 1e-5

    return target_delta_value

def train(net, dataset, loss_function, n_epoch, batchsize, lr, k_train=32,
          adj_matrices=None, graphs=None, dglgraphs=None, greedy_perfs=None, closed_edges=None, closed_graphs=None,
          model_name=None, d=None, seed=None, subgraph_size=None, num_subgraph=None,
          score_upper_bound=None, rate=None, settings=None, round=None, with_dp=False, dataset_name=None,
          target_epsilon=None, gamma=None, dir_name=None,gnns_name=None, K=None):

    if USE_CUDA_TRAIN:
        net.cuda()
    print(f"Learning rate: {lr:.2}")

    target_delta_value = cal_target_delta(dataset_name)

    avg_train_losses = []
    avg_train_perfs = []

    # Early stopping
    train_losses=[]
    train_perfs=[]
    early_stopping = EarlyStopping(patience=3, verbose=True)

    # # parmas for dp
    if with_dp:
        target_delta = target_delta_value
        l2_norm_clip = 3
        print(f"Target epsilon: {target_epsilon}, target delta: {target_delta}")

        orders = np.arange(1, 100, 0.1)[1:]
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            step_num=20,
            orders=orders,
            num_message_passing_steps=2,
            max_node_degree=10,
            num_hops=2,
            batch_size=batchsize,
            train_num=len(dataset),
            upper_bound=score_upper_bound,
            dp_method="DP_ipd_0617")
        if settings == 'HP':
            W = np.random.exponential(scale=1.0)
            sigma = np.sqrt(W) * sigma
        print(f"Given target epsilon, sigma is set to:{sigma}")
        sens = compute_base_sensitivity(max_degree=10,
                                        num_message_passing_steps=3,
                                        num_hops=2, batch_size=batchsize,
                                        upper_bound=score_upper_bound,
                                        dp_method='DP_ipd_0617')
        print(f"All the noise: {sens * sigma}")
        optimizer = DPSGD(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=sens * sigma,
            minibatch_size=batchsize,
            microbatch_size=1,
            params=net.parameters(),
            lr=lr,
            momentum=0.9,
        )
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=my_collate_fn)
    timer_list = []
    # best_model_wts = net.state_dict()
    # best_loss = float('inf')

    for epoch in range(n_epoch):
        timer = time.time()
        train_perfs = []
        train_losses = []


        for data in enumerate(train_loader):
            batch_idx = data[1]['idx']
            batch_edges = data[1]['edge']

            batch_adj = []
            batch_graph = []
            batch_dglgraph = []
            batch_greedy_perf = []
            batch_closed_edges = []
            batch_closed_graph = []

            for idx in batch_idx:
                batch_adj.append(adj_matrices[idx])
                batch_graph.append(graphs[idx])
                batch_dglgraph.append(dglgraphs[idx])
                batch_greedy_perf.append(greedy_perfs[idx])
                batch_closed_edges.append(closed_edges[idx])
                batch_closed_graph.append(closed_graphs[idx])


            loss_list = train_single(
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_function=loss_function,
                batch_adj_matrices=batch_adj,
                batch_graphs=batch_graph,
                batch_dglgraphs=batch_dglgraph,
                batch_greedy_perfs=batch_greedy_perf,
                batch_closed_edges=batch_closed_edges,
                batch_closed_graphs=batch_closed_graph,
                k_train=k_train,
                epoch=epoch,
                with_dp=with_dp
            )

            # train_perfs.append(perf)
            train_losses.append(sum(loss_list))
        
        train_loss = np.sum(train_losses)
        avg_train_losses.append(train_loss)
        avg_train_loss = train_loss / len(train_loader.dataset)



        # if train_loss < best_loss:
        #     best_loss = train_loss
        #     best_model_wts = net.state_dict()
        #
        # train_perf = np.average(train_perfs)
        # avg_train_perfs.append(train_perf)

        print(f"Train Epoch {epoch} | Loss: {avg_train_loss:.2f} | Elapsed Time: {time.time() - timer:.2f}")
        timer_list.append(time.time() - timer)

        # early_stopping(train_loss, net)
        # early_stopping(-train_perf, net)
        # if early_stopping.early_stop:
        #     logging.info("Early stopping")
        #     break

    # 保存权重
    if with_dp:
        torch.save(net.state_dict(), f=f"D:/demo/result/0915_0925_multi_results/{dataset_name}_M/eps_{target_epsilon}/n_{subgraph_size}/M_{score_upper_bound}/model/model_{model_name}_d_{d}_seed_{seed}_round_{round}.pt")
    else:
        torch.save(net.state_dict(), f=f"D:/demo/result/0915_0925_multi_results/{dir_name}/eps_10000/model/model_{model_name}_d_{d}_seed_{seed}_round_{round}.pt")
    print(f"Time per epoch: {np.mean(timer_list)}")
    # torch.save(net.state_dict(), f=f"D:/demo/result/d_{d}/kr/size_300_350/K_{K}_r_{r}/num_{num_subgraph}/model/model_{model_name}_d_{d}_seed_{seed}.pt")
    # net.load_state_dict(best_model_wts)
    # torch.save(net.state_dict(), f=f"D:/demo/fastCover-master/output/params/params_model_GRAT3_d_3_seed_42_dp_eps_0.1.pt")
    return avg_train_losses, avg_train_perfs, net

def train_single(net, optimizer, scheduler, loss_function, batch_adj_matrices,
                 batch_graphs, batch_dglgraphs, batch_greedy_perfs,
                 batch_closed_edges, batch_closed_graphs, k_train, epoch, with_dp=True):
    """
    单个batch的训练
    """
    loss_list = []

    adj_mats = batch_adj_matrices
    graphs = batch_graphs
    dglgraphs = batch_dglgraphs
    greedy_perfs = batch_greedy_perfs
    closed_graphs = batch_closed_graphs if batch_closed_graphs is not None else None

    net.train()
    if with_dp:
        optimizer.zero_accum_grad()
        total_loss = 0
        # optimizer.zero_grad()

        for dglgraph, adj_mat, graph, greedy_perf, closed_graph in zip(dglgraphs, adj_mats, graphs, greedy_perfs, closed_graphs):
            optimizer.zero_microbatch_grad()
            out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)
            loss = loss_function(out, adj_mat, k_train)
            # total_loss += loss
            loss.backward()
            optimizer.microbatch_step()
            loss_list.append(loss.item())
        optimizer.step_dp()
        # optimizer.step()
        scheduler.step()
    else:
        # optimizer.zero_accum_grad()
        total_loss = 0
        optimizer.zero_grad()

        for dglgraph, adj_mat, graph, greedy_perf, closed_graph in zip(dglgraphs, adj_mats, graphs, greedy_perfs,
                                                                       closed_graphs):
            # optimizer.zero_microbatch_grad()
            out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)
            loss = loss_function(out, adj_mat, k_train)
            # total_loss += loss
            loss.backward()
            # optimizer.microbatch_step()
            loss_list.append(loss.item())
        # optimizer.step_dp()
        optimizer.step()
        scheduler.step()

    # scheduler.step()
    # total_loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()

    # scheduler.step()


    # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    # print(f"Train epsilon: {epsilon:.2f}, best alpha: {best_alpha}")
    #
    # net.eval()
    # perf_list = []
    # for dglgraph, adj_mat, graph, greedy_perf, closed_graph in zip(dglgraphs, adj_mats, graphs, greedy_perfs, closed_graphs):
    #     logits = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)
    #     _, indices = torch.topk(logits, k_train)
    #
    #     if closed_graph is None:
    #         train_perf = get_influence(graph, indices)
    #     else:
    #         train_perf = get_influence(closed_graph, indices)
    #     perf_ratio = train_perf/greedy_perf
    #     perf_list.append(perf_ratio)
    #     # print(f"Train influence: {train_perf}/{greedy_perf}={perf_ratio:.2f}")

    # final_perf = np.average(perf_list)

    return loss_list

def test_single(net, graph_name, graph, dglgraph, ks):
    n_node = graph.vcount()
    n_edge = graph.ecount()
    print(f"Testing on the graph {graph_name}.")
  
    greedy_ts = []  # timing
    greedy_perfs = []
    nn_t0s = []
    nn_t1s = []
    nn_ts = []
    nn_perfs = []
    memories = []

    for k in ks:
        # greedy
        if k <= n_node:
            t0 = time.time()
            _, greedy_perf = greedy(graph, k)
            t1 = time.time()  # t1 - t0: calculate greedy influence

            out = net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(
        1)
            t2 = time.time()  # t2 - t1: nn
            _, nn_seeds = torch.topk(out, k)
            t3 = time.time()  # t3 - t2: get top k seeds.

            nn_perf = get_influence(graph, nn_seeds)  # we move it out
            memory = get_memory()

            greedy_time = t1 - t0
            nn_time_0 = t2 - t1
            nn_time_1 = t3 - t2
            nn_time = nn_time_0 + nn_time_1

            logging.info(
                f"n_node:{n_node}; n_edge:{n_edge}; k:{k}; perf ratio:{nn_perf/greedy_perf:.2f}; time ratio: {nn_time/greedy_time:.2f}; topk ratio: {nn_time_1/nn_time:.2f}"
            )
        
            greedy_ts.append(greedy_time)
            greedy_perfs.append(greedy_perf)
            nn_t0s.append(nn_time_0)
            nn_t1s.append(nn_time_1)
            nn_ts.append(nn_time)
            nn_perfs.append(nn_perf)
            memories.append(memory)
        else:
            greedy_ts.append(-1)
            greedy_perfs.append(-1)
            nn_t0s.append(-1)
            nn_t1s.append(-1)
            nn_ts.append(-1)
            nn_perfs.append(-1)     
            memories.append(-1)       

    result = {
        "graph_name": graph_name,
        "n": n_node,
        "m": n_edge,
        "k": ks,
        "greedy_perf": greedy_perfs,
        "greedy_time": greedy_ts,
        "nn_perf": nn_perfs,
        "nn_time": nn_ts,
        "nn_time_0": nn_t0s,
        "nn_time_1": nn_t1s,
        "memory": memories,
    }

    return result
