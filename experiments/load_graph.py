import random
import time
from collections import deque

import dgl
import igraph
import networkx as nx
import numpy as np
import torch
from igraph import Graph
from matplotlib import pyplot as plt

from graph_util import get_rev_dgl, gen_erdos_graphs
import os


# ==================== PARAMETERS ====================
USE_CUDA_TRAIN = True
USE_CUDA_TEST = True

# undirected
# DIRECTED_TRAIN = False
# PATH_TO_TRAIN = "dataset/graphs/undirected/"

# DIRECTED_TEST = True
# PATH_TO_TEST = "/home/featurize/dataset/dataset/graphs/directed/"
# TRAIN_LIST = [
#     "fb-pages-tvshow.txt",
#     "fb-pages-politician.txt",
#     "fb-pages-government.txt",
#     "fb-pages-public-figure.txt",
#     # "HepPh.txt",
# ]
# VAL_LIST = []

# directed
DIRECTED_TRAIN = True
DIRECTED_VAL = True
# TRAIN_LIST = [
#     "0.txt","1.txt","2.txt","3.txt","4.txt","5.txt","6.txt","7.txt","8.txt","9.txt",
#     "10.txt","11.txt","12.txt","13.txt","14.txt","15.txt","16.txt","17.txt","18.txt","19.txt"
# ]

# TRAIN_LIST = [
#     "soc-email.txt"
# ]

VAL_LIST = [
    "soc-epinions.txt",
    "soc-anybeat.txt",
]


FEATURE_TYPE = "1"
N_VAL_GRAPH = 5
N_TEST_NODE = 1000
# ==================== END OF PARAMETERS ====================


def load_train(input_dim, dataset, directed_train=DIRECTED_TRAIN,
               feature_type=FEATURE_TYPE, use_cuda=USE_CUDA_TRAIN,
               path_to_train=PATH_TO_TRAIN):
    graphs = []
    dglgraphs = []
    for i in range(1):
        ############################## dataset ################################
        if dataset == "email":
            file = f"graphs/directed/trainset/email-Eu-core_train.txt"
        elif dataset == "bitcoin":
            file = f"graphs/directed/trainset/bitcoinotc_train.txt"
        elif dataset == "lastfm":
            file = f"graphs/directed/trainset/di_lastfm_asia_train.txt"
        elif dataset == "hepph":
            file = f"graphs/directed/trainset/di_hepph_train.txt"
        elif dataset == "facebook":
            file = f"graphs/directed/trainset/di_Facebook_train.txt"
        elif dataset == "gowalla":
            file = f"graphs/directed/trainset/di_gowalla_train.txt"
        #######################################################################

        g = igraph.Graph().Read_Edgelist(
            f"{path_to_train}{file}", directed=directed_train)
        dg = get_rev_dgl(g, feature_type, input_dim, directed_train, use_cuda)
        graphs.append(g)
        dglgraphs.append(dg)

    return graphs, dglgraphs

def load_train_2000(input_dim, directed_train=DIRECTED_TRAIN, feature_type=FEATURE_TYPE, use_cuda=USE_CUDA_TRAIN, path_to_train=PATH_TO_TRAIN):
    graphs = []
    dglgraphs = []
    for file in TRAIN_LIST:
        g = igraph.Graph().Read_Edgelist(
            f"{path_to_train}{file}", directed=directed_train)
        dg = get_rev_dgl(g, feature_type, input_dim, directed_train, use_cuda)
        graphs.append(g)
        dglgraphs.append(dg)

    return graphs, dglgraphs

def load_val(input_dim, val_list=VAL_LIST, directed_val=DIRECTED_VAL, feature_type=FEATURE_TYPE, use_cuda=USE_CUDA_TRAIN, path_to_val=PATH_TO_VAL):
    graphs=[]
    dglgraphs=[]
    for file in val_list:
        g = igraph.Graph().Read_Edgelist(
            f"{path_to_val}{file}", directed=directed_val)
        dg = get_rev_dgl(g, feature_type, input_dim, directed_val, use_cuda)
        graphs.append(g)
        dglgraphs.append(dg)

    return val_list, graphs, dglgraphs

def limit_indegree(graph, dglgraph, K):
    for node in graph.vs:
        predecessors = graph.predecessors(node)
        if len(predecessors) > K:
            excess_predecessors = predecessors[:len(predecessors) - K]
            edges_to_remove = [edge.index for edge in graph.es if edge.target==node.index and edge.source in excess_predecessors]
            graph.delete_edges(edges_to_remove)
            dglgraph = dgl.remove_edges(dglgraph, edges_to_remove)
    return graph, dglgraph


def limit_outdegree(graph, dglgraph, K):
    for node in graph.vs:
        if len(graph.successors(node)) > K:
            for edge in graph.es:
                if edge.source == node.index:
                    graph.delete_edges(edge.index)
                    dglgraph.remove_edges(edge.index)
    return graph, dglgraph

def sample_out_edges(graph, dglgraph, K):
    edges_to_remove = []
    for node in graph.vs:
        out_edges = [edge for edge in graph.es if edge.source == node.index and edge.target != node.index]
        num_out_edges = len(out_edges)

        if num_out_edges > 0:
            p = min(K / (2 * num_out_edges), 1)
            edges_to_keep = random.sample(out_edges, int(num_out_edges * p))
            for edge in out_edges:
                if edge not in edges_to_keep:
                    edges_to_remove.append(edge.index)

    graph.delete_edges(edges_to_remove)
    dglgraph.remove_edges(edges_to_remove)

    return graph, dglgraph

def bfs_split_dp(graphs, dglgraphs, num_subgraphs, min_node_threshold, K, r, k_training):
    split_graphs = []
    split_dglgraphs = []
    min_subgraphs_per_layer = 0
    num_layers = 0

    graph_index = random.randint(0, len(graphs) - 1)  # 随机选择一个图
    graph = graphs[graph_index]
    dglgraph = dglgraphs[graph_index]
    graph, dglgraph = sample_out_edges(graph, dglgraph, K)
    graph, dglgraph = limit_indegree(graph, dglgraph, K)

    total_nodes = len(graph.vs)

    unused_nodes = list(range(len(graph.vs)))  # 未使用的节点
    used_nodes = set()

    while len(split_graphs) < num_subgraphs:
        current_layer_subgraphs = []
        current_layer_dglsubgraphs = []
        current_layer_unused_nodes = unused_nodes.copy()
        layer_attempts = 0

        while current_layer_unused_nodes:
            start_node = random.choice(current_layer_unused_nodes)  # 在选定的图中随机选择一个起始节点

            queue = deque([(start_node, 0)])  # (节点, 深度)
            visited = set([start_node])
            level_nodes = {0: [start_node]}

            while queue:
                current_node, depth = queue.popleft()

                if depth <= r:
                    neighbors = list(graph.successors(current_node))
                    neighbors = sorted(neighbors, key=lambda x: len(graph.successors(x)))

                    for neighbor in neighbors:
                        if neighbor not in visited and neighbor in current_layer_unused_nodes:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))
                            if depth + 1 not in level_nodes:
                                level_nodes[depth + 1] = []
                            level_nodes[depth + 1].append(neighbor)

                if len(visited) >= 300:
                    break

            if 300 <= len(visited) <= 350:
                nodes_to_include = list(visited)

                if not any(node in used_nodes for node in nodes_to_include):
                    subgraph = graph.subgraph(nodes_to_include)
                    subdglgraph = dglgraph.subgraph(nodes_to_include)

                    current_layer_subgraphs.append(subgraph)
                    current_layer_dglsubgraphs.append(subdglgraph)
                    current_layer_unused_nodes = [node for node in current_layer_unused_nodes if node not in nodes_to_include]
                    layer_attempts = 0
                else:
                    layer_attempts += 1
            else:
                layer_attempts += 1

            if layer_attempts >= 100:
                break

            if len(current_layer_unused_nodes) < 300:
                num_layers += 1
                if min_subgraphs_per_layer == 0 or len(current_layer_subgraphs) < min_subgraphs_per_layer:
                    min_subgraphs_per_layer = len(current_layer_subgraphs)
                split_graphs.extend(current_layer_subgraphs)
                split_dglgraphs.extend(current_layer_dglsubgraphs)
                if len(split_graphs) >= num_subgraphs:
                    split_graphs = split_graphs[:num_subgraphs]
                    split_dglgraphs = split_dglgraphs[:num_subgraphs]
                break

    print(f"Number of layers: {num_layers}")
    print(f"Min subgraphs per layer: {min_subgraphs_per_layer}")

    return split_graphs, split_dglgraphs, num_layers, min_subgraphs_per_layer

def reduce_out_degree(dglgraph, threshold):
    out_degrees = dglgraph.out_degrees()

    for node in range(dglgraph.number_of_nodes()):
        if out_degrees[node] > threshold:
            successors = dglgraph.successors(node).tolist()
            edges_to_remove = out_degrees[node] - threshold
            if edges_to_remove > len(successors):
                edges_to_remove = len(successors)
            edges_to_remove = random.sample(successors, edges_to_remove)
            dglgraph.remove_edges(edges_to_remove)

    nodes = dglgraph.nodes().to('cpu').numpy()
    edges = dglgraph.edges()

    graph = igraph.Graph(directed=True)
    graph.add_vertices(len(nodes))
    for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
        graph.add_edge(src.item(), dst.item())  # 转换为 int

    return graph, dglgraph

def reduce_in_degree(dglgraph, threshold):
    in_degrees = dglgraph.in_degrees()

    for node in range(dglgraph.number_of_nodes()):
        if in_degrees[node] > threshold:
            predecessors = dglgraph.predecessors(node).tolist()
            edges_to_remove = in_degrees[node] - threshold
            if edges_to_remove > len(predecessors):
                edges_to_remove = len(predecessors)
            edges_to_remove = random.sample(predecessors, edges_to_remove)
            dglgraph.remove_edges(edges_to_remove)

    nodes = dglgraph.nodes().to('cpu').numpy()
    edges = dglgraph.edges()

    graph = igraph.Graph(directed=True)
    graph.add_vertices(len(nodes))
    for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
        graph.add_edge(src.item(), dst.item())  # 转换为 int

    return graph, dglgraph


from igraph import Graph


def calculate_subgraph_score(subgraph, node_map, all_subgraphs, nodes_score):
    score = 0
    original_nodes = node_map.values()
    for node in original_nodes:
        node_count_in_others = sum([1 for sg_item in all_subgraphs if node in sg_item['node_map'].values()])-1
        score += (1 + nodes_score[node][1]) * node_count_in_others
    return score


def adjust_subgraphs(current_layer_random_node_map_list, current_layer_node_map_list, node_score, num_layers):
    # Step 1: Calculate scores for all subgraphs
    current_scores = [
        calculate_subgraph_score(item['subgraph'], item['node_map'], current_layer_node_map_list, node_score)
        for item in current_layer_node_map_list]
    random_scores = [
        calculate_subgraph_score(item['subgraph'], item['node_map'], current_layer_random_node_map_list, node_score)
        for item in current_layer_random_node_map_list]

    # Step 2: Replace subgraph if necessary
    min_current_score = min(current_scores)
    max_current_score = max(current_scores)

    for rand_item, rand_score in zip(current_layer_random_node_map_list, random_scores):
        if rand_score < min_current_score:
            # Find index of the subgraph with the highest score to remove
            max_score_index = current_scores.index(max_current_score)
            # Replace subgraph in current_layer_subgraphs
            current_layer_node_map_list[max_score_index] = rand_item
            current_scores[max_score_index] = rand_score
            max_current_score = max(current_scores)  # Update max score


    # Step 3: Remove subgraphs if node usage exceeds threshold
    final_subgraph_node_map_list = []
    current_layer_node_map_list = sorted(current_layer_node_map_list, key=lambda x: calculate_subgraph_score(x['subgraph'], x['node_map'], current_layer_node_map_list, node_score))
    for item in current_layer_node_map_list:
        exceeds_threshold = False
        for node in item['node_map'].values():
            if node_score[node][1] > num_layers:
                exceeds_threshold = True
                break
        if not exceeds_threshold:
            for node in item['node_map'].values():
                node_score[node][1] += 1
            final_subgraph_node_map_list.append(item)

    return final_subgraph_node_map_list, node_score

def start_nodes_sample_rate(total_nodes):
    q_b = 256 / total_nodes
    return q_b

def connect_node_to_graph(visited, dglgraph):
    nodes_to_include = list(visited)
    subdglgraph = dglgraph.subgraph(nodes_to_include)

    node_map = {i: nodes_to_include[i] for i in range(len(nodes_to_include))}

    nodes = subdglgraph.nodes().to('cpu').numpy()
    edges = subdglgraph.edges()

    subgraph = igraph.Graph(directed=True)
    subgraph.add_vertices(len(nodes))
    for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
        subgraph.add_edge(src.item(), dst.item())  # 转换为 int

    return subgraph, subdglgraph

def remove_edges_exceeding_degree(dglgraph, max_degree, is_directed):
    out_degrees = dglgraph.out_degrees()

    nodes_exceeding_degree = torch.nonzero(out_degrees > max_degree).squeeze(1)
    edges_to_remove = []
    reverse_edges_to_remove = []

    for node in nodes_exceeding_degree:
        src, dst, eid = dglgraph.out_edges(node, form='all')
        num_edges_to_remove = out_degrees[node] - max_degree

        edges_to_remove.extend(eid[:num_edges_to_remove].tolist())

        if not is_directed:
            dst_to_remove = dst[:num_edges_to_remove]
            reverse_eids = dglgraph.edge_ids(dst_to_remove, node)
            reverse_edges_to_remove.extend(reverse_eids.tolist())

    if is_directed:
        dglgraph = dgl.remove_edges(dglgraph, edges_to_remove)
    if not is_directed:
        all_removed_edges = list(set(edges_to_remove + reverse_edges_to_remove))
        dglgraph = dgl.remove_edges(dglgraph, all_removed_edges)

    return dglgraph

def judge_directed_or_not(dataset) -> bool:
    if dataset[0] in ["email", "bitcoin"]:
        return True
    else:
        return False

def transfer_dgl_to_igraph(filtered_dglgraph):
    isolated_nodes_dgl = filtered_dglgraph.in_degrees() + filtered_dglgraph.out_degrees() == 0
    filtered_dglgraph = dgl.remove_nodes(filtered_dglgraph,
                                         isolated_nodes_dgl.nonzero(as_tuple=True)[0].tolist())
    nodes = filtered_dglgraph.nodes().to('cpu').numpy()
    edges = filtered_dglgraph.edges()

    filtered_graph = igraph.Graph(directed=True)
    filtered_graph.add_vertices(len(nodes))

    edge_list = list(zip(edges[0].tolist(), edges[1].tolist()))
    filtered_graph.add_edges(edge_list)

    return filtered_dglgraph, filtered_graph

def random_walking_with_restart(graphs, dglgraphs, args):
    ############# 初始化 #############
    split_subgraphs = []
    split_subdglgraphs = []
    max_attempts = 100
    restart_prob = 0.3

    max_degree = args.K
    max_hops = args.r
    subgraph_size = args.subgraph_size
    settings = args.settings[0]
    model_name = args.model_name[0]
    score_upper_bound = args.score_upper_bound
    directed_or_not = judge_directed_or_not(args.dataset)
    print(f"################## General Settings ##################\n"
          f"Dataset Name: {args.dataset[0]}, Directed : {directed_or_not}\n"
          f"Model Name: {model_name}, Settings: {settings},\n"
          f"Max Degree: {max_degree}, Max Hops: {max_hops},\n"
          f"Subgraph Size: {subgraph_size}\n")

    ############# 删除孤立点和自环结构 #############
    graph_index = random.randint(0, len(graphs) - 1)  # 随机选择一个图
    graph = graphs[graph_index]
    dglgraph = dglgraphs[graph_index]
    # 删掉自环
    selfloop_edges = [edge for edge in graph.es if edge.source == edge.target]
    graph.delete_edges(selfloop_edges)
    dglgraph = dgl.remove_self_loop(dglgraph)

    isolated_nodes = [v.index for v in graph.vs if v.degree() == 0]
    graph.delete_vertices(isolated_nodes)

    isolated_nodes_dgl = dglgraph.in_degrees() + dglgraph.out_degrees() == 0
    dglgraph = dgl.remove_nodes(dglgraph, isolated_nodes_dgl.nonzero(as_tuple=True)[0].tolist())

    total_nodes = len(graph.vs)
    total_edges = len(graph.es)

    # 根据训练集点数确定采样率
    q_b = start_nodes_sample_rate(total_nodes)
    target_subgraph_count = int(q_b * total_nodes)

    print(f"################## Graph Info ##################\n"
          f"Total Nodes: {total_nodes}, Total Edges: {total_edges}\n"
          f"q_b: {q_b}, Target Subgraph Number: {target_subgraph_count}\n")

    ############# 带有返回原点的随机游走 #############
    # 用一个二维数组存储每个点以及被使用的次数
    unused_nodes = list(range(len(graph.vs)))  # 未使用的节点
    node_score = [[node, 0] for node in range(len(graph.vs))]
    failed_nodes = set()

    subgraphs_collected = 0
    ########################### Settings == "PrivIM" ###########################
    if settings == "PrivIM":
        # 开始计时
        time_start = time.time()
        while subgraphs_collected < target_subgraph_count:
            # 从得分非最高的节点中随机选择一个作为起始节点
            start_node_candidates = []
            count = 0
            for node, score in node_score:
                if score != score_upper_bound and node not in failed_nodes:
                    start_node_candidates.append(node)
                    count += 1
                if count >= 1000:
                    break
            if (not start_node_candidates or
                    min([score for node, score in node_score])== score_upper_bound):
                print("No start node candidates left.")
                break

            # 随机选择一个起始节点
            start_node = random.choice(start_node_candidates)
            attempts = 0
            current_node = start_node
            visited = set()

            while len(visited) < subgraph_size and attempts < max_attempts:
                if random.random() < restart_prob:
                    current_node = start_node

                neighbors = list(set(graph.neighbors(current_node)))
                filtered_neighbors = [neighbor for neighbor in neighbors
                                      if node_score[neighbor][1] < score_upper_bound]
                if not filtered_neighbors:
                    break

                neighbor_scores = np.array([node_score[filtered_neighbor][1]
                                            for filtered_neighbor in filtered_neighbors])
                if neighbor_scores.sum() == 0:
                    probabilities = np.ones_like(neighbor_scores) / len(filtered_neighbors)
                else:
                    probabilities = (1 / (neighbor_scores + 1)) / np.sum(1 / (neighbor_scores + 1))

                # 采样下一个节点
                next_node = np.random.choice(filtered_neighbors, p=probabilities)

                if next_node not in visited:
                    visited.add(next_node)
                    current_node = next_node
                    attempts = 0
                else:
                    current_node = next_node
                    attempts += 1

            if len(visited) == subgraph_size:
                subgraph, subdglgraph = connect_node_to_graph(visited, dglgraph)
                split_subgraphs.append(subgraph)
                split_subdglgraphs.append(subdglgraph)
                subgraphs_collected += 1
                for node in visited:
                    node_score[node][1] += 1
            else:
                failed_nodes.add(start_node)
        # 计时结束
        time_end = time.time()
        print(f"PrivIM Sampling Time Cost: {time_end - time_start:.2f}s")

    ########################### Settings == "PrivIM+" ###########################
    elif settings == "PrivIM+":
        # 开始计时
        time_start = time.time()
        while subgraphs_collected < target_subgraph_count:
            # 从得分非最高的节点中随机选择一个作为起始节点
            start_node_candidates = []
            count = 0
            for node, score in node_score:
                if score != score_upper_bound and node not in failed_nodes:
                    start_node_candidates.append(node)
                    count += 1
                if count >= 1000:
                    break
            if (not start_node_candidates or
                    min([score for node, score in node_score])== score_upper_bound):
                print("No start node candidates left.")
                break

            # 随机选择一个起始节点
            start_node = random.choice(start_node_candidates)
            attempts = 0
            current_node = start_node
            visited = set()

            while len(visited) < subgraph_size and attempts < max_attempts:
                if random.random() < restart_prob:
                    current_node = start_node

                neighbors = list(set(graph.neighbors(current_node)))
                filtered_neighbors = [neighbor for neighbor in neighbors
                                      if node_score[neighbor][1] < score_upper_bound]
                if not filtered_neighbors:
                    break

                neighbor_scores = np.array([node_score[filtered_neighbor][1]
                                            for filtered_neighbor in filtered_neighbors])
                if neighbor_scores.sum() == 0:
                    probabilities = np.ones_like(neighbor_scores) / len(filtered_neighbors)
                else:
                    probabilities = (1 / (neighbor_scores + 1)) / np.sum(1 / (neighbor_scores + 1))

                # 采样下一个节点
                next_node = np.random.choice(filtered_neighbors, p=probabilities)

                if next_node not in visited:
                    visited.add(next_node)
                    current_node = next_node
                    attempts = 0
                else:
                    current_node = next_node
                    attempts += 1

            if len(visited) == subgraph_size:
                subgraph, subdglgraph = connect_node_to_graph(visited, dglgraph)
                split_subgraphs.append(subgraph)
                split_subdglgraphs.append(subdglgraph)
                subgraphs_collected += 1
                for node in visited:
                    node_score[node][1] += 1
            else:
                failed_nodes.add(start_node)

        # 大图采样完成，开始采样小图
        print(f"Large Subgraphs Collected: {subgraphs_collected}")
        failed_nodes = set()
        start_node_candidates = []
        minigraphs_collected = 0
        while minigraphs_collected < target_subgraph_count / 4:
            minigraph_size = int(subgraph_size / 2)
            # 从得分非最高的节点中随机选择一个作为起始节点
            start_node_candidates = []
            count = 0
            for node, score in node_score:
                if score != score_upper_bound and node not in failed_nodes:
                    start_node_candidates.append(node)
                    count += 1
                if count >= 1000:
                    break
            if (not start_node_candidates or
                    min([score for node, score in node_score])== score_upper_bound):
                print("No start node candidates left.")
                break

            start_node = random.choice(start_node_candidates)
            attempts = 0
            current_node = start_node
            visited = set()
            while len(visited) < minigraph_size and attempts < max_attempts:
                if random.random() < restart_prob:
                    current_node = start_node

                neighbors = list(set(graph.neighbors(current_node)))
                filtered_neighbors = [neighbor for neighbor in neighbors
                                      if node_score[neighbor][1] < score_upper_bound]
                if not filtered_neighbors:
                    break

                neighbor_scores = np.array([node_score[filtered_neighbor][1]
                                            for filtered_neighbor in filtered_neighbors])
                if neighbor_scores.sum() == 0:
                    probabilities = np.ones_like(neighbor_scores) / len(filtered_neighbors)
                else:
                    probabilities = (1 / (neighbor_scores + 1)) / np.sum(1 / (neighbor_scores + 1))

                # 采样下一个节点
                next_node = np.random.choice(filtered_neighbors, p=probabilities)

                if next_node not in visited:
                    visited.add(next_node)
                    current_node = next_node
                    attempts = 0
                else:
                    current_node = next_node
                    attempts += 1

            if len(visited) == minigraph_size:
                subgraph, subdglgraph = connect_node_to_graph(visited, dglgraph)
                split_subgraphs.append(subgraph)
                split_subdglgraphs.append(subdglgraph)
                subgraphs_collected += 1
                minigraphs_collected += 1
                for node in visited:
                    node_score[node][1] += 1
            else:
                failed_nodes.add(start_node)
        print(f"Small Subgraphs Collected: {minigraphs_collected}")
        # 计时结束
        time_end = time.time()
        print(f"PrivIM+ Sampling Time Cost: {time_end - time_start:.2f}s")
    ########################### Settings == "Random" ###########################
    return split_subgraphs, split_subdglgraphs
def rate_split_dp(graphs, dglgraphs, num_subgraphs, min_node_threshold, K, r, k_training, args):
    split_graphs = []
    split_dglgraphs = []
    min_subgraphs_per_layer = 0
    num_layers = 0

    graph_index = random.randint(0, len(graphs) - 1)  # 随机选择一个图
    graph = graphs[graph_index]
    dglgraph = dglgraphs[graph_index]
    # 删掉自环
    selfloop_edges = [edge for edge in graph.es if edge.source == edge.target]
    graph.delete_edges(selfloop_edges)
    dglgraph = dgl.remove_self_loop(dglgraph)

    isolated_nodes = [v.index for v in graph.vs if v.degree() == 0]
    graph.delete_vertices(isolated_nodes)

    isolated_nodes_dgl = dglgraph.in_degrees() + dglgraph.out_degrees() == 0
    dglgraph = dgl.remove_nodes(dglgraph, isolated_nodes_dgl.nonzero(as_tuple=True)[0].tolist())

    # graph, dglgraph = sample_out_edges(graph, dglgraph, K)
    # graph, dglgraph = limit_indegree(graph, dglgraph, K)

    total_nodes = len(graph.vs)

    unused_nodes = list(range(len(graph.vs)))  # 未使用的节点
    # 用一个二维数组存储每个点以及被使用的次数
    node_score = [[node, 0] for node in range(len(graph.vs))]

    used_nodes = set()
    sub_size = args.subgraph_size  # 初始子图大小
    count = 0  # 计数器
    switch_count = int((args.rate / 100) * num_subgraphs)
    large_count = 0 # 计数器，用于记录大子图的数量

    # 下面两行是为了计算出度，为heter服务
    out_degrees = np.array(graph.outdegree())
    sample_rate = 1.0/ (out_degrees + 1e-6)

    split_sub_subgraphs = []
    split_sub_dglsubgraphs = []

    while large_count < num_subgraphs and unused_nodes:
        current_layer_subgraphs = []
        current_layer_dglsubgraphs = []

        random_subgraphs = []
        random_dglsubgraphs = []

        current_layer_node_map_list = []
        current_layer_random_node_map_list = []
        final_node_map_list = []

        current_layer_unused_nodes = unused_nodes.copy()
        layer_attempts = 0
        mini_num = 0
        count = 0
        if count >= switch_count:
            sub_size = args.subgraph_size

        while current_layer_unused_nodes:
            if count >= switch_count:
                sub_size = args.subgraph_size
            mini_size = int(sub_size / 2)
            start_node = random.choice(current_layer_unused_nodes)
            if args.sample_mode == "bfs":
                visited, bfs_edges = bfs_sample(graph, start_node, sub_size, r, current_layer_unused_nodes)
            else:
                visited = dfs_sample(graph, start_node, sub_size, r, current_layer_unused_nodes)

            if len(visited) >= sub_size:
                nodes_to_include = list(visited)
                if not any(node in used_nodes for node in nodes_to_include):
                    subdglgraph = dglgraph.subgraph(nodes_to_include)

                    node_map = {i: nodes_to_include[i] for i in range(len(nodes_to_include))}

                    subdglgraph_bfs_edges = [(list(node_map.keys())[list(node_map.values()).index(src)],
                                              list(node_map.keys())[list(node_map.values()).index(dst)])
                                             for src, dst in bfs_edges if
                                             src in node_map.values() and dst in node_map.values()]

                    subdglgraph_edges = set(zip(subdglgraph.edges()[0].tolist(), subdglgraph.edges()[1].tolist()))
                    bfs_edge_set = set(subdglgraph_bfs_edges)
                    remaining_edges = subdglgraph_edges - bfs_edge_set

                    ##############################  以下代码是为了做消融实验  ##############################
                    if len(subdglgraph_edges) > (args.theta * args.subgraph_size):
                        num_edges_to_remove = int(len(subdglgraph_edges) - (args.theta * args.subgraph_size))
                        edges_to_remove = random.sample(remaining_edges, num_edges_to_remove)
                        edges_to_remove_id = subdglgraph.edge_ids(*zip(*edges_to_remove))
                        subdglgraph.remove_edges(edges_to_remove_id)

                    num_edges_to_remove = int(subdglgraph.number_of_edges() * 0)  # 删除 30% 的边
                    edges_to_remove = random.sample(range(subdglgraph.number_of_edges()), num_edges_to_remove)
                    subdglgraph.remove_edges(edges_to_remove)
                    ##############################  以上代码是为了做消融实验  ##############################

                    nodes = subdglgraph.nodes().to('cpu').numpy()
                    edges = subdglgraph.edges()

                    subgraph = igraph.Graph(directed=True)
                    subgraph.add_vertices(len(nodes))
                    for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
                        subgraph.add_edge(src.item(), dst.item())  # 转换为 int

                    if (args.theta * args.subgraph_size) - 1 <= subdglgraph.number_of_edges():
                        current_layer_subgraphs.append(subgraph)
                        current_layer_dglsubgraphs.append(subdglgraph)
                        current_layer_unused_nodes = [node for node in current_layer_unused_nodes if node not in nodes_to_include]
                        count += 1
                        layer_attempts = 0
                        current_layer_node_map_list.append({'subgraph': subgraph,'subdglgraph': subdglgraph, 'node_map': node_map})
                else:
                    layer_attempts += 1
            else:
                layer_attempts += 1


            if len(current_layer_unused_nodes) < (sub_size/2) or layer_attempts >= 200 or len(current_layer_subgraphs) >= num_subgraphs:
                # 此时采集不重叠子图任务完成，接下来随机采集子图，用于调整不重叠子图

                while len(random_subgraphs) < len(current_layer_subgraphs) and unused_nodes:
                    sub_size = args.subgraph_size
                    start_node = random.choice(unused_nodes)  # 在选定的图中随机选择一个起始节点
                    # unused_nodes.remove(start_node)  # 从未使用的节点中移除起始节点

                    if args.sample_mode == "bfs":
                        visited, bfs_edges = bfs_sample_random(graph, start_node, sub_size, r, set())
                    else:
                        visited = dfs_sample(graph, start_node, sub_size, r, set())

                    if len(visited) >= sub_size:
                        nodes_to_include = list(visited)
                        # 有向子图提取
                        # subgraph = graph.subgraph(nodes_to_include)
                        subdglgraph = dglgraph.subgraph(nodes_to_include)
                        node_map = {i: nodes_to_include[i] for i in range(len(nodes_to_include))}

                        subdglgraph_bfs_edges = [(list(node_map.keys())[list(node_map.values()).index(src)],
                                                  list(node_map.keys())[list(node_map.values()).index(dst)])
                                                 for src, dst in bfs_edges if
                                                 src in node_map.values() and dst in node_map.values()]

                        subdglgraph_edges = set(zip(subdglgraph.edges()[0].tolist(), subdglgraph.edges()[1].tolist()))
                        bfs_edge_set = set(subdglgraph_bfs_edges)
                        remaining_edges = subdglgraph_edges - bfs_edge_set

                        # if len(subdglgraph_edges) > (args.theta * args.subgraph_size):
                        #     num_edges_to_remove = len(subdglgraph_edges) - (args.theta * args.subgraph_size)
                        #     edges_to_remove = random.sample(remaining_edges, num_edges_to_remove)
                        #     edges_to_remove_id = subdglgraph.edge_ids(*zip(*edges_to_remove))
                        #     subdglgraph.remove_edges(edges_to_remove_id)

                        # num_edges_to_remove = int(subdglgraph.number_of_edges() * 0)  # 删除 30% 的边
                        # edges_to_remove = random.sample(range(subdglgraph.number_of_edges()), num_edges_to_remove)
                        # subdglgraph.remove_edges(edges_to_remove)

                        nodes = subdglgraph.nodes().to('cpu').numpy()
                        edges = subdglgraph.edges()

                        subgraph = igraph.Graph(directed=True)
                        subgraph.add_vertices(len(nodes))
                        for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
                            subgraph.add_edge(src.item(), dst.item())  # 转换为 int

                        if (args.theta * args.subgraph_size) - 1 <= subdglgraph.number_of_edges():
                            random_dglsubgraphs.append(subdglgraph)
                            random_subgraphs.append(subgraph)
                            current_layer_random_node_map_list.append({'subgraph': subgraph, 'subdglgraph': subdglgraph, 'node_map': node_map})

                final_node_map_list, node_score = adjust_subgraphs(current_layer_random_node_map_list, current_layer_node_map_list, node_score, num_layers)
                current_layer_subgraphs = [item['subgraph'] for item in final_node_map_list]
                current_layer_dglsubgraphs = [item['subdglgraph'] for item in final_node_map_list]

                num_layers += 1
                if min_subgraphs_per_layer == 0 or len(current_layer_subgraphs) < min_subgraphs_per_layer:
                    min_subgraphs_per_layer = len(current_layer_subgraphs)-mini_num
                split_graphs.extend(current_layer_subgraphs)
                split_dglgraphs.extend(current_layer_dglsubgraphs)
                large_count += count
                if large_count >= num_subgraphs:
                    if args.method == "ours":
                        split_graphs = split_graphs[:num_subgraphs]
                        split_dglgraphs = split_dglgraphs[:num_subgraphs]
                    elif args.method == "ours_plus":
                        split_graphs = split_graphs[:num_subgraphs] + split_sub_subgraphs
                        split_dglgraphs = split_dglgraphs[:num_subgraphs] + split_sub_dglsubgraphs
                break

    appearances = [item[1] for item in node_score]
    max_value = max(appearances)

    used_index = []
    unused_index = []
    for i, value in enumerate(appearances):
        if value == max_value:
            used_index.append(i)
        else:
            unused_index.append(i)

    if args.method == "ours_plus":
        mini_size = int(args.subgraph_size / 2)
        split_sub_subgraphs = []
        split_sub_dglsubgraphs = []
        current_layer_unused_nodes = unused_index.copy()
        layer_attempts = 0
        current_layer_node_map_list = []
        # 采集一半大小的子图
        while current_layer_unused_nodes and len(current_layer_node_map_list) <= 0.1 * num_subgraphs:
            start_node = random.choice(current_layer_unused_nodes)
            if args.sample_mode == "bfs":
                visited, bfs_edges = bfs_sample(graph, start_node, mini_size, r, current_layer_unused_nodes)
            else:
                visited = dfs_sample(graph, start_node, mini_size, r, current_layer_unused_nodes)
            if len(visited) >= mini_size:
                nodes_to_include = list(visited)
                if not any(node in used_nodes for node in nodes_to_include):
                    subdglgraph = dglgraph.subgraph(nodes_to_include)

                    node_map = {i: nodes_to_include[i] for i in range(len(nodes_to_include))}

                    subdglgraph_bfs_edges = [(list(node_map.keys())[list(node_map.values()).index(src)],
                                              list(node_map.keys())[list(node_map.values()).index(dst)])
                                             for src, dst in bfs_edges if
                                             src in node_map.values() and dst in node_map.values()]

                    subdglgraph_edges = set(zip(subdglgraph.edges()[0].tolist(), subdglgraph.edges()[1].tolist()))

                    bfs_edge_set = set(subdglgraph_bfs_edges)
                    remaining_edges = subdglgraph_edges - bfs_edge_set

                    nodes = subdglgraph.nodes().to('cpu').numpy()
                    edges = subdglgraph.edges()

                    subgraph = igraph.Graph(directed=True)
                    subgraph.add_vertices(len(nodes))
                    for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
                        subgraph.add_edge(src.item(), dst.item())  # 转换为 int

                    if ((args.theta * args.subgraph_size) / 2) - 1 <= subdglgraph.number_of_edges():
                        current_layer_unused_nodes = [node for node in current_layer_unused_nodes if
                                                    node not in nodes_to_include]
                        mini_num += 1
                        layer_attempts = 0
                        current_layer_node_map_list.append({'subgraph': subgraph, 'subdglgraph': subdglgraph, 'node_map': node_map})
                else:
                    layer_attempts += 1
            else:
                layer_attempts += 1
            if layer_attempts >= 50:
                break

        # 检查是否有子图的节点度数超过阈值
        final_subgraph_node_map_list = []
        current_layer_node_map_list = sorted(current_layer_node_map_list,
                                             key=lambda x: calculate_subgraph_score(x['subgraph'], x['node_map'],
                                                                                    current_layer_node_map_list,
                                                                                    node_score))
        for item in current_layer_node_map_list:
            exceeds_threshold = False
            for node in item['node_map'].values():
                if node_score[node][1] >= max(appearances):
                    exceeds_threshold = True
                    break
            if not exceeds_threshold:
                for node in item['node_map'].values():
                    node_score[node][1] += 1
                final_subgraph_node_map_list.append(item)

        split_sub_subgraphs = [item['subgraph'] for item in final_subgraph_node_map_list]
        split_sub_dglsubgraphs = [item['subdglgraph'] for item in final_subgraph_node_map_list]
        split_graphs = split_graphs + split_sub_subgraphs[:len(split_graphs)]
        split_dglgraphs = split_dglgraphs + split_sub_dglsubgraphs[:len(split_dglgraphs)]

    appearances_new = [item[1] for item in node_score]

def rate_split(graphs, dglgraphs, num_subgraphs, min_node_threshold, K, r, k_training, args):
    split_graphs = []
    split_dglgraphs = []

    graph_index = random.randint(0, len(graphs) - 1)  # 随机选择一个图
    graph = graphs[graph_index]
    dglgraph = dglgraphs[graph_index]

    # graph, dglgraph = reduce_out_degree(dglgraph, (args.theta * args.subgraph_size))
    # graph, dglgraph = reduce_in_degree(dglgraph, (args.theta * args.subgraph_size))

    ###################  以下代码是为了生成soc-advogato的训练集和测试集  ###################
    # nodes = list(range(graph.vcount()))
    # random.shuffle(nodes)
    # split_index = int(len(nodes) * 0.5)
    # train_nodes = nodes[:split_index]
    # test_nodes = nodes[split_index:]
    #
    # train_graph = graph.subgraph(train_nodes)
    # train_dglgraph = dglgraph.subgraph(train_nodes)
    #
    # test_graph = graph.subgraph(test_nodes)
    # test_dglgraph = dglgraph.subgraph(test_nodes)
    #
    #
    # graph = train_graph
    # dglgraph = train_dglgraph
    # ###################  以上代码是为了生成soc-advogato的训练集和测试集  ###################

    # 删掉自环
    selfloop_edges = [edge for edge in graph.es if edge.source == edge.target]
    graph.delete_edges(selfloop_edges)
    dglgraph = dgl.remove_self_loop(dglgraph)

    isolated_nodes = [v.index for v in graph.vs if v.degree() == 0]
    graph.delete_vertices(isolated_nodes)

    isolated_nodes_dgl = dglgraph.in_degrees() + dglgraph.out_degrees() == 0
    dglgraph = dgl.remove_nodes(dglgraph, isolated_nodes_dgl.nonzero(as_tuple=True)[0].tolist())

    # fanout = 10
    # split_graphs, split_dglgraphs = sample_subgraph(dglgraph, fanout, target_size=200, num_subgraphs=num_subgraphs)

    # graph, dglgraph = limit_outdegree(graph, dglgraph, K)
    # graph, dglgraph = limit_indegree(graph, dglgraph, K)

    unused_nodes = list(range(len(graph.vs)))  # 未使用的节点

    sub_size = args.subgraph_size  # 初始子图大小
    count = 0  # 计数器
    switch_count = int((args.rate / 100) * num_subgraphs)

    while len(split_graphs) < num_subgraphs and unused_nodes:
        if count >= switch_count:
            sub_size = args.subgraph_size
        start_node = random.choice(unused_nodes)  # 在选定的图中随机选择一个起始节点
        # unused_nodes.remove(start_node)  # 从未使用的节点中移除起始节点

        if args.sample_mode == "bfs":
            visited, bfs_edges = bfs_sample_random(graph, start_node, sub_size, r, set())
        else:
            visited = dfs_sample(graph, start_node, sub_size, r, set())

        if len(visited) >= sub_size:
            nodes_to_include = list(visited)
            # 有向子图提取
            # subgraph = graph.subgraph(nodes_to_include)
            subdglgraph = dglgraph.subgraph(nodes_to_include)
            node_map = {i: nodes_to_include[i] for i in range(len(nodes_to_include))}

            subdglgraph_bfs_edges = [(list(node_map.keys())[list(node_map.values()).index(src)],
                                      list(node_map.keys())[list(node_map.values()).index(dst)])
                                     for src, dst in bfs_edges if src in node_map.values() and dst in node_map.values()]

            subdglgraph_edges = set(zip(subdglgraph.edges()[0].tolist(), subdglgraph.edges()[1].tolist()))
            bfs_edge_set = set(subdglgraph_bfs_edges)
            remaining_edges = subdglgraph_edges - bfs_edge_set

            # if len(subdglgraph_edges) > (args.theta * args.subgraph_size):
            #     num_edges_to_remove = len(subdglgraph_edges) - (args.theta * args.subgraph_size)
            #     edges_to_remove = random.sample(remaining_edges, num_edges_to_remove)
            #     edges_to_remove_id = subdglgraph.edge_ids(*zip(*edges_to_remove))
            #     subdglgraph.remove_edges(edges_to_remove_id)

            # num_edges_to_remove = int(subdglgraph.number_of_edges() * 0)  # 删除 30% 的边
            # edges_to_remove = random.sample(range(subdglgraph.number_of_edges()), num_edges_to_remove)
            # subdglgraph.remove_edges(edges_to_remove)

            nodes = subdglgraph.nodes().to('cpu').numpy()
            edges = subdglgraph.edges()

            subgraph = igraph.Graph(directed=True)
            subgraph.add_vertices(len(nodes))
            for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
                subgraph.add_edge(src.item(), dst.item())  # 转换为 int

            if (args.theta * args.subgraph_size) -1 <= subdglgraph.number_of_edges():
                split_graphs.append(subgraph)
                split_dglgraphs.append(subdglgraph)
                unused_nodes.remove(start_node)
                count += 1

    return split_graphs, split_dglgraphs


def sample_subgraph(dglgraph, fanout, target_size, num_subgraphs):
    # 随机选择一个起始节点
    count = 0
    split_graphs = []
    split_dglgraphs = []
    while (count < num_subgraphs):
        num_nodes = dglgraph.num_nodes()
        seed_node = random.randint(0, num_nodes - 1)
        seed_nodes = torch.tensor([seed_node], device=dglgraph.device)

        # 邻居采样
        sampler = dgl.dataloading.MultiLayerNeighborSampler([fanout, fanout, fanout, fanout, fanout])
        dataloader = dgl.dataloading.DataLoader(
            dglgraph, seed_nodes, sampler, batch_size=len(seed_nodes), shuffle=False, drop_last=False
        )

        for input_nodes, output_nodes, blocks in dataloader:
            subgraph = blocks[0].to(dglgraph.device)
            break

        new_dglgraph = dgl.graph((subgraph.edges()[0], subgraph.edges()[1]),device=dglgraph.device)
        if new_dglgraph.num_nodes() > 0:
            new_dglgraph.ndata.update(subgraph.srcdata)
            new_dglgraph.edata.update(subgraph.edata)
        else:
            continue

        nodes = new_dglgraph.nodes().to('cpu').numpy()
        edges = new_dglgraph.edges()

        new_igraph = igraph.Graph(directed=True)
        new_igraph.add_vertices(len(nodes))
        for src, dst in zip(*edges):  # 使用 zip 解压 edges 中的源节点和目标节点
            new_igraph.add_edge(src.item(), dst.item())  # 转换为 int

        if target_size <= new_dglgraph.num_nodes() <= 4 * target_size:
            count += 1
            split_graphs.append(new_igraph)
            split_dglgraphs.append(new_dglgraph)

    return split_graphs, split_dglgraphs


def bfs_split(graphs, dglgraphs, num_subgraphs, min_node_threshold, K, r, k_training, args):
    split_graphs = []
    split_dglgraphs = []

    graph_index = random.randint(0, len(graphs) - 1)  # 随机选择一个图
    graph = graphs[graph_index]
    dglgraph = dglgraphs[graph_index]
    graph, dglgraph = sample_out_edges(graph, dglgraph, K)
    graph, dglgraph = limit_indegree(graph, dglgraph, K)

    unused_nodes = list(range(len(graph.vs)))  # 未使用的节点

    while len(split_graphs) < num_subgraphs and unused_nodes:
        start_node = random.choice(unused_nodes)  # 在选定的图中随机选择一个起始节点
        unused_nodes.remove(start_node)  # 从未使用的节点中移除起始节点

        if args.sample_mode == "bfs":
            visited = bfs_sample(graph, start_node, 300, r, set())
        else:
            visited = dfs_sample(graph, start_node, 300, r, set())

        if len(visited) >= 300:
            nodes_to_include = list(visited)
            subgraph = graph.subgraph(nodes_to_include)
            subdglgraph = dglgraph.subgraph(nodes_to_include)

            split_graphs.append(subgraph)
            split_dglgraphs.append(subdglgraph)

    return split_graphs, split_dglgraphs

def dfs_sample(graph, start_node, target_size, max_depth, excluded_nodes):
    stack = [(start_node, 0)]  # (节点, 深度)
    visited = set([start_node])

    # 深度优先搜索
    while stack and len(visited) < target_size:
        current_node, depth = stack.pop()

        if depth <= (max_depth-1):
            successors = graph.successors(current_node)

            neighbors = list(successors)
            neighbors = sorted(neighbors, key=lambda x: len(graph.successors(x)))
            neighbors = neighbors[:40]

            for neighbor in neighbors:
                # dp状态下把neighbor not in excluded_nodes改成了neighbor in excluded_nodes
                if neighbor not in visited and neighbor not in excluded_nodes:
                    visited.add(neighbor)
                    stack.append((neighbor, depth + 1))

                if len(visited) >= target_size:
                    break

            if len(visited) >= target_size:
                break
    return visited

def bfs_sample(graph, start_node, target_size, max_depth, excluded_nodes):
    queue = deque([(start_node, 0)])  # (节点, 深度)
    visited = set([start_node])
    bfs_edges = []

    while queue:
        current_node, depth = queue.popleft()

        if depth > (max_depth - 1):
            continue  # 如果超过 max_depth-1，则跳过

        successors = graph.successors(current_node)
        # 随机从邻居中挑选k个邻居
        if len(successors) > 300:
            neighbors = random.sample(list(successors), 300)
        else:
            neighbors = list(successors)

        for neighbor in neighbors:
            if neighbor not in visited and neighbor in excluded_nodes:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                bfs_edges.append((neighbor, current_node))  # 记录边

                if len(visited) == target_size:
                    return visited, bfs_edges

    # 如果没有达到目标大小或者超过最大深度，则返回空集合和空边集
    return set(), []

def bfs_sample_random(graph, start_node, target_size, max_depth, excluded_nodes):
    queue = deque([(start_node, 0)])  # (节点, 深度)
    visited = set([start_node])
    bfs_edges = []

    while queue:
        current_node, depth = queue.popleft()

        if depth > (max_depth - 1):
            continue  # 如果超过 max_depth-1，则跳过

        successors = graph.successors(current_node)
        neighbors = list(successors)

        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in excluded_nodes:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                bfs_edges.append((neighbor, current_node))  # 记录边

                if len(visited) == target_size:
                    return visited, bfs_edges

    # 如果没有达到目标大小或者超过最大深度，则返回空集合和空边集
    return set(), []

import numpy as np
from collections import deque

def bfs_sample_heter(graph, start_node, target_size, max_depth, sample_rate):
    queue = deque([(start_node, 0)])  # (节点, 深度)
    visited = set([start_node])
    bfs_edges = []

    while queue:
        current_node, depth = queue.popleft()

        if depth > (max_depth - 1):
            continue  # 如果超过 max_depth-1，则跳过

        successors = graph.successors(current_node)
        neighbors = list(successors)

        if neighbors:
            # 取出邻居的采样概率
            neighbors_sample_rates = sample_rate[neighbors]

            # 归一化邻居的采样概率
            neighbors_sample_rates /= neighbors_sample_rates.sum()

            # 从[0,1]之间均匀生成一个概率，采样邻居采样概率高于p的邻居
            p = np.random.uniform(0, 1)
            selected_mask = neighbors_sample_rates > p
            selected_neighbors = []
            for i, mask in enumerate(selected_mask):
                if mask:
                    selected_neighbors.append(neighbors[i])

            for neighbor in selected_neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    bfs_edges.append((neighbor, current_node))  # 记录边

                    if len(visited) == target_size:
                        return visited, bfs_edges

    # 如果没有达到目标大小或者超过最大深度，则返回已访问节点和边集
    return set(), []

# def bfs_sample(graph, start_node, target_size, max_depth, excluded_nodes):
#     queue = deque([(start_node, 0)])  # (节点, 深度)
#     visited = set([start_node])
#     bfs_edges = []
#
#     while queue and len(visited) < target_size:
#         current_node, depth = queue.popleft()
#
#         if depth <= (max_depth-1):
#             successors = graph.successors(current_node)
#
#             neighbors = list(successors)
#             # neighbors = sorted(neighbors, key=lambda x: len(graph.successors(x)))
#             # neighbors = neighbors[:20]
#
#             for neighbor in neighbors:
#                 # dp状态下把neighbor not in excluded_nodes改成了neighbor in excluded_nodes
#                 if neighbor not in visited and neighbor not in excluded_nodes:
#                     visited.add(neighbor)
#                     queue.append((neighbor, depth + 1))
#                     bfs_edges.append((neighbor, current_node)) # 记录边
#
#                 if len(visited) >= target_size:
#                     break
#
#             if len(visited) >= target_size:
#                 break
#
#     return visited, bfs_edges

def bfs_split_two_parts(graphs, dglgraphs, num_subgraphs, min_node_threshold, K, r, k_training):
    split_graphs = []
    split_dglgraphs = []

    graph_index = random.randint(0, len(graphs) - 1)  # 随机选择一个图
    graph = graphs[graph_index]
    dglgraph = dglgraphs[graph_index]
    graph, dglgraph = sample_out_edges(graph, dglgraph, K)
    graph, dglgraph = limit_indegree(graph, dglgraph, K)

    unused_nodes = list(range(len(graph.vs)))  # 未使用的节点

    while len(split_graphs) < num_subgraphs and unused_nodes:
        start_node_part1 = random.choice(unused_nodes)  # 在选定的图中随机选择一个起始节点
        unused_nodes.remove(start_node_part1)  # 从未使用的节点中移除起始节点

        # BFS生成第一个子图（200节点左右）
        visited_part1 = bfs_sample(graph, start_node_part1, 250, r, set())

        # 找到第二个子图的起始节点，确保其不与第一个子图中的任何点相连
        start_node_part2 = None
        for node in unused_nodes:
            connected_to_part1 = False
            successors = graph.successors(node)
            predecessors = graph.predecessors(node)

            for neighbor in successors:
                if neighbor in visited_part1:
                    connected_to_part1 = True
                    break

            for neighbor in predecessors:
                if neighbor in visited_part1:
                    connected_to_part1 = True
                    break

            if not connected_to_part1:
                start_node_part2 = node
                break

        if start_node_part2 is None:
            continue  # 如果没有找到合适的起始节点，则跳过此次循环

        # BFS生成第二个子图（100节点左右）
        visited_part2 = bfs_sample(graph, start_node_part2, 50, r, visited_part1)

        # 记录从第一个子图到第二个子图的边，以及从第二个子图到第一个子图的边
        edges_to_remove = set()
        for u in visited_part1:
            for v in visited_part2:
                if graph.are_connected(u, v):  # 判断边是否存在
                    eid = graph.get_eid(u, v, error=False)
                    if eid != -1:  # 如果边存在
                        edges_to_remove.add((u, v, eid))

        for u in visited_part2:
            for v in visited_part1:
                if graph.are_connected(u, v):  # 判断边是否存在
                    eid = graph.get_eid(u, v, error=False)
                    if eid != -1:  # 如果边存在
                        edges_to_remove.add((u, v, eid))

        # 创建子图和子DGL图
        nodes_to_include = list(visited_part1.union(visited_part2))
        subgraph = graph.subgraph(nodes_to_include)
        subdglgraph = dglgraph.subgraph(nodes_to_include)

        node_id_map = {old_id: new_id for new_id, old_id in enumerate(nodes_to_include)}

        # 删除边
        subgraph_edges_to_remove = []
        subdglgraph_edges_to_remove = []
        for u, v, eid in edges_to_remove:
            if eid != -1:
                sub_u = node_id_map.get(u)
                sub_v = node_id_map.get(v)
                if sub_u is not None and sub_v is not None:
                    try:
                        sub_eid = subgraph.get_eid(sub_u, sub_v, error=False)
                        if sub_eid != -1:
                            subgraph_edges_to_remove.append(sub_eid)
                            subdglgraph_edges_to_remove.append((sub_u, sub_v))
                    except ValueError:
                        pass

        if subgraph_edges_to_remove:
            subgraph.delete_edges(subgraph_edges_to_remove)

        if subdglgraph_edges_to_remove:
            for edge in subgraph_edges_to_remove:
                subdglgraph.remove_edges(edge)

        split_graphs.append(subgraph)
        split_dglgraphs.append(subdglgraph)

    return split_graphs, split_dglgraphs


def get_test_names(path_to_test=PATH_TO_VAL):
    test_names = []
    for rt, _, files in os.walk(path_to_test):
        if rt == path_to_test:
            for file in files:
                if file.endswith('.txt') and 'friend' not in file:
                    test_names.append(file)
    return test_names


def gen_random_test(input_dim, directed_test=DIRECTED_VAL, feature_type=FEATURE_TYPE, n_test_graph=N_VAL_GRAPH, n_test_node=N_TEST_NODE, p=1e-2):
    graphs = gen_erdos_graphs(n_test_graph, n_test_node, p, directed_test)

    dglgraphs = [get_rev_dgl(
        graph, feature_type, input_dim, directed_test, False) for graph in graphs]
    return graphs, dglgraphs


def load_test(input_dim, directed_test=DIRECTED_VAL, feature_type=FEATURE_TYPE, use_cuda=USE_CUDA_TEST, path_to_test=PATH_TO_VAL):

    test_list = get_test_names(path_to_test)
    for graph_name in test_list:
        graph = igraph.Graph().Read_Edgelist(
            f"{path_to_test}{graph_name}", directed=directed_test)
        dglgraph = get_rev_dgl(graph, feature_type,
                               input_dim, directed_test, use_cuda)
        yield graph_name, graph, dglgraph


def load_igraph(filename, path=PATH_TO_VAL, is_directed=DIRECTED_VAL):
    graph = igraph.Graph().Read_Edgelist(
        f"{path}{filename}", directed=is_directed)
    return graph
