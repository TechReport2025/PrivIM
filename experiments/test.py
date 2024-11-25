import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
import powerlaw


# 创建一个Barabási–Albert模型的有向图
def generate_scale_free_graph(num_nodes, num_edges_to_attach):
    G = nx.barabasi_albert_graph(num_nodes, num_edges_to_attach)
    D = nx.DiGraph()

    # 将无向图转换为有向图
    for u, v in G.edges():
        if u != v:
            D.add_edge(u, v)
    return D


# 生成幂律分布的度数序列
def generate_powerlaw_degree_sequence(num_nodes, exponent, min_degree, max_degree, target_max_degree, try_count):
    while True:
        degree_sequence = np.random.zipf(exponent, num_nodes)
        degree_sequence = np.clip(degree_sequence, min_degree, max_degree)
        if try_count >= 20:
            degree_sequence[1] = target_max_degree

        min_count = np.min(degree_sequence)
        max_count = np.max(degree_sequence)
        if min_count >= min_degree and max_count <= max_degree:
            break
    return degree_sequence


# 创建一个具有给定度数序列的有向图
def generate_powerlaw_digraph(num_nodes, exponent, min_degree, max_degree, target_max_degree, try_count):
    while True:
        in_degrees = generate_powerlaw_degree_sequence(num_nodes, exponent, min_degree, max_degree, target_max_degree, try_count)
        out_degrees = generate_powerlaw_degree_sequence(num_nodes, exponent, min_degree, max_degree, target_max_degree, try_count)
        if sum(in_degrees) == sum(out_degrees):
            break
    # 创建配对图模型并调整成有向图
    G = nx.generators.degree_seq.directed_configuration_model(in_degrees, out_degrees)
    D = nx.DiGraph(G)

    # 移除自环和多重边
    D.remove_edges_from(nx.selfloop_edges(D))
    D = nx.DiGraph(nx.Graph(D))

    return D


# 参数设置
# 这里面的degree指的是入度和出度，不是和，和需要乘以2
num_nodes = 2000
exponent = 2.5
min_degree = 1
max_degree = 400
target_max_degree = max_degree
try_count = 0

# 生成图
# D = generate_powerlaw_digraph(num_nodes, exponent, min_degree, max_degree, target_max_degree, try_count)


# # 生成具有100个节点，每次连接2个新边的有向图
# num_nodes = 2000
# num_edges_to_attach = 30
# D = generate_scale_free_graph(num_nodes, num_edges_to_attach)
#

# 计算度数分布
degree_sequence = sorted([d for n, d in D_dense.degree()], reverse=True)
# if 2 * (max_degree-50) <= degree_sequence[0] <= 2 * max_degree:
degree_count = Counter(degree_sequence)
deg, cnt = zip(*degree_count.items())

data = []
for value, count in degree_count.items():
    data.extend([value] * count)

fit = powerlaw.Fit(data,xmin=5)

print(f"alpha: {fit.power_law.alpha}")
print(f"xmin: {fit.power_law.xmin}")

# with open(f"D:/demo/fastCover-master/morty_data/er-graphs/4000/sparse_1_{max_degree}.txt", "w") as f:
#     for edge in D.edges():
#         f.write(f"{edge[0]} {edge[1]}\n")

# 绘制度数分布直方图
plt.figure(figsize=(12, 8))
plt.bar(deg, cnt, width=0.2, color='b')
plt.title("Email Testset", fontsize=16)
plt.xlabel("Degree", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.yscale('log')
# plt.xscale('log')
plt.xlim(0, 200)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which="both", ls="--")
# plt.savefig(f"D:/demo/result/0709_0716/fig/degree_email_test.png", dpi=600)
plt.show()
#     break
#
# else:
#     try_count += 1