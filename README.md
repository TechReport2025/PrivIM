# PrivIM
This repo contains the code and full version of the paper "PrivIM: Differentially Private Graph Neural Networks for Influence Maximization".

## Requirements
Run the following command to install requirements.

```
$ pip install -r requirements.txt
```

## Instructions
* `baselines/`: Heuristic algorithms for CELF
* `dataset/`: Training and test graphs
* `experiments/`: Training and evaluation launchers
* `model/`: GNN layers

## Training
```
$ python train_models.py --target_epsilon 3 --settings PrivIM+ --dataset bitcoin
```

## Evaluation
```
$ python evaluate_models.py
```

## Overview
The abstract from the paper is the following:

*Influence Maximization (IM), aiming to identify a small set of highly influential nodes in social networks, is a critical problem in graph analysis. Recently, Graph Neural Networks (GNNs) have demonstrated superior effectiveness in addressing IM. However, a trained GNN still raises significant privacy concerns, as it may expose sensitive node features and structural information. While Differential Privacy (DP) techniques have been widely applied to GNNs for node-level tasks, they cannot be directly extended to IM problems. This is because IM requires more complex structural information for training, resulting in an extremely larger DP noise scale than node-level tasks. To tackle these issues, we propose PrivIM, a novel differentially private subgraph-based GNN framework for IM tasks, which ensures node-level DP guarantees. Within PrivIM, we design a unique dual-stage adaptive frequency sampling scheme to optimize the model utility. First, it reduces the correlation between nodes by dynamically adjusting each node's sampling probability. Then additional subgraphs are incorporated to supplement boundary structural information, enhancing utility without increasing privacy budget. Extensive experiments on six real-world datasets demonstrate that PrivIM maintains high utility in IM compared to baseline methods.*

## Important Python Libraries
* igraph=0.9.1
* torch=1.8.1
* dgl=0.6.0 (based on the CUDA version)
* furl=2.0.0
* timeout-decorator=0.5.0

## Dataset
We evaluate our methods on six widely-used real-world directed or undirected graph datasets with varying sizes and edge density, including the email network (Email), the social networks (Bitcoin, LastFM, Facebook, Gowalla), and the citation network (HepPh).

| **Dataset** | **$\vert V \vert$** | **$\vert E \vert$** | **Type** | **Avg. Degree** |
| ----------- | ------------------- | ------------------- | -------- | --------------- |
| **Email**   |       1K            |         25.6K       | Directed |    25.44        |
| **Bitcoin** |      5.9K           |         35.6K       | Directed |    6.05         |
| **LastFM**   |       7.6K          |         27.8K       | Undirected |    7.29         |
| **HepPh**   |       12K           |         118.5K       | Undirected |    19.74        |
| **Facebook**   |       22.5K            |         171K       | Undirected |    15.22        |
| **Gowalla**   |       196K            |         950.3K       | Undirected |    9.67         |

More real-world test graphs can be found in [SNAP](https://snap.stanford.edu/data/)
