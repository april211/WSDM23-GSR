import torch
import numpy as np


def topk_uniques(sim, k=5) -> dict:
    """
    Return the most common nodes in the top-k neighbors of each node \
        and their "degree" (i.e., the number of times they appear in the top-k neighbors of all nodes).
    """
    topk_neighbors = torch.topk(sim, k).indices.flatten().cpu().numpy().tolist()

    # sum(topk_neighbors == v): "degree" of node v.
    std_degree_dict = {v: sum(topk_neighbors == v) / sim.shape[0] for v in np.unique(topk_neighbors)}
    return {v: round(std_degree_dict[v], 4) for v in sorted(std_degree_dict, key=std_degree_dict.get, reverse=True)}

