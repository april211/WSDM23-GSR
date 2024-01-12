#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from ogb.nodeproppred import DglNodePropPredDataset

import utils.util_funcs as util_funcs
from utils.proj_settings import *


def get_stochastic_loader(g, train_nids, batch_size, num_workers):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    return dgl.dataloading.NodeDataLoader(
        g.cpu(), train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)


def graph_normalization(g, cuda):
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    return g


def stratified_train_test_split(label_idx, labels, num_nodes, train_percentage, seed=2021):
    """
    Note: validation and test set are of the same size.
    """
    num_train_nodes = int(train_percentage / 100 * num_nodes)
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)

    train_idx, test_and_valid_idx = train_test_split(
        label_idx, test_size=test_rate_in_labeled_nodes, random_state=seed, shuffle=True, stratify=labels)
    
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx, test_size=.5, random_state=seed, shuffle=True, stratify=labels[test_and_valid_idx])
    
    return train_idx, valid_idx, test_idx


def normalize_features(mx):
    """
    Row-normalize feature matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def preprocess_data(dataset_name, train_percentage):
    """
    Preprocess data for training, validation and testing.
    train_percentage : percentage of training data (0 ~ 100), if <= 0, use the default split.

    return : graph, features, feature_dims, nclass, labels, train_idx, val_idx, test_idx
    """
    import dgl

    # Modified from AAAI21 FA-GCN
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        load_default_split = train_percentage <= 0

        edge = np.loadtxt(f'{DATA_PATH}/{dataset_name}/{dataset_name}.edge', dtype=int).tolist()
        features = np.loadtxt(f'{DATA_PATH}/{dataset_name}/{dataset_name}.feature', dtype=float)
        labels = np.loadtxt(f'{DATA_PATH}/{dataset_name}/{dataset_name}.label', dtype=int)

        if load_default_split:
            train_idx = np.loadtxt(f'{DATA_PATH}/{dataset_name}/{dataset_name}.train', dtype=int)
            val_idx = np.loadtxt(f'{DATA_PATH}/{dataset_name}/{dataset_name}.val', dtype=int)
            test_idx = np.loadtxt(f'{DATA_PATH}/{dataset_name}/{dataset_name}.test', dtype=int)
        else:
            train_idx, val_idx, test_idx = stratified_train_test_split(np.arange(len(labels)), 
                                                                labels, len(labels), train_percentage)
        
        n_class = len(set(labels.tolist()))

        print(dataset_name, n_class)

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        features = normalize_features(features)

        features =torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train_idx = torch.LongTensor(train_idx)
        val_idx = torch.LongTensor(val_idx)
        test_idx = torch.LongTensor(test_idx)

    elif dataset_name in ['airport', 'blogcatalog', 'flickr']:
        load_default_split = train_percentage <= 0

        # sparse
        adj_orig = pickle.load(open(f'{DATA_PATH}/{dataset_name}/{dataset_name}_adj.pkl', 'rb'))
        features = pickle.load(open(f'{DATA_PATH}/{dataset_name}/{dataset_name}_features.pkl', 'rb'))

        # tensor
        labels = pickle.load(open(f'{DATA_PATH}/{dataset_name}/{dataset_name}_labels.pkl', 'rb'))
        
        if torch.is_tensor(labels):
            labels = labels.numpy()

        if load_default_split:
            # `tvt_nids` contains 3 arrays: train(0), val(1), test(2)
            tvt_nids = pickle.load(open(f'{DATA_PATH}/{dataset_name}/{dataset_name}_tvt_nids.pkl', 'rb'))  
            
            train_idx = tvt_nids[0]
            val_idx = tvt_nids[1]
            test_idx = tvt_nids[2]
        else:
            train_idx, val_idx, test_idx = stratified_train_test_split(np.arange(len(labels)), labels, len(labels),
                                                           train_percentage)
        n_class = len(set(labels.tolist()))
        print(dataset_name, n_class)

        adj_orig = adj_orig.tocoo()
        U = adj_orig.row.tolist()
        V = adj_orig.col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        # ** only normalize features for airport dataset ** ?
        if dataset_name in ['airport']:
            features = normalize_features(features)

        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())
        else:
            features = torch.FloatTensor(features)

        labels = torch.LongTensor(labels)
        train_idx = torch.LongTensor(train_idx)
        val_idx = torch.LongTensor(val_idx)
        test_idx = torch.LongTensor(test_idx)

    elif dataset_name in ['arxiv']:
        # Get the ogbn-arxiv dataset object.
        dataset_obj = DglNodePropPredDataset(name='ogbn-arxiv', root='data/ogb_arxiv')

        # Use the default split.
        split_idx = dataset_obj.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        
        # Use index `0` to get the only graph in the dataset.
        g, labels = dataset_obj[0]
        features = g.ndata['feat']
        n_class = 40
        labels = labels.squeeze()
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)
    
    # ** Add self loop only for citeseer dataset. **  ?
    # Zero in-degree nodes will lead to invalid output value. 
    # This is because no message will be passed to those nodes, the aggregation function will be appied on empty input. 
    # https://docs.dgl.ai/en/1.1.x/generated/dgl.nn.mxnet.conv.GraphConv.html
    if dataset_name in ['citeseer']:
        g = dgl.add_self_loop(g)
    
    return g, features, features.shape[1], n_class, labels, train_idx, val_idx, test_idx


# * ============================= Torch =============================

def get_topk_sim_edges(sim_mat, k, row_start_id, largest : bool):
    """
    sim_mat : similarity matrix of a batch of nodes, shape == (batch_size, N).
    row_start_id : the start index of the batch in the global adjacency matrix.
    largest : controls whether to return largest or smallest elements.
    return : global coords & similarity tensor.
    """
    sim_tensor, flat_idx = torch.topk(sim_mat.flatten(), k, largest=largest)
    coords = np.array(np.unravel_index(flat_idx.cpu().numpy(), sim_mat.shape)).T
    coords[:, 0] = coords[:, 0] + row_start_id          # local (batch) idx to global idx
    coords_tensor = torch.tensor(coords).to(sim_mat.device)
    return coords_tensor, sim_tensor


def global_topk(input, k, largest):
    """
    largest : controls whether to return largest or smallest elements.
    """
    _, flat_idx = torch.topk(input.flatten(), k, largest=largest)
    return np.array(np.unravel_index(flat_idx.cpu().numpy(), input.shape)).T.tolist()


def contains_zero_lines(h):
    """
    If h contains zero lines, return True.
    """
    zero_lines = torch.where(torch.sum(h, 1) == 0)[0]   # (tensor,)
    if len(zero_lines) > 0:
        print(f'{len(zero_lines)} zero lines are detected!')
        print(f"Indices of zero lines: {zero_lines}")
        return True
    return False


def batch_pairwise_cos_sim(mat, batch_size):
    # Normalization ?
    print("Call batch_pairwise_cos_sim!! Empty function!!")
    return


def cosine_similarity(x1, x2=None, eps=1e-8):
    """
    Calculate cosine similarity between x1 and x2.
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def edge_list_2_set(edge_list):
    return set(list(map(tuple, edge_list)))


def graph_2_loe(graph):
    """
    Convert a graph to a list of (row_id, col_id) edge tuples.
    """
    return list(map(tuple, np.column_stack([col.cpu().numpy() for col in graph.edges()]).tolist()))

def scalable_graph_refine(graph, emb, rm_num, add_num, batch_size, beta_f, device, norm=False):
    """
    Refine the given graph by adding and removing edges based on similarity matrix.

    `graph` : graph to be refined.
    `emb` : dict of embeddings, {'F': feature_emb, 'S': structure_emb}.
    `batch_size` : batch size for calculating local similarity matrix.
    `beta_f` : weight for feature similarity.
    `norm` : whether to normalize similarity matrix with respect to the minimum similarity.
    `rm_num` : number of **directed edges** to remove.
    `add_num` : number of **directed edges** to add.

    Note: Both `rm_num` and `add_num` have to be even numbers! 
            (`xxx_num // 2` **undirected edges** will be removed/added.)
    Inspired by the paper of `GAuG`.
    """
    def _update_topk_by_batch(batch_sim, start, mask, k, prev_idx, prev_sim, largest):
        """
        Use current batch to update TopK similarity and inds ** for the whole graph ** .
        """

        # Calculate results of the current batch.
        top_coords, top_sim = get_topk_sim_edges(batch_sim + mask, k, start, largest)

        # Concat previous results & sort again.
        temp_coords = torch.cat((prev_idx, top_coords))     
        temp_sim = torch.cat((prev_sim, top_sim))
        current_best_2idx = temp_sim.topk(k, largest=largest).indices
        return temp_sim[current_best_2idx], temp_coords[current_best_2idx]

    # ! The edges in these graphs are bi-directed, which means that (i, j) and (j, i) are both included.
    edges = set(graph_2_loe(graph))
    num_batches = int(graph.num_nodes() / batch_size) + 1

    # *** Detect whether the graph contains self-loops.
    contains_self_loop = False
    for edge in edges:
        if edge[0] == edge[1]:
            contains_self_loop = True
            break

    if add_num + rm_num == 0:
        return graph.edges()

    if norm:
        # Since maximum value of a similarity matrix is fixed as 1 (self-similarity == 1), 
        # we only have to calculate the minimum value.
        fsim_min, ssim_min = 99, 99         # Set the default minimum value as 99.

        for row_i in tqdm(range(num_batches), desc='Calculating minimum similarity'):
            
            # ! Initialize batch inds
            start = row_i * batch_size
            end = min((row_i + 1) * batch_size, graph.num_nodes())
            if end <= start:                # for robustness
                break

            # ! Calculate similarity LB for feature and structure embeddings.
            fsim_min = min(fsim_min, cosine_similarity(emb['F'][start:end], emb['F']).min())
            ssim_min = min(ssim_min, cosine_similarity(emb['S'][start:end], emb['S']).min())
    
    # ! Init index and similairty tensor, which will be updated in each batch.
    # Edge indexes should not be saved as floats in triples, 
    # since the number of nodes may well exceeds torch maximum of float16 (65504).
    rm_coords, add_coords = [torch.tensor([(0, 0) for i in range(_)]).type(torch.int32).to(device)
                         for _ in [1, 1]]  # Init with one random point (0, 0)
    add_sim = torch.ones(1).type(torch.float16).to(device) * -99
    rm_sim = torch.ones(1).type(torch.float16).to(device) * 99

    for row_i in tqdm(range(num_batches), desc='Batch filtering edges'):
        # ! Initialize batch inds
        start = row_i * batch_size
        end = min((row_i + 1) * batch_size, graph.num_nodes())
        if end <= start:                    # for robustness    
            break

        # ! Calculate similarity matrix by combining feature and structure embeddings.
        f_sim = cosine_similarity(emb['F'][start:end], emb['F'])
        s_sim = cosine_similarity(emb['S'][start:end], emb['S'])
        if norm:
            f_sim = (f_sim - fsim_min) / (1 - fsim_min)
            s_sim = (s_sim - ssim_min) / (1 - ssim_min)
        sim = beta_f * f_sim + (1 - beta_f) * s_sim

        # ! Get masks for edges to not add and remove. 
        # Edge mask (** directed edges **), which considers both directions.
        edge_mask, diag_mask = [torch.zeros_like(sim).type(torch.int8) for _ in range(2)]
        row_idx, col_idx = graph.out_edges(graph.nodes()[start: end])
        
        # global row-idx to local row-idx
        edge_mask[row_idx - start, col_idx] = 1

        # Diag mask
        diag_row_idx, diag_col_idx = zip(*[(global_idx - start, global_idx) 
                                                for global_idx in range(start, end)])
        diag_mask[diag_row_idx, diag_col_idx] = 1

        # Add masks for both existing edges and diag edges.
        # They could include the same edge twice, which conflicts with the definition of edge_mask (torch.int8).
        if contains_self_loop:              # for citeseer dataset, edge_mask is sufficient.
            add_mask = edge_mask * -99
        else:
            add_mask = (edge_mask + diag_mask) * -99

        # Remove masks: Non-Existing edges should be masked.
        # Note that all diag edges have maximum similarity (== 1), so they will not be removed naturally.
        rm_mask = (1 - edge_mask) * 99

        # ! Update edges to remove and add.
        if rm_num > 0:
            k = max(len(rm_sim), rm_num)          # ? k == rm_num, init_len(rm_sim) == 1 ?
            rm_sim, rm_coords = _update_topk_by_batch(sim, start, rm_mask, k, rm_coords, rm_sim, largest=False)
        if add_num > 0:
            k = max(len(add_sim), add_num)
            add_sim, add_coords = _update_topk_by_batch(sim, start, add_mask, k, add_coords, add_sim, largest=True)

    # ! Graph refinement, note that we treat one undirected edge as two directed edges.
    if rm_num > 0:
        rm_edges = [tuple(_) for _ in rm_coords.cpu().numpy().astype(int).tolist()]
        edges -= set(rm_edges)
    if add_num > 0:
        add_edges = [tuple(_) for _ in add_coords.cpu().numpy().astype(int).tolist()]
        edges |= set(add_edges)

    return edges


@util_funcs.timing
def cosine_similarity_batch(m1=None, m2=None, dist_batch_size=100, device=None):
    NoneType = type(None)

    if type(m1) is not torch.Tensor:  # only numpy conversion supported
        m1 = torch.from_numpy(m1).float()
    if type(m2) is not torch.Tensor and type(m2) is not NoneType:
        m2 = torch.from_numpy(m2).float()  # m2 could be None

    m2 = m1 if m2 is None else m2
    assert m1.shape[1] == m2.shape[1]

    result = torch.zeros([1, m2.shape[0]])      # We will delete this row later.

    for row_i in tqdm(range(0, int(m1.shape[0] / dist_batch_size) + 1), 
                                        desc='Calculating pairwise similarity'):
        start = row_i * dist_batch_size
        end = min([(row_i + 1) * dist_batch_size, m1.shape[0]])
        if end <= start:
            break
        rows = m1[start: end]
        # sim = cosine_similarity(rows, m2) # rows is O(1) size
        sim = cosine_similarity(rows.to(device), m2.to(device))

        result = torch.cat((result, sim.cpu()), 0)

    result = result[1:, :]  # deleting the first row, as it was used for setting the size only
    del sim
    return result  # return 1 - ret # should be used with sklearn cosine_similarity


def mp_to_relations(mp):
    return [f"{mp[t_id]}{mp[t_id + 1]}" for t_id in range(len(mp) - 1)]


# ! Torch Scaling Functions

def standarize(input):
    return (input - input.mean(0, keepdims=True)) / input.std(0, keepdims=True)


def row_l1_norm(input):
    return F.normalize(input, p=1, dim=1)


def col_l1_norm(input):
    return F.normalize(input, p=1, dim=0)


def min_max_scaling(input, type='col'):
    """
    min-max scaling modified from this link:
    
    https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/5

    input (2 dimensional torch tensor): input data to scale.
    type (str): type of scaling. Options: `row`, `col`, or `global`.

    Returns (2 dimensional torch tensor): min-max scaled torch tensor.

    Example input tensor (list format):
        [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    
    Scaled tensor (list format):
        [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]]
    """

    if type in ['row', 'col']:
        dim = 0 if type == 'col' else 1

        input -= input.min(dim).values
        input /= (input.max(dim).values - input.min(dim).values)            # ***

        # corner case: the row/col's minimum value equals the maximum value.
        input[input.isnan()] = 0
        return input
    elif type == 'global':
        return (input - input.min()) / (input.max() - input.min())
    else:
        ValueError('Invalid type of min-max scaling.')


if __name__ == "__main__":

    g, features, _, _, _, _, _, _ = preprocess_data('airport', 0)

    # Get the max degree in graph `g` 
    # to check if it's consistent with the dim of the "one-hot degree vectors".
    # Each entry of the "one-hot degree vectors" corresponds to one degree.
    print(g.in_degrees().max())

