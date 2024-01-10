import sys
import argparse
import os.path as osp

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import torch
import torch.nn.functional as F

import dgl

from pprint import pformat

from models.GraphSAGE.model import GraphSAGE
from models.GraphSAGE.config import GraphSAGEConfig

from utils.conf_utils import *
from utils.evaluation import *
from utils.early_stopper import EarlyStopping
from utils.data_utils import preprocess_data
from utils.util_funcs import exp_init, timing
from utils.experiment import FullBatchNCExp


@timing
def semi_sup_ncls_exp(args, display_params=True):

    device = exp_init(seed=args.seed, gpu_id=args.gpu)

    # ! config
    cf = GraphSAGEConfig(args)
    cf.device = device

    # ! Load Graph
    g, features, feature_dims, cf.n_class, \
                        labels, train_idx, val_idx, test_idx = preprocess_data(cf.dataset, cf.train_percentage)

    features = features.to(cf.device)
    g = dgl.add_self_loop(g).to(cf.device)

    data_obj = Dict2Attrs({'labels': labels, 'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx})

    if display_params:
        print(f'{pformat(cf.get_parameter_dict())}\nStart training..')
    else:
        print('\nStart training..')

    model = GraphSAGE(feature_dims, cf.n_hidden, cf.n_class, cf.n_layer, F.relu, cf.dropout_prob)
    model.to(cf.device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    
    if cf.early_stop > 0:
        stopper = EarlyStopping(patience=cf.early_stop, ckpoint_path=cf.checkpoint_file)
    else:
        stopper = None

    # ! Train
    trainer = FullBatchNCExp(model=model, g=g, cf=cf, features=features,
                               data_obj=data_obj, stopper=stopper, optimizer=optimizer,
                               loss_func=torch.nn.CrossEntropyLoss())
    trainer.run_exp()
    trainer.eval_and_save()

    return cf


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training settings")

    # dataset = 'pubmed'
    # dataset = 'citeseer'
    # dataset = 'cora'
    dataset = 'arxiv'

    # ! Settings
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-t", "--train_percentage", default=0, type=int)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    # ! Train
    # python src/models/GraphSAGE/semi_sup_ncls_exp.py -d arxiv -g 0
    cf = semi_sup_ncls_exp(args)

