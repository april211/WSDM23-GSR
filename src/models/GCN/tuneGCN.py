import sys
import os.path as osp
sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import argparse

from utils.tune_utils import Tuner
import utils.util_funcs as util_funcs
from models.GCN.semi_sup_ncls_exp import semi_sup_ncls_exp
from models.GCN.config import GCNConfig



model = 'GCN'
TUNE_DICT = {
    'dropout_prob': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'lr': [0.005, 0.01, 0.05],
    'n_hidden': [8, 16, 32, 64, 128, 256],
}


@util_funcs.timing
def tune_GCN():
    # * =============== Init Args =================
    exp_name = 'GCN_NCLS'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_times', type=int, default=5)
    parser.add_argument('-s', '--start_ind', type=int, default=0)
    parser.add_argument('-v', '--reverse_iter', action='store_true', help='reverse iter or not')
    parser.add_argument('-l', '--log_on', action='store_true', help='enable log or not')
    parser.add_argument('-d', '--dataset', type=str, default='cora')
    parser.add_argument('-e', '--exp_name', type=str, default=exp_name)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-t', '--train_percentage', type=int, default=0)
    args = parser.parse_args()

    # * =============== Fine Tune (grid search) =================

    args.__dict__.update({'model': model})
    config = GCNConfig(args)
    tuner = Tuner(model_config_obj=config, train_func=semi_sup_ncls_exp, search_dict=TUNE_DICT)
    tuner.grid_search()
    
    # * =============== Result Summary  =================

    tuner.summarize()


if __name__ == '__main__':
    tune_GCN()

# nohup python src/models/GCN/tuneGCN.py -d arxiv -g 1 -l > train_arxiv.log 2>&1 &

