import os
import sys
import logging
import pickle
import numpy as np
import time
import datetime
import pytz
import functools


# * ============================= Init =============================

def exp_init(seed, gpu_id):
    """
    Specify the gpu and set random seed for all related libraries.
    """
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Choose the first gpu you specified above.
    # https://discuss.pytorch.org/t/os-environ-cuda-visible-devices-does-not-work-well/132461/8
    import torch
    device = torch.device("cuda:0" if gpu_id >= 0 else "cpu")

    init_random_state(seed)
    return device

    # ** Torch related import should be imported afterward setting ** 


def init_random_state(seed=0):
    # ** Libraries using GPU should be imported after specifying GPU-ID ** 
    import torch
    import random
    import dgl
    dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# * ============================= Log Related =============================\

def timing(func):
    """
    Define a decorator to log the running time of a function. 
    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret
    return wrapper


def get_subset_dict(dct : dict, sub_keys):
    """
    Generate a new dict with only sub_keys.
    """
    return {k: dct[k] for k in sub_keys if k in dct}


def print_dict(dct : dict, end_string='\n\n'):
    for key in dct.keys():
        if isinstance(dct[key], dict):
            print('\n', end='')
            print_dict(dct[key], end_string='')
        elif isinstance(dct[key], int):
            print('{}: {:04d}'.format(key, dct[key]), end=', ')
        elif isinstance(dct[key], float):
            print('{}: {:.4f}'.format(key, dct[key]), end=', ')
        else:
            print('{}: {}'.format(key, dct[key]), end=', ')
    print(end_string, end='')


def write_nested_dict(dct : dict, path : str):
    mkdirs_p([path])
    with open(path, 'a+') as f:
        f.write('\n')
        for key in dct.keys():
            if isinstance(dct[key], dict):
                f.write(str(dct[key]) + '\n')
    print(f"Write the nested dict into: `{path}`.")


def disable_logs():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def print_log(log_dict : dict):
    log_formatter = lambda log: f'{log:.4f}' if isinstance(log, float) else f'{log:04d}'
    print(' | '.join([f'{k} {log_formatter(v)}' for k, v in log_dict.items()]))


def list_2_str(lst : list):
    """
    Join the elements in a list with '_'.
    """
    return '_'.join(lst)


# * ============================= Pickle Operations =============================

def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def save_pickle(var, path):
    mkdirs_p([path])
    pickle.dump(var, open(path, 'wb'))
    print(f'Pickle `{path}` successfully saved!')


# * ============================= Path Operations =============================

def check_path(path : str):
    if not os.path.exists(path):
        os.makedirs(path)


def get_parent_dir(path : str):
    from pathlib import Path
    return Path(path).parent.resolve()


def get_grand_parent_dir(path : str):
    return get_parent_dir(get_parent_dir(path))


def get_abs_path(relavive_path : str):
    """
    Get the absolute path of the specified relative path.
    """
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    return os.path.join(root_path, relavive_path)


def mkdir_p(abspath : str, verbose=True):
    """
    Create necessary directories for the specified abspath. \n
    verbose : bool
        Whether to print the process of creating directories.
    """
    if os.path.exists(abspath): 
        return
    
    # fix the permission issue of creating directories.
    os.umask(0)
    os.makedirs(name=abspath, mode=0o755, exist_ok=False)

    if verbose:
        print('Created directory {}'.format(abspath))


def mkdirs_p(path_list : list, is_relative=True, verbose=True):
    """
    Turn a list of paths into a list of absolute paths, then create necessary directories for them. \n
    If `is_relative` is True, the paths in `path_list` will be relative to the root path of the project. \n
    """
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    for path in path_list:
        abspath = os.path.join(root_path, path) if is_relative else path
        absdir = os.path.dirname(abspath)
        mkdir_p(absdir, verbose)


# * ============================= Time Related =============================

def time2str(sec : int):
    """
    Convert seconds to a string of format 'xxday' or 'xxh' or 'xxmin' or 'xxs'.
    """
    if sec > 86400:
        return '{:.2f}day'.format(sec / 86400)
    if sec > 3600:
        return '{:.2f}h'.format(sec / 3600)
    elif sec > 60:
        return '{:.2f}min'.format(sec / 60)
    else:
        return '{:.2f}s'.format(sec)


def get_cur_time(timezone='Asia/Shanghai', t_format='%y-%m-%d, %H:%M:%S'):
    """
    Get current time in the specified timezone and format.
    """
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


class Dict2Config():
    """
    Convert dict to a config object for better attribute acccessing.
    """
    def __init__(self, conf):
        self.__dict__.update(conf)      # convert conf dict to object attributes


# * ============================= Itertool Related =============================

def zip_tuples(list_of_tuples : list):
    """
    Zip a list of tuples, then convert them to a map of lists.
    """
    return map(list, zip(*list_of_tuples))


if __name__ == '__main__':

    from proj_settings import TEST_PATH

    # test for lot_to_tol
    a = [(1, 2), (3, 4), (5, 6)]
    for idx, lst in enumerate(zip_tuples(a)):
        print(idx, lst)

    # test for get_parent_dir
    print(get_parent_dir("/home/wgk/"))
    print(get_parent_dir("/home/wgk/Git_Repositories/WSDM23-GSR/src/utils/util_funcs.py"))

    # test for get_grand_parent_dir
    print(get_grand_parent_dir("/home/wgk/"))
    print(get_grand_parent_dir("/home/wgk/Git_Repositories/WSDM23-GSR/src/utils/util_funcs.py"))

    # test for write_nested_dict
    dct = {'a': 1, 'b': {'b1': 2, 'b2': 2}, 'c': {'c1': 3, 'c2': 3}}
    write_nested_dict(dct, f'{TEST_PATH}test_for_write_nested_dict.txt')

