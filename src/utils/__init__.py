import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
os.chdir(root_path)
sys.path.append(root_path + 'src')

# ! IMPORTANT NOTES
# ! These functions shan't use cuda related packages, since it will cause wrong assignment of GPU-ID
from .util_funcs import exp_init, timing, print_log
from .proj_settings import *
from .conf_utils import Dict2Attrs
from .tune_utils import Tuner

import argparse
from time import time

