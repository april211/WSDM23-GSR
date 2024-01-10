import utils.util_funcs as util_funcs
from abc import abstractmethod, ABCMeta
from utils.proj_settings import RES_PATH


class ModelConfigABS(metaclass=ABCMeta):
    """
    The ModelConfig class is an abstract base class \
    that cannot be instantiated on its own and is expected to be a blueprint for other classes.

    The ABCMeta metaclass allows the ModelConfig class to include abstract methods. 
    And the abstractmethod decorator indicates that any class that inherits from ModelConfig must implement the method.
    """

    def __init__(self, model):
        
        self.seed = 0
        self.model = model
        self.exp_name = 'default'

        self._related_file_list = ['checkpoint_file', 'res_file']

    @property
    @abstractmethod
    def f_prefix(self):
        """
        Result file prefix (file name without extension) string.
        """
        return ValueError('The model config file name must be defined!')

    @property
    @abstractmethod
    def checkpoint_file(self):
        """
        Checkpoint file path string.
        """
        return ValueError('The checkpoint file name must be defined!')

    @property
    def res_file(self):
        """
        Temperorary result file path string.
        """
        return f'{RES_PATH}{self.model}/{self.dataset}/l{self.train_percentage:02d}/{self.f_prefix}.txt'

    def get_sub_conf_dict(self, sub_conf_list):
        """
        Generate subconfig dict using sub_conf_list.
        """
        dct = {k: self.__dict__[k] for k in sub_conf_list}
        dct = dict(sorted(dct.items()))
        return dct
    
    @abstractmethod
    def get_parameter_dict(self):
        return self.__dict__


class Dict2Attrs():
    """
    Dict2Config: convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf : dict):
        self.__dict__.update(conf)

