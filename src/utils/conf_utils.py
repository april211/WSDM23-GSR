import utils.util_funcs as util_funcs
from abc import abstractmethod, ABCMeta
from utils.proj_settings import RES_PATH


class ModelConfig(metaclass=ABCMeta):
    """
    The ModelConfig class is an abstract base class \
    that cannot be instantiated on its own and is expected to be a blueprint for other classes.

    The ABCMeta metaclass allows the ModelConfig class to include abstract methods. 
    And the abstractmethod decorator indicates that any class that inherits from ModelConfig must implement the method.
    """

    def __init__(self, model):
        self.model = model

        self.exp_name = 'default'
        self.seed = 0
        self.birth_time = util_funcs.get_cur_time()

        self._model_conf_list = None
        self._file_conf_list = ['checkpoint_file', 'res_file']
        self._interested_conf_list = ['model']

    def __str__(self):
        """
        Print all attributes including data and other path settings added to the config object \
        except for the ones in the _interested_conf_list.
        """
        return str({k: v for k, v in self.model_conf.items() if k != '_interested_conf_list'})

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

    @property
    def model_conf(self):
        """
        Print the model settings only.
        """
        return {k: self.__dict__[k] for k in self._model_conf_list}

    def get_sub_conf_dict(self, sub_conf_list):
        """
        Generate subconfig dict using sub_conf_list.
        """
        return {k: self.__dict__[k] for k in sub_conf_list}

    def update__model_conf_list(self, new_conf=[]):
        """
        Maintain a list of interested configs.
        """
        other_configs = ['_model_conf_list', '_file_conf_list']

        if len(new_conf) == 0:  # initialization, which includes `self._interested_conf_list`.
            self._model_conf_list = sorted(list(self.__dict__.copy().keys()))
            for other_config in other_configs:
                self._model_conf_list.remove(other_config)
        else:
            # Update self._model_conf_list only.
            self._model_conf_list = sorted(self._model_conf_list + new_conf)

    def update_modified_conf(self, new_conf_dict):
        """
        Update the config object with `new_conf_dict` and drop the unwanted items. \n
        `self.__dict__` and `self._interested_conf_list` are updated.
        """
        self.__dict__.update(new_conf_dict)
        self._interested_conf_list += list(new_conf_dict)

        unwanted_items = ['log_on', 'gpu', 'train_phase', 'num_workers']
        for item in unwanted_items:
            self._interested_conf_list.remove(item)
        
        # Create necessary directories for `checkpoint_file` and `res_file`.
        util_funcs.mkdirs_p([getattr(self, _) for _ in self._file_conf_list])


class Dict2Config():
    """
    Dict2Config: convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf : dict):
        self.__dict__.update(conf)

