from utils.conf_utils import ModelConfigABS
from utils.proj_settings import TEMP_PATH
from utils.util_funcs import mkdirs_p


class GCNConfig(ModelConfigABS):

    def __init__(self, args):
        super(GCNConfig, self).__init__('GCN')

        # ! Model settings
        self.lr = 0.01
        self.dropout_prob = 0.5
        self.n_hidden = 64
        self.n_layer = 2
        self.weight_decay = 5e-4
        self.early_stop = 100
        self.epochs = 500

        # ! Other settings
        self.__dict__.update(args.__dict__)

        # Create necessary directories for `checkpoint_file` and `res_file`.
        mkdirs_p([getattr(self, file) for file in self._related_file_list])

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_layer{self.n_layer}_nhidden{self.n_hidden}_{self.model}_lr{self.lr}_dropout{self.dropout_prob}_es{self.early_stop if self.early_stop > 0 else 0}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"
    
    def get_parameter_dict(self):
        excluded_config_list = ['f_prefix', 'checkpoint_file', 'res_file', '_related_file_list',
                                'gpu', 'log_on', 'device']
        parameter_dict = {}
        for attr_name in self.__dict__.keys():
            if attr_name not in excluded_config_list:
                parameter_dict.update({attr_name: self.__dict__[attr_name]})
        return parameter_dict
    


