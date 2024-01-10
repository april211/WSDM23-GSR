import os
import ast
import time
import traceback
import pandas as pd
import multiprocessing
from itertools import product
from pprint import pformat
from copy import deepcopy

from .util_funcs import get_cur_time, timing, disable_logs,\
                     enable_logs, time2str, mkdirs_p, write_nested_dict
from .proj_settings import EVAL_METRIC, SUM_PATH, RES_PATH


class Tuner():
    """
    Major functions: \n 
    ✅ Maintains dataset specific tune dict \n
    ✅ Tune dict to tune dataframe (para combinations) \n
    ✅ Beautiful printer \n
    ✅ Build-in grid search function \n
    ✅ Result summarization \n
    ✅ Try-catch function to deal with bugs \n
    ✅ Tune report to txt.
    """
    def __init__(self, model_config_obj, train_func, search_dict : dict, default_dict : dict = None):
        """
        Explain the meaning of parameters: `exp_config`, `search_dict`, `default_dict`.
        """

        self.model_config_obj = model_config_obj

        self.birth_time = get_cur_time()

        # self._d: hyper-parameters to be tuned.
        self.hyperparam_dict = deepcopy(default_dict) if default_dict is not None else {}
        if 'data_spec_configs' in search_dict:
            self.update_dataset_specific_configs(search_dict['data_spec_configs'])
            search_dict.pop('data_spec_configs')
        self.hyperparam_dict.update(search_dict)        

        # Set the training function.
        self.train_func = train_func

        # Get the experiment configs for printing, excluding the pre-defined params in `XXConfig`.
        excluded_config_list = ['seed'] + list(self.hyperparam_dict.keys())
        self.global_config_list = []
        for config in self.model_config_obj.__dict__:
            if config not in excluded_config_list:
                self.global_config_list.append(config)

    def update_dataset_specific_configs(self, dct : dict):
        for k, v in dct.items():
            self.hyperparam_dict.update({k: v[self.model_config_obj.dataset]})
        self.hyperparam_dict = dict(sorted(self.hyperparam_dict.items()))
    
    def __str__(self):
        gf_dct = self.model_config_obj.get_sub_conf_dict(self.global_config_list)
        return f'\nGlobal configs: \n{pformat(gf_dct)}\n' \
               f'\nGrid search params: \n{pformat(self.hyperparam_dict)}\n'
    
    # * ============================= Properties =============================

    @property
    def hyper_param_df(self):
        """
        Generate dataframe from `self._d`.
        Each row of this dataframe stands for a trial (hyper-parameter combination).
        """
        # convert the values of parameters to list
        # param_name : [param_value1, param_value2, ...]
        for param in self.hyperparam_dict.keys():
            if not isinstance(self.hyperparam_dict[param], list):
                self.hyperparam_dict[param] = [self.hyperparam_dict[param]]
        return pd.DataFrame.from_records(dict_product(self.hyperparam_dict))

    # * ============================= Tuning =============================

    @timing
    def grid_search(self):
        """
        Grid search for hyper-parameters tuning.
        """
        print(self)

        failed_trials_num, skipped_trials_num, finished_trials_num = 0, 0, 0
        total_trials_num = len(self.hyper_param_df) - self.model_config_obj.start_ind

        tune_dict_list = self.hyper_param_df.to_dict('records')
        
        outer_start_time = time.time()

        # skip the 0 ~ `self.start_ind` trials
        for seq_cnt in range(self.model_config_obj.start_ind, len(tune_dict_list)):

            # `trial_idx` can start from the tail of `tune_dict_list` or the head of `tune_dict_list`.
            trial_idx = len(tune_dict_list) - seq_cnt - 1 if self.model_config_obj.reverse_iter else seq_cnt

            # Get the current trial's hyper-parameters & update the predefined parameters.
            self.model_config_obj.__dict__.update(tune_dict_list[trial_idx])

            inner_start_time = time.time()

            print(f"\n{seq_cnt}/{len(tune_dict_list)} <{self.model_config_obj.exp_name}> " + 
                  f"Start tuning: \n{pformat(self.model_config_obj.get_parameter_dict())}, " + 
                  f"\ncurrent time: {get_cur_time()}")

            res_file = self.model_config_obj.res_file
            if can_skip_trial(res_file):
                total_trials_num -= 1
                skipped_trials_num += 1
            else:
                try:
                    # Try all avaliable seeds and rerun `run_times` for `calc_mean_std`.
                    for seed in range(self.model_config_obj.run_times):          

                        # Note that seed has been changed here! 
                        # But `f_prefix` doesn't include `seed`, so the results will be saved to the same file.
                        self.model_config_obj.seed = seed

                        if not self.model_config_obj.log_on:
                            disable_logs()
                        else:
                            enable_logs()
                        
                        # Perform model tuning & record results to `res_file`.
                        # Note: this process will add/update some attrs in `self.model_config_obj`.
                        self.model_config_obj = self.train_func(self.model_config_obj, 
                                                                        display_params=False)
                        
                        # `seed + 1` to avoid zero division.
                        iter_time_estimate(f'\n\tSeed {seed}/{self.model_config_obj.run_times}', '',
                                           inner_start_time, seed + 1, self.model_config_obj.run_times)
                    
                    finished_trials_num += 1
                    iter_time_estimate(f'------ Trial {trial_idx} finished, ', '------',
                                       outer_start_time, finished_trials_num, total_trials_num)
                except Exception as e:

                    # Create a log file for the bug report.
                    log_file = f'log/{self.model_config_obj.model}-{self.model_config_obj.dataset}-{self.model_config_obj.exp_name}-{self.birth_time}.log'
                    mkdirs_p([log_file])

                    # TODO Write the error message to the log file.
                    error_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                    with open(log_file, 'a+') as f:
                        f.write(
                            f'\nTrial failed at {get_cur_time()} while running \n{pformat(self.hyperparam_dict)}\n at seed {seed}, error message:{error_msg}\n')
                        f.write(f'{"-" * 100}')

                    # If the log is disabled, enable it.
                    if not self.model_config_obj.log_on: 
                        enable_logs()
                    
                    # Print the error message on the screen.
                    print(f'Trial failed! error message: {error_msg}')
                    failed_trials_num += 1
                    continue
                
                # calculate mean and std value for this set of hyper-parameters (w.r.t different seeds)
                calc_mean_std(self.model_config_obj.res_file)
        
        print(f'\n{"="* 24 + " Grid Search Finished! " + "="* 24}\n'
              f'Successfully run {finished_trials_num} trials, skipped {skipped_trials_num} previous trials,'
              f'failed {failed_trials_num} trials.')
        if failed_trials_num > 0:
            print(f'Please check the log file `{log_file}` for bug reports.\n\n')

    # * ============================= Results Processing =============================

    def summarize(self, metric=EVAL_METRIC):
        """
        Summarize all the trials' results to a single file.
        One should use this function to summarize the results of **current** percentage setting.
        """
        exp_name = self.model_config_obj.exp_name
        model = self.model_config_obj.model
        dataset = self.model_config_obj.dataset
        train_percentage = self.model_config_obj.train_percentage

        res_file_list = self.tune_df_to_flist()

        out_prefix = f'{SUM_PATH}{model}/{dataset}/{model}_{dataset}<l{train_percentage:02d}><{exp_name}>'
        print(f'\n\nSummarizing expriment {exp_name}... ')

        try:
            sum_file = res_to_excel(res_file_list, out_prefix, f'avg_{metric}')
            print(f'Summary of {exp_name} finished. It have been saved to `{sum_file}`.')
        except Exception as e:
            error_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
            print(error_msg)
            print(f"!!!!!!Can't summarize {exp_name}!!!!!!")


    def tune_df_to_flist(self):
        """
        Return a list of result files corresponding to the tuning dataframe.
        """
        res_file_list = []
        tune_df = self.hyper_param_df
        tune_dict_list = tune_df.to_dict('records')

        # Each row of the dataframe stands for a trial (hyper-parameter combination).
        for row_idx in range(len(tune_df)):

            # Get the current trial's config & result file.
            self.model_config_obj.__dict__.update(tune_dict_list[row_idx])
            res_file = self.model_config_obj.res_file

            if os.path.exists(res_file):
                res_file_list.append(res_file)
        return res_file_list
    

    def summarize_by_percentage(self, dataset, model, metric=EVAL_METRIC):
        '''
        Summarize model results under all existing percentage settings.
        This method is used to summarize the results of **all** percentage settings, \
        and should be called after all the percentage settings are finished.
        '''
        model_res_path = f'{RES_PATH}{model}/{dataset}/'
        print(f'Summarizing--------{model_res_path}')

        # Get all existing results' dirs categorized by `train_percentage`.
        tp_dir_list = os.listdir(model_res_path)
        print(tp_dir_list)

        for train_percentage_dir in tp_dir_list:
            if ('.' not in train_percentage_dir) and len(train_percentage_dir) > 0 and train_percentage_dir != 'debug':
                print(f'Summarizing expriment: {train_percentage_dir}')

                res_path = f'{model_res_path}{train_percentage_dir}/'
                out_prefix = f'{model_res_path.replace(RES_PATH, SUM_PATH)}{model}_{dataset}<{train_percentage_dir}>AllRes_'
    
                mkdirs_p([out_prefix])
                res_file_list = [f'{res_path}{f}' for f in os.listdir(res_path)]

                try:
                    res_to_excel(res_file_list, out_prefix, f'avg_{metric}')
                    print(f'Summary of {res_path} finished.')
                except:
                    print(f'!!!!!!Cannot summarize {res_path}\tf_list:{os.listdir(res_path)}\n{res_path}'
                          f'was not summarized and skipped!!!!')


def iter_time_estimate(prefix, postfix, start_time, iters_finished, total_iters):
    """
    Generates progress bar AFTER the ith epoch.
    
    `prefix`: the prefix of printed string.
    `postfix`: the postfix of printed string.
    `start_time`: start time of the iteration.
    `iters_finished`: finished iterations.
    `max_i`: max iteration index.
    `total_iters`: total iteration to run, not necessarily \
            equals to max_i since some trials are skiped.
    """
    cur_run_time = time.time() - start_time
    total_estimated_time = cur_run_time * total_iters / iters_finished
    print(
        f'{prefix} [{time2str(cur_run_time)}/{time2str(total_estimated_time)}, {time2str(total_estimated_time - cur_run_time)} left] {postfix} [{get_cur_time()}]\n')


def dict_product(dct : dict):
    keys = dct.keys()
    return [dict(zip(keys, trial)) for trial in product(*dct.values())]


def add_tune_df_common_params(tune_df, param_dict):
    for param in param_dict:
        tune_df[param] = [param_dict[param] for _ in range(len(tune_df))]
    return tune_df


@timing
def run_multiple_process(func, func_arg_list):
    '''
    `func` : Function to run
    `func_arg_list` : An iterable object that contains several dict. Each dict has the input (**kwargs) of the tune_func
    '''
    process_list = []
    for func_arg in func_arg_list:
        process = multiprocessing.Process(target=func, kwargs=func_arg)
        process_list.append(process)
        process.start()
    
    # Ensure that a child process has completed before the main process.
    for process in process_list:
        process.join()


def res_to_excel(res_file_list : list, out_prefix : str, interested_metric : str):
    """
    Generate a summary excel file for all the result files in `res_file_list`.
    """
    sum_dict_list = []
    for res_file in res_file_list:
        if os.path.isfile(res_file) and res_file[-3:] == 'txt':

            # One `res_file` corresponds to one trial & one `sum_dict` & one `param_dict`.
            sum_dict, param_dict = {}, {}
            with open(res_file, 'r') as f:
                res_lines = f.readlines()

                # Find the avg & std string in the txt file generated by `calc_mean_std`.
                for res_line in res_lines:
                    if res_line[0] == '{':
                        dct : dict = ast.literal_eval(res_line.strip('\n'))

                        if 'dataset' in dct.keys():  # the *last* line of parameters in the txt file
                            param_dict = dct.copy()
                            param_dict.pop('seed')
                        elif 'avg_' in list(dct.keys())[0]:  # mean results
                            avg_res_dict = dict(zip(dct.keys(), [float(v) for v in dct.values()]))
                        elif 'std_' in list(dct.keys())[0]:
                            std_res_dict = dict(zip(dct.keys(), [float(v) for v in dct.values()]))
                        else:
                            pass

                # Generate the summary dict: model params + mean results + std results
                try:
                    sum_dict.update(param_dict)
                    sum_dict.update(avg_res_dict)
                    sum_dict.update(std_res_dict)
                except:
                    print(f'!!!!File {f.name} is not summarized, skipped!!!!')
                    continue
                
                # Save `param_dict` to deal with NA columns.
                sum_dict['param_dict'] = param_dict     
                sum_dict_list.append(sum_dict)

    sum_df = pd.DataFrame.from_dict(sum_dict_list).sort_values(interested_metric, ascending=False, ignore_index=True)
    max_res = sum_df.loc[0, interested_metric]

    # ! Use mean and std to generate final results.
    metric_names = [cname[4:] for cname in sum_df.columns if 'avg' in cname]
    for metric_name in metric_names:
        sum_df['avg_' + metric_name] = sum_df['avg_' + metric_name].apply(lambda x: f'{x:.2f}')
        sum_df['std_' + metric_name] = sum_df['std_' + metric_name].apply(lambda x: f'{x:.2f}')
        sum_df[metric_name] = sum_df['avg_' + metric_name] + '±' + sum_df['std_' + metric_name]
        sum_df = sum_df.drop(columns=['avg_' + metric_name, 'std_' + metric_name])
    
    # ! Deal with NA columns. ?
    for col in sum_df.columns[sum_df.isnull().any()]:
        for index, row in sum_df.iterrows():
            sum_df.loc[index, col] = row.param_dict[col]            # ? dict[col]

    # Reorder column order list : move param_dict to the end.
    col_names = list(sum_df.columns) + ['param_dict']
    col_names.remove('param_dict')          # Remove first occurrence of value.
    sum_df = sum_df[col_names]

    # Save to excel file.
    mkdirs_p([out_prefix])
    res_file = f'{out_prefix}{max_res:.2f}.xlsx'
    sum_df.to_excel(res_file)

    return res_file


def calc_mean_std(res_file):
    """
    Load results from `res_file` and calculate mean and std value for the corresponding trial.
    """
    if os.path.exists(res_file):
        out_df, metric_list = load_dict_results(res_file)
    else:
        print('Result file missing, skipped!!')
        return
    
    mean_res = out_df[metric_list].mean()
    std_res = out_df[metric_list].std()

    # Convert the values of certain metrics to percentage.
    for metric in metric_list:
        for substring in ['acc', 'Acc', 'AUC', 'ROC', 'f1', 'F1']:
            if substring in metric:
                mean_res[metric] = mean_res[metric] * 100
                std_res[metric] = std_res[metric] * 100
    
    mean_dict = dict(zip([f'avg_{metric}' for metric in metric_list], 
                                [f'{mean_res[metric]:.2f}' for metric in metric_list]))
    std_dict = dict(zip([f'std_{metric}' for metric in metric_list], 
                                [f'{std_res[metric]:.2f}' for metric in metric_list]))

    # Append the mean and std value to the result file.
    with open(res_file, 'a+') as f:
        f.write('\n\n' + '#' * 10 + 'AVG RESULTS' + '#' * 10 + '\n')

        for metric in metric_list:
            f.write(f'{metric}: {mean_res[metric]:.4f} ({std_res[metric]:.4f})\n')
        
        f.write('#' * 10 + '###########' + '#' * 10)
    
    # for `res_to_excel`
    write_nested_dict({'avg': mean_dict, 'std': std_dict}, res_file)


def load_dict_results(res_file):
    """
    Read results of a trial from `res_file` and return a dataframe and a list of metrics.
    """
    trial_idx = 0
    parameters = {}
    metric_list = None

    with open(res_file, 'r') as f:
        res_lines = f.readlines()

        for res_line in res_lines:
            if res_line[0] == '{':                  # ensure the line is a dict string
                dct : dict = ast.literal_eval(res_line.strip('\n'))

                if 'model' in dct.keys():             # the line contains parameters: lr, seed, etc.
                    trial_idx += 1
                    parameters[trial_idx] = res_line.strip('\n')      # string of param dict
                elif 'avg_' in list(dct.keys())[0] or 'std_' in list(dct.keys())[0]:
                    pass                            # ?
                else:                               # the line contains results: test_acc, val_acc, etc.
                    if metric_list is None:
                        metric_list = list(dct.keys())
                        for metric in metric_list:          # init metric dict
                            exec(f'{metric.replace("-", "")}=dict()')
                    
                    for metric in metric_list:
                        exec(f'{metric.replace("-", "")}[trial_idx]=float(dct[\'{metric}\'])')
    
    # `metric_set_str` is like: 'val_acc, test_acc, ...'
    metric_list_str = str(metric_list).replace('\'', '').strip('[').strip(']').replace("-", "")
    exec(f'out_list_ = [parameters, {metric_list_str}]', globals(), locals())

    out_list = locals()["out_list_"]
    out_df = pd.DataFrame.from_records(out_list).T      # row: parameters, metrics | col: eid
    out_df.columns = ['parameters', *metric_list]

    return out_df, metric_list


def can_skip_trial(res_file):
    """
    Judge whether the trial can be skipped by checking the previous results.

    Case 1: Previous results exists and summarized => skip => Return True.
    Case 2: Previous results exists but unfinished => clear and rerun => Clear and return False.
    Case 3: Previous results doesn't exist => run => Return False.
    """
    if os.path.isfile(res_file):
        with open(res_file, 'r') as f:
            for line in f.readlines():
                if line[0] == '{':
                    d : dict = ast.literal_eval(line.strip('\n'))
                    if 'avg_' in list(d.keys())[0]:
                        # ! Case 1: Previous results exists and summarized => skip => Return True
                        print('Found previous results! Skipping current trial... ')
                        return True
            # ! Case 2: Previous results exists but unfinished => clear and rerun => Clear and return False
            os.remove(res_file)
            print(f'Previous results exists but unfinished! Resuming from {res_file}... ')
            return False
    else:
        # ! Case 3: Previous results doesn't exist => run => Return False
        print("Previous results doesn't exist! Performing current trial... ")
        return False

