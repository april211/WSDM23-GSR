from time import time
from abc import abstractmethod, ABCMeta

import torch

from utils.data_utils import get_stochastic_loader
from utils.util_funcs import print_log
from utils.evaluation import eval_classification, save_results, accuracy, logits_2_pred
from utils.early_stopper import EarlyStopping
from utils.conf_utils import ModelConfigABS
# Copyright


class NodeClassificationExpABC(metaclass=ABCMeta):
    def __init__(self, model, g, features, optimizer, stopper, loss_func, data_obj, cf):
        self.g = g.cpu()
        self.features = features
        
        self.model = model
        self.optimizer = optimizer
        self.stopper : EarlyStopping = stopper
        self.loss_func = loss_func
        self.evaluator = lambda logits, target: accuracy(logits_2_pred(logits), target)

        self.cf : ModelConfigABS = cf
        self.device = cf.device
        self.epochs = cf.epochs
        self.n_class = cf.n_class

        self.train_idx, self.val_idx, self.test_idx = \
            [data_idx.to(cf.device) for data_idx in [data_obj.train_idx, data_obj.val_idx, data_obj.test_idx]]
        self.labels = data_obj.labels.to(cf.device)

    @abstractmethod
    def _train(self):
        """
        Train `self.model` for one epoch and return its training loss and accuracy.
        """
        return None, None

    @abstractmethod
    def _eval(self):
        """
        Evaluate `self.model` and return its val accuracy and test accuracy.
        """
        return None, None

    def run_exp(self):
        """
        Complete Node Classification experiment and return the best model.
        """
        for epoch in range(self.epochs):
            t0 = time()

            loss, train_acc = self._train()
            val_acc, test_acc = self._eval()

            print_log({'Epoch': epoch, 'Time': time() - t0, 'Loss': loss,
                       'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc})
            
            if self.stopper is not None:
                if self.stopper.step(val_acc, self.model, epoch):
                    print(f"Early stopped at epoch {epoch}, " + 
                          f"will return the best model from epoch {self.stopper.best_epoch}.")
                    break
        
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.ckpoint_path))
        
        return self.model

    def eval_and_save(self):
        """
        Evaluate `self.model` and save the results.
        Note: this method should be called after `self.run()`.
        """
        val_acc, test_acc = self._eval()
        res_dict = {'test_acc': f'{test_acc:.4f}', 'val_acc': f'{val_acc:.4f}'}

        if self.stopper is not None: 
            res_dict['best_epoch'] = self.stopper.best_epoch

        save_results(self.cf, res_dict)


class FullBatchNCExp(NodeClassificationExpABC):
    def __init__(self, **kwargs):
        super(FullBatchNCExp, self).__init__(**kwargs)
        self.g = self.g.to(self.device)
        self.features = self.features.to(self.device)

    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.g, self.features)
        loss = self.loss_func(logits[self.train_idx], self.labels[self.train_idx])
        train_acc = self.evaluator(logits[self.train_idx], self.labels[self.train_idx])
        loss.backward()
        self.optimizer.step()
        return loss.item(), train_acc

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        logits = self.model(self.g, self.features)
        val_acc = self.evaluator(logits[self.val_idx], self.labels[self.val_idx])
        test_acc = self.evaluator(logits[self.test_idx], self.labels[self.test_idx])
        return val_acc, test_acc


class StochasticNCExp(NodeClassificationExpABC):
    def __init__(self, **kwargs):
        super(StochasticNCExp, self).__init__(**kwargs)
        self.train_loader = get_stochastic_loader(self.g, self.train_idx.cpu(), self.cf.cla_batch_size, self.cf.num_workers)
        self.val_loader = get_stochastic_loader(self.g, self.val_idx.cpu(), self.cf.cla_batch_size, self.cf.num_workers)
        self.test_loader = get_stochastic_loader(self.g, self.test_idx.cpu(), self.cf.cla_batch_size, self.cf.num_workers)

    def _train(self):
        self.model.train()
        loss_list = []
        pred = torch.ones_like(self.labels).to(self.device) * -1
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(self.train_loader):
            blocks = [b.to(self.cf.device) for b in blocks]
            input_features = self.features[input_nodes].to(self.device)
            output_labels = self.labels[output_nodes].to(self.device)
            out_logits = self.model(blocks, input_features, stochastic=True)
            loss = self.loss_func(out_logits, output_labels)
            pred[output_nodes] = torch.argmax(out_logits, dim=1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_id + 1 < len(self.train_loader) or len(self.train_loader) == 1:
                # Metrics of the last batch (high variance) shouldn't be added
                loss_list.append(loss.item())
        train_acc, train_f1, train_mif1 = eval_classification(pred[self.train_idx], self.train_y, n_class=self.n_class)

        return sum(loss_list) / len(loss_list), train_acc

    @torch.no_grad()
    def _eval(self):
        def _eval_model(loader, val_x, val_y):
            pred = torch.ones_like(self.labels).to(self.device) * -1
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(loader):
                blocks = [b.to(self.cf.device) for b in blocks]
                input_features = self.features[input_nodes].to(self.device)
                out_logits = self.model(blocks, input_features, stochastic=True)
                pred[output_nodes] = torch.argmax(out_logits, dim=1)
            acc, val_f1, val_mif1 = eval_classification(pred[val_x], val_y, n_class=self.n_class)
            return acc

        self.model.eval()
        val_acc = _eval_model(self.val_loader, self.val_idx, self.val_y)
        test_acc = _eval_model(self.test_loader, self.test_idx, self.test_y)
        return val_acc, test_acc

