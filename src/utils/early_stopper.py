"""
Early stopping provided by DGL.
"""
import torch


class EarlyStopping:
    def __init__(self, patience=10, ckpoint_path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.best_model = None
        self.should_stop = False
        self.ckpoint_path = ckpoint_path

    def _update_best_record(self, score, epoch, model):
        self.counter = 0
        self.best_score = score
        self.best_epoch = epoch
        self.best_model = model
        self.save_checkpoint(model)

    def step(self, acc, model, epoch):
        """
        Return True if should stop training (exceed patience).
        """
        score = acc
        if self.best_score is None:
            self._update_best_record(score, epoch, model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter}/{self.patience}, best_val_score:{self.best_score:.4f} at E{self.best_epoch}')
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self._update_best_record(score, epoch, model)

        return self.should_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.ckpoint_path)

