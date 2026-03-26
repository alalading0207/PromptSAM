import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.loops import EpochBasedTrainLoop   
from mmengine.runner.amp import autocast
from mmengine.runner.utils import calc_dynamic_intervals
from .adapt_base_loop import BaseLoop_adapt



@LOOPS.register_module()
class EpochBasedTrainLoop_adapt(BaseLoop_adapt):
    """Custom Epoch-Based Train Loop with additional functionality."""

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        
        # get stage from runner's config, default to 'stage1' if not specified
        self.stage = runner.cfg.get('stage', 'stage1') 
        print(f"Initializing training in stage: {self.stage}")
        super().__init__(runner, dataloader, stage = self.stage)

        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader['target'])   
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        self.stop_training = False
        if hasattr(self.dataloader['target'].dataset, 'metainfo'):  
            self.runner.visualizer.dataset_meta = \
                self.dataloader['target'].dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader["target"].dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()   
        for idx, (source_batch, target_batch) in enumerate(zip(self.dataloader['source'], self.dataloader['target'])):  
        # for idx, (target_batch, source_batch) in enumerate(zip(self.dataloader['target'], self.dataloader['source'])):
            data_batch = {'source': source_batch, 'target': target_batch}
            self.run_iter(idx, data_batch)   

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Dict[str, dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss. 

        # train_step training
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(      #
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]



class WrappedDataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    @property
    def dataset(self):
        return self.dataloader['target'].dataset

    @property
    def metainfo(self):
        return getattr(self.dataloader['target'].dataset, 'metainfo', None)