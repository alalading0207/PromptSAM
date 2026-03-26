from typing import Any, Dict, Union

from torch.utils.data import DataLoader
from mmengine.runner.loops import BaseLoop


class BaseLoop_adapt(BaseLoop): 

    def __init__(self, runner, dataloader: Dict[str, Union[DataLoader, Dict]], stage: str) -> None:
        self._runner = runner
        self.stage = stage
        self.dataloader = {}  
       
        # construct dataloader
        for name, dataloader in dataloader.items():
            if isinstance(dataloader, dict):
                diff_rank_seed = runner._randomness_cfg.get(
                    'diff_rank_seed', False)
                self.dataloader[name] = runner.build_dataloader(
                    dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            else:
                self.dataloader[name] = dataloader


    @property
    def runner(self):
        return self._runner


    def run(self) -> Any:
        for name, dataloader in self.dataloader.items():
            print(f"Running dataloader {name}")
        return None
