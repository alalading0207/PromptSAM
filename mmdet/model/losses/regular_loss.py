
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.registry import MODELS
import torch



    
def mrent(pred: Tensor) -> Tensor:
    p = torch.sigmoid(pred)  # (num_query, H*W)
    entropy = - (p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
    loss = entropy.mean()
    return loss


def mrkld(pred: Tensor) -> Tensor:
    p = torch.sigmoid(pred)  # (num_query, H*W)
    kl_div = p * torch.log(2 * p + 1e-6) + (1 - p) * torch.log(2 * (1 - p) + 1e-6)
    loss = kl_div.mean()
    return loss


@MODELS.register_module()
class RegularLoss(nn.Module):


    def __init__(self,
                 regular_type: str = 'MRKLD',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert regular_type in ['MRENT', 'MRKLD'], "regular_type must be MRENT or MRKLD"
        self.regular_type = regular_type
        self.loss_weight = loss_weight

        if self.regular_type == 'MRENT':
            self.cls_criterion = mrent     
        else:                   # 'MRKLD'
            self.cls_criterion = mrkld     

    def forward(self,
                pred: Tensor) -> Tensor:

        loss = self.loss_weight * self.cls_criterion(pred)
        return loss


