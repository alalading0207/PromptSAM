# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from mmdet.registry import MODELS
from .utils import weighted_loss

# kd_loss = F.kl_div(
#     F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (T * T)


@weighted_loss
def knowledge_distillation_kl_div_loss(pred: Tensor,
                                       soft_label: Tensor,
                                       T: int,
                                       ignore_index: float = 250,
                                       detach_target: bool = True) -> Tensor:
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()


    target_ = F.softmax(soft_label / T, dim=1)
    
    target = target_.clamp(min=1e-6, max=1.0-1e-6)   # teacher
    if detach_target:
        target = target.detach()
    # pred_ = pred.clamp(min=-70, max=70)

    kd_loss_ = F.kl_div(F.log_softmax(pred / T, dim=1), target, reduction='none')
    

    valid_mask = (soft_label != ignore_index).float()
    kd_loss = kd_loss_ * valid_mask  

    kd_loss = kd_loss.sum() / (valid_mask.sum() + 1e-6)  
    kd_loss = kd_loss * (T * T) 


    if torch.isnan(kd_loss):
        print('target.max',target.max())
        print('target.min',target.min())
        print('target_.max',target_.max())
        print('target_.min',target_.min())
        print('pred.max',pred.max())
        print('pred.min',pred.min())
        print('F.log_softmax(pred / T, dim=1).max', F.log_softmax(pred / T, dim=1).max)
        print('F.log_softmax(pred / T, dim=1).min', F.log_softmax(pred / T, dim=1).min)
        print('kd_loss_.max', kd_loss_.max)
        print('kd_loss_.min', kd_loss_.min)
        print('kd_loss.sum()',kd_loss.sum())
        print('valid_mask.sum() + 1e-6', valid_mask.sum() + 1e-6)

        kd_loss = torch.tensor(0.0, device=kd_loss.device)


    return kd_loss


@MODELS.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 ignore_index: float = 250,
                 T: int = 10,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert T >= 1
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.T = T
        self.loss_weight = loss_weight
        
        
    def forward(self,
                pred: Tensor,
                soft_label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            T=self.T,
            ignore_index=self.ignore_index,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_kd
