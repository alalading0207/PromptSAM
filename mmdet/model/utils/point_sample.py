# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import point_sample
from torch import Tensor


def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
    """Estimate uncertainty based on pred logits.

    We estimate uncertainty as L1 distance between 0.0 and the logits
    prediction in 'mask_preds' for the foreground class in `classes`.

    Args:
        mask_preds (Tensor): mask predication logits, shape (num_rois,
            num_classes, mask_height, mask_width).

        labels (Tensor): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score,
            shape (num_rois, 1, mask_height, mask_width)
    """
    if mask_preds.shape[1] == 1:
        gt_class_logits = mask_preds.clone()
    else:
        inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
        gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
    return -torch.abs(gt_class_logits)


def get_uncertain_point_coords_with_randomness(
        mask_preds: Tensor, labels: Tensor, num_points: int,        # 12544
        oversample_ratio: float, importance_sample_ratio: float) -> Tensor:
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (Tensor): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (float): Oversampling parameter.  在计算不确定性之前要采样多少点
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.   importance_sample_ratio这个比率的不确定点，剩下的都是随机采样

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_preds.shape[0]        # 5
    num_sampled = int(num_points * oversample_ratio)    # 12544*3=37632
    point_coords = torch.rand(
        batch_size, num_sampled, 2, device=mask_preds.device, dtype=mask_preds.dtype)   # 随机生成 num_sampled 个点
    point_logits = point_sample(mask_preds, point_coords)  # 从mask_preds中获取随即点的预测值   mask_preds大小是[5, 1, 120, 120]
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_uncertainties = get_uncertainty(point_logits, labels)     # 计算每个采样点的不确定性值 torch.Size([5, 1, 37632])
    num_uncertain_points = int(importance_sample_ratio * num_points)   # 可选的不确定点数量 9408
    num_random_points = num_points - num_uncertain_points              # 可选的的随即点数量 3136
    # 提取这些不确定点坐标
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]   # torch.Size([5, 9408])
    shift = num_sampled * torch.arange(batch_size, dtype=torch.long, device=mask_preds.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(batch_size, num_random_points, 2, device=mask_preds.device, dtype=mask_preds.dtype)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1) # 将随机点和不确定点 的 点坐标 合并
    return point_coords
