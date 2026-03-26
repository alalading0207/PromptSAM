# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..task_modules.prior_generators import MlvlPointGenerator
from .positional_encoding import SinePositionalEncoding
from .transformer import Mask2FormerTransformerEncoder

# 常用于特征金字塔网络FPN的Transformer结合使用
@MODELS.register_module()
class MSDeformAttnPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention.具有多尺度可变形注意力的像素解码器
    这个类能够处理输入的多尺度特征图，应用多尺度可变形注意力机制，并输出掩码特征和多尺度特征图，用于后续的目标检测或分割任务
    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer
            encoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int],
                                    Tuple[int]] = [256, 512, 1024, 2048],
                 strides: Union[List[int], Tuple[int]] = [4, 8, 16, 32],
                 feat_channels: int = 256,
                 out_channels: int = 256,
                 num_outs: int = 3,   # 输出尺度的数量
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 encoder: ConfigType = None,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.strides = strides      # 4 8 16 32 64
        self.num_input_levels = len(in_channels)    # 5 个256特征
        self.num_encoder_levels = \
            encoder.layer_cfg.self_attn_cfg.num_levels  # 3层layer
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []  # 构造输入卷积和编码器
        
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1, self.num_input_levels - self.num_encoder_levels - 1,-1):     # 4、3、2
            input_conv = ConvModule(   # 为每个输入级别创建卷积模块并添加到input_conv
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)      # 三层Conv2d(256, 128）
        
        # 初始化Transformer编码器和位置编码器
        self.encoder = Mask2FormerTransformerEncoder(**encoder)     # 三层DeformableDetrTransformerEncoderLayer(每层一个注意力和一个ffn)
        self.postional_encoding = SinePositionalEncoding(**positional_encoding)     # inePositionalEncoding(num_feats=64, temperature=10000, normalize=True, scale=6.283185307179586, eps=1e-06)
        
        # high resolution to low resolution 级别编码：为每个编码级别创建嵌入
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)       # Embedding(3, 128)
        
        # 构造侧向卷积和输出卷积
        # fpn-like structure  
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):     # 1\0
            lateral_conv = ConvModule(  # 为每个级别创建侧向卷积模块，并将其添加到 lateral_convs 中。
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(  # 为每个级别创建输出卷积模块，并将其添加到 output_convs 中
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        # 构造掩码特征卷积俄点生成器
        self.mask_feature = Conv2d(  # 创建一个卷积层，用于生成掩码特征
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)    # Conv2d(128, 256)
        
        self.num_outs = num_outs  # 初始化多层次点生成器，用于生成多尺度特征图的点
        self.point_generator = MlvlPointGenerator(strides)
    # 使用 Xavier 初始化和 Caffe2 Xavier 初始化方法初始化卷积层的权重，使用正态分布初始化嵌入层的权重，初始化 Transformer 编码器的权重
    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for m in self.encoder.layers.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self, feats: List[Tensor]) -> Tuple[Tensor, Tensor]:   # 输入为特征图列表
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            tuple: A tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - multi_scale_features (list[Tensor]): Multi scale \
                        features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]      # feats是多尺度序列
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []  
        
        
        # 编码器输入准备
        for i in range(self.num_encoder_levels):        # 3层encoder
            level_idx = self.num_input_levels - i - 1   # input为5层  level_idx=4/3/2
            feat = feats[level_idx]                     # 从最后一个开始[1, 256, 16, 16] / [1, 256, 32，32] / [1, 256, 64，64]
            feat_projected = self.input_convs[i](feat)  # 进行卷积投影，减少通道数[1, 128,  16, 16] / [1, 128, 32，32] / [1, 128, 64，64]
            feat_hw = torch._shape_as_tensor(feat)[2:].to(feat.device)      # [16, 16] / [32，32] / [64，64]

            # no padding
            padding_mask_resized = feat.new_zeros((batch_size, ) + feat.shape[-2:], dtype=torch.bool)  # [1, 16, 16] 全为false
            pos_embed = self.postional_encoding(padding_mask_resized).to(feat.dtype)     # 位置编码[1, 128, 16, 16]
            level_embed = self.level_encoding.weight[i]                                  # 层级编码[128] / [128] / [128]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed     # [1, 128, 16, 16]
            
            # (h_i * w_i, 2) 生成当前层级的的参考点
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)         # # [256, 2] 生成了256个点 / 1024 / 4096
            # normalize
            feat_wh = feat_hw.unsqueeze(0).flip(dims=[0, 1])    # size[1, 2]  仍为[16, 16] / [32，32]
            factor = feat_wh * self.strides[level_idx]          # [16, 16]*64=[1024，1024]  [32，32]*32=[1024，1024]  [64, 64]*16=[1024，1024] 
            reference_points = reference_points / factor        # 点坐标在0.0312至0.0938之间  共256个点  / 1024个点  / 4096个点

            # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)     # [1, 256, 128]  压扁后换位
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)   # [1, 256, 128]  压扁后换位  / [1, 1024, 128]  / [1, 4096, 128] 
            padding_mask_resized = padding_mask_resized.flatten(1)          # [1, 256] / [1, 1024]

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat_hw)
            reference_points_list.append(reference_points)


        # 编码器输入拼接 所有层级拼接在一起
        # shape (batch_size, total_num_queries), total_num_queries=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)     # 填充掩码拼接 torch.Size([1, 5376])  256+1024+4096=5376
        # shape (total_num_queries, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)   # 特征图拼接torch.Size([1, 5376, 128])
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=1)   # 位置编码拼接
        
        # shape (num_encoder_levels, 2), from low resolution to high resolution
        num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]   # 256 1024 2048分别是每一级的queries数
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)  # 记录每个层级特征图的空间形状
        
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)  计算每个层级的起始索引
        # prod(1)计算每行的积[256,1024,4096] ，cumsum(0)计算乘积的累计和[256,1280,5376]。最后的tensor[0，256，1280]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) 
        
        reference_points = torch.cat(reference_points_list, dim=0)  # [5376 2] 共5376个点
        # size[1, 5376, 3, 2])   reference_points[None, :, None]让[5376 2]变成了[1,5376,1,2]
        reference_points = reference_points[None, :, None].repeat(batch_size, 1, self.num_encoder_levels, 1)  
        # size[1, 3, 2] 并且全部为1  只是根据reference_points的数据类型和设备来构建
        valid_radios = reference_points.new_ones((batch_size, self.num_encoder_levels, 2))      


        # shape (num_total_queries, batch_size, c) 编码器前向传播
        memory = self.encoder(
            query=encoder_inputs,               # 三个level的特征连起来 torch.Size([1, 5376, 128]) 
            query_pos=level_positional_encodings, # encoder_inputs的位置编码，torch.Size([1, 5376, 128])
            key_padding_mask=padding_masks,     # torch.Size([1, 5376]) 目前全为false
            spatial_shapes=spatial_shapes,      # torch.Size([1, 5376, 3, 2]) 里面是[16,16]/[32,32]/[64,64]
            reference_points=reference_points,  # torch.Size([1, 5376, 3, 2])  1189个点，分3个level
            level_start_index=level_start_index,# 从[0，256，1280]这三个位置开始下一个level
            valid_ratios=valid_radios)          # size[1, 3, 2] 并且全部为1
        # (batch_size, c, num_total_queries)
        memory = memory.permute(0, 2, 1)        # torch.Size([1, 5376, 128])--> torch.Size([1, 128, 5376])


        # 编码器输出处理
        # from low resolution to high resolution  
        outs = torch.split(memory, num_queries_per_level, dim=-1)     # 还原为[1,128,256][1,128,1024][1,128,2048]
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
            spatial_shapes[i][1]) for i, x in enumerate(outs)         # 还原为[1,128,16,16][1,128,32，32][1,128,64,64]
        ]       

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1): # 1/0
            x = feats[i]                        # torch.Size([1, 256, 128, 128]) / [1, 256, 256, 256]
            cur_feat = self.lateral_convs[i](x) # torch.Size([1, 128, 128, 128]) / [1, 128, 256, 256]
            # 尺度融合    将outs的最后一个重采样，先采样为 128,128 ，再采样成 256，256
            y = cur_feat + F.interpolate(outs[-1],size=cur_feat.shape[-2:],mode='bilinear',align_corners=False)        
            y = self.output_convs[i](y)
            outs.append(y)                  # 为out追加了两个大尺度的
            
        multi_scale_features = outs[:self.num_outs] # 提取前三个为multi_scale_features  都是128通道的  [1,128,16,16][1,128,32，32][1,128,64,64]
        mask_feature = self.mask_feature(outs[-1])  # 让outs的最后一层从[1, 128, 256, 256]变为[1, 256, 256, 256]
        
        return mask_feature, multi_scale_features   # 最后一层特征和多尺度特征
