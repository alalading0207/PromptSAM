import copy
import warnings
import einops
import numpy as np
import torch
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.ops import point_sample
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from peft import get_peft_config, get_peft_model
from torch import nn, Tensor
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, \
    SamPromptEncoder, SamModel, SamVisionEncoderOutput
from typing import List, T, Tuple, Optional, Dict, Union
from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding, Mask2Former, Mask2FormerHead, \
    MaskFormerFusionHead, BaseDetector
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances, multi_apply, \
    get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample, OptSampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList, reduce_mean
import torch.nn.functional as F

from mmpretrain.models import LayerNorm2d


@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



@MODELS.register_module()
class RSPrompterQuery(Mask2Former):
    def __init__(
            self,
            shared_image_embedding,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})   # get peft_config 
        # call the constructor of the parent class Mask2Former
        super().__init__(*args, **kwargs)    
        self.decoder_freeze = decoder_freeze    # false
        self.with_mask2formerhead = False if isinstance(self.panoptic_head, RSMask2FormerHead) else True  # false
        self.shared_image_embedding = MODELS.build(shared_image_embedding)      # used for generation and sharing of image embeddings
        
        #  frozen backbone
        self.frozen_modules = []
        if peft_config is None:  
            self.frozen_modules += [self.backbone]
        # frozen mask_decoder 
        if self.decoder_freeze:    # false
            self.frozen_modules += [
                self.shared_image_embedding,
                self.panoptic_head.mask_decoder,
            ]
        self._set_grad_false(self.frozen_modules)  

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False


    # generate image-level positional embeddings
    def get_image_wide_positional_embeddings(self, size):  
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype    # torch.float16
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype) 
        y_embed = grid.cumsum(dim=0) - 0.5  
        x_embed = grid.cumsum(dim=1) - 0.5  
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))     # RSSamPositionalEmbedding
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width 


    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:  
        vision_outputs = self.backbone(batch_inputs)      
        
        # backbone input type SamVisionEncoderOutput
        if isinstance(vision_outputs, SamVisionEncoderOutput): 
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]  
        elif isinstance(vision_outputs, tuple):     # true
            image_embeddings = vision_outputs[0]   
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        # positional positional_embedding for vision_outputs
        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1]) 
        # repeat with batch size
        batch_size = image_embeddings.shape[0] 
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1) 
        # vision_outputs to neck
        x = self.neck(vision_hidden_states) 
        return x, image_embeddings, image_positional_embeddings


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:

        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)  # extract features and embeddings

        if self.with_mask2formerhead:
            losses = self.panoptic_head.loss(x, batch_data_samples)   
        else:
            losses = self.panoptic_head.loss(x, batch_data_samples,   #  x after neck, multi-scale features
                                             image_embeddings=image_embeddings, # sam encoder output embeddings
                                             image_positional_embeddings=image_positional_embeddings)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        # x muiti-scale features / image_embeddings  / image_positional_embeddings
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)   

        if self.with_mask2formerhead:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(x, batch_data_samples)  
        else:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                x, batch_data_samples,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings
            )
        # results_list[]'ins_result'] contains bbpxes/labels/masks/metainfo/scores
        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=rescale) 
        results = self.add_pred_to_datasample(batch_data_samples, results_list)     # add results_list to pred_instance of batch_data_samples 

        return results


@MODELS.register_module()
class RSMask2FormerHead(Mask2FormerHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            decoder_plus,
            with_sincos=True,
            per_pointset_point=1,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_plus = decoder_plus
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity     # none
        self.target_embedding = target_embedding             # none
        self.output_attentions = output_attentions           # none

        self.mask_decoder = MODELS.build(mask_decoder)
        # prompt_encoder
        prompt_encoder = dict(
            type='RSSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        prompt_encoder = MODELS.build(prompt_encoder)       # prompt_encoder contains mask\non-mask\point embeddings\non-point embeddings
        prompt_encoder.init_weights()       # none 
        if self.decoder_plus:
            self.sam_mask_embed = prompt_encoder.prompt_encoder.mask_embed  # sam.prompt_encoder.mask_embed deal mask data
        else:
            self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed
            del self.mask_embed
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos

        self.feat_channels = kwargs['feat_channels']
        self.out_channels = kwargs['out_channels']
        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(     
            nn.Linear(self.feat_channels, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels // 2, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels // 2, self.out_channels * num_sincos * per_pointset_point)
        )
        del self.cls_embed      # del parent cls_embed
        self.cls_embed = nn.Sequential(     # redefine cls_embed for binary classification
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.num_classes + 1))
        

        # ===== PromptSAM Random Query =====
        # self.random_query = True
        # if self.random_query:   
        #     for p in self.query_feat.parameters():
        #         p.requires_grad = False
        #     for p in self.query_embed.parameters():
        #         p.requires_grad = False


        # ===== PromptSAM 50% Random + 50% Learnrd Query =====
        # self.freeze_queries = 15
        # self._freeze_partial_queries()
    

    def _freeze_partial_queries(self):
        freeze_n = self.freeze_queries
        def grad_mask(grad):
            mask = torch.ones_like(grad)
            mask[:freeze_n] = 0.0   # first half queries are frozen, no gradient update
            return grad * mask

        self.query_embed.weight.register_hook(grad_mask)
        self.query_feat.weight.register_hook(grad_mask)


    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,  # decoder_out actually is query_feat
                      attn_mask_target_size: Tuple[int, int],
                      image_embeddings=None,     
                      image_positional_embeddings=None
                      ) -> Tuple[Tensor]:
        
        img_bs = image_embeddings.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]  

        # class prediction 
        decoder_out = self.transformer_decoder.post_norm(decoder_out) 
        cls_pred = self.cls_embed(decoder_out)         

        # point embedings 
        point_embedings = self.point_emb(decoder_out) 
        point_embedings = einops.rearrange(point_embedings, 'b n_set (n_point c) -> b n_set n_point c', n_point=self.per_pointset_point)    
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        # sparse embeddings
        sparse_embeddings = einops.rearrange(point_embedings, 'b n_set n_point c -> (b n_set) n_point c') 
        sparse_embeddings = sparse_embeddings.unsqueeze(1)     

        # dense embeddings
        if self.decoder_plus:
            mask_embed = self.mask_embed(decoder_out) 
            mask_pred_plus = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature) 
            input_masks = mask_pred_plus.detach()
            input_masks = einops.repeat(input_masks, 'b n h w -> (b n) c h w', c=1)
            dense_embeddings = self.sam_mask_embed(input_masks) 
        else:
            mask_pred_plus = None
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(img_bs, -1, image_embedding_size[0], image_embedding_size[1])

        image_embeddings = torch.repeat_interleave(image_embeddings, repeats=self.num_queries, dim=0)   
        image_positional_embeddings = torch.repeat_interleave(image_positional_embeddings, repeats=self.num_queries, dim=0)   
        
        # call sam model
        mask_pred, iou_predictions, mask_dencoder_attentions = self.mask_decoder(       
            image_embeddings=image_embeddings,             
            image_positional_embeddings=image_positional_embeddings,    
            sparse_prompt_embeddings=sparse_embeddings,    
            dense_prompt_embeddings=dense_embeddings,      
            multimask_output=self.multimask_output,        
            attention_similarity=self.attention_similarity, # none
            target_embedding=self.target_embedding,         # none
            output_attentions=self.output_attentions,       # none
        )
        mask_pred = mask_pred.reshape(img_bs, -1, *mask_pred.shape[-2:]) 
        
        # attention mask generation: scale mask_pred and mask_pred_plus to target size attn_mask_target_size
        if not self.decoder_plus:
            h, w = mask_pred.shape[-2:]
            attn_mask_pred = mask_pred.reshape(img_bs, -1, h, w)
        else:
            attn_mask_pred = mask_pred_plus
        attn_mask = F.interpolate(attn_mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)  
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)    
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        # return class prediction, mask prediction, attention mask prediction, and coarse mask prediction 
        return cls_pred, mask_pred, attn_mask, mask_pred_plus     

    def forward(self, x: List[Tensor],      
                batch_data_samples: SampleList,
                image_embeddings=None,
                image_positional_embeddings=None
                ) -> Tuple[List[Tensor]]:
        
        # Mask2Former Pixel decoder
        # the highest resolution mask_features to final predct, three multi-scale features to transformer decoder for cross-attention.
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)   
        
        
        # multi-scale features to transformer decoder
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):   
            
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])   
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)    # learnable level embedding for each multi-scale feature
            decoder_input = decoder_input + level_embed                 

            # positional encoding for multi-scale features
            mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)    
            decoder_positional_encoding = self.decoder_positional_encoding(mask).to(decoder_input.dtype) 
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(0, 2, 1) 
            
            # store
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)       
        
        
        # initialize query features and query embeddings
        query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))    
        query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))             
        
        
        # Mask2Former Transformer decoder :start decodeing 
        cls_pred_list = []          
        mask_pred_list = []         
        mask_pred_plus_list = []    
        attn_mask = None            

        # initial decoder layer, initial predictions
        # use the initial query features (query_feat) and the highest resolution mask features(mask_features) to generate initial predictions, 
        # providing a starting point for subsequent decoder layers and allowing information interaction between query features and input features from the beginning.
        cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(
                                                        query_feat, mask_features, multi_scale_memorys[0].shape[-2:], 
                                                        image_embeddings, image_positional_embeddings) 
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        mask_pred_plus_list.append(mask_pred_plus)
        

        # multiple transformer decoder :use dert train query_feat every level
        for i in range(self.num_transformer_decoder_layers):    # 6 layers

            level_idx = i % self.num_transformer_feat_level     
            if attn_mask is not None:
                mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1) # 
                attn_mask = attn_mask & mask_sum

            # cross_attn + self_attn  
            layer = self.transformer_decoder.layers[i]

            # deformable dert decoder train query_feat 
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],       
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,          
                query_key_padding_mask=None,
                key_padding_mask=None)      
            
            # again interact with mask features through sam decoder to get updated predictions
            cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(
                                                            query_feat, mask_features, 
                                                            multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                                                            image_embeddings, image_positional_embeddings) 

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            mask_pred_plus_list.append(mask_pred_plus)
        return cls_pred_list, mask_pred_list, mask_pred_plus_list

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        image_embeddings=None,
        image_positional_embeddings=None
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg) 
            else:
                batch_gt_semantic_segs.append(None)

        # forward  7 layers of transformer decoder
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(x, batch_data_samples, image_embeddings, image_positional_embeddings)
        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)     
        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_mask_preds_plus,
                                   batch_gt_instances, batch_img_metas)
        return losses

    def loss_by_feat(self,
                     all_cls_scores: Tensor,
                     all_mask_preds: Tensor,
                     all_mask_preds_plus,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        num_dec_layers = len(all_cls_scores)    # 7
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_mask_plus, losses_dice_plus = multi_apply(
            self._loss_by_feat_single,
            all_cls_scores, all_mask_preds,
            all_mask_preds_plus,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_mask_plus'] = losses_mask_plus[-1]
        loss_dict['loss_dice_plus'] = losses_dice_plus[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_mask_plus_i, loss_dice_plus_i in zip(
            losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_mask_plus[:-1], losses_dice_plus[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_mask_plus'] = loss_mask_plus_i
            loss_dict[f'd{num_dec_layer}.loss_dice_plus'] = loss_dice_plus_i

            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self,
                             cls_scores: Tensor,
                             mask_preds: Tensor,
                             mask_preds_plus,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)  
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        mask_preds_plus_list = [mask_preds_plus[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_plus_list,      
                                        batch_gt_instances, batch_img_metas)        

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)  
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)      
        labels = labels.flatten(0, 1)               
        label_weights = label_weights.flatten(0, 1) 

        class_weight = cls_scores.new_tensor(self.class_weight)   
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))  # 5
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]      
        mask_preds_plus = mask_preds_plus[mask_weights > 0] 

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_dice_plus = mask_preds_plus.sum()
            loss_mask_plus = mask_preds_plus.sum()
            return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness( # random sample num_points uncertain points from the predicted masks
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)      
            # points_coords = points_coords.to(mask_preds.dtype)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords).squeeze(1)   #  torch.Size([5, 12544])
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1) 
        mask_point_preds_plus = point_sample(
            mask_preds_plus.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
        loss_dice_plus = self.loss_dice(
            mask_point_preds_plus, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        mask_point_preds = mask_point_preds.reshape(-1)   
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_preds_plus = mask_point_preds_plus.reshape(-1)


        # loss_mask
        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets)
        loss_mask_plus = self.loss_mask(mask_point_preds_plus, mask_point_targets)
        return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                image_embeddings=None,
                image_positional_embeddings=None
                ) -> Tuple[Tensor]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(
                                        x, batch_data_samples, image_embeddings=image_embeddings,
                                        image_positional_embeddings=image_positional_embeddings)    
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]            
        mask_pred_plus_results = all_mask_preds_plus[-1]
        # upsample masks
        try:
            img_shape = batch_img_metas[0]['batch_input_shape']
        except:
            img_shape = batch_img_metas[0]['pad_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)       

        return mask_cls_results, mask_pred_results


@MODELS.register_module()
class RSMaskFormerFusionHead(MaskFormerFusionHead):
    def predict(self,
                mask_cls_results: Tensor,
                mask_pred_results: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> List[dict]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        instance_on_large = self.test_cfg.get('instance_on_large', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'
        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, batch_img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]           
            ori_img_height, ori_img_width = meta['ori_shape'][:2]   
            scale_factor = meta['scale_factor']
            ori_scaled_height = int(ori_img_height * scale_factor[1])
            ori_scaled_width = int(ori_img_width * scale_factor[0])  
            mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]  

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]   
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],      
                    size=(ori_height, ori_width),  
                    mode='bilinear',
                    align_corners=False)[:, 0]     

            result = dict()
            if panoptic_on:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results

            if instance_on: # True
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)  
                result['ins_results'] = ins_results     # ins_results comtains bboxes/labels/masks/metainfo/scores

            if instance_on_large: # True
                ins_results = self.instance_postprocess_large(
                    mask_cls_result, mask_pred_result)  
                result['ins_results'] = ins_results   

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            results.append(result)

        return results


@MODELS.register_module()
class RSSamModel(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name)
        if extra_config is not None:
            sam_config.update(extra_config)
        self.sam_model = SamModel(sam_config)

        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.sam_model, init_cfg.get('checkpoint'))
            self.sam_model.is_init = True

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        return self.sam_model(*args, **kwargs)


@MODELS.register_module()
class RSSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class RSSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            peft_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = SamVisionEncoder(sam_config)
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class MMPretrainSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            img_size=1024,
            peft_config=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder_cfg = dict(
            type='mmpretrain.ViTSAM',
            arch=hf_pretrain_name.split('-')[-1].split('_')[-1],    # base
            img_size=img_size,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
        )
        vision_encoder = MODELS.build(vision_encoder_cfg) 
        # load checkpoint only encoder  
        if init_cfg is not None:   
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[
                    (r'^module\.', ''),
                    (r'^vision_encoder\.', ''),
                    (r'.layer_norm1.', '.ln1.'),
                    (r'.layer_norm2.', '.ln2.'),
                    (r'.mlp.lin1.', '.ffn.layers.0.0.'),
                    (r'.mlp.lin2.', '.ffn.layers.1.'),
                    (r'neck.conv1.', 'channel_reduction.0.'),
                    (r'neck.ln1.', 'channel_reduction.1.'),
                    (r'neck.conv2.', 'channel_reduction.2.'),
                    (r'neck.ln2.', 'channel_reduction.3.'),
                ]
            )

        if peft_config is not None and isinstance(peft_config, dict):   
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            
            if is_main_process():       
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True    

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)


@MODELS.register_module()
class RSFPN(BaseModule):
    def __init__(
            self,
            feature_aggregator=None,
            feature_spliter=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        if hasattr(self, 'feature_aggregator'):
            x = self.feature_aggregator(inputs)   
        else:
            x = inputs
        if hasattr(self, 'feature_spliter'):    
            x = self.feature_spliter(x)      
        else:
            x = (x,)
        return x


@MODELS.register_module()
class PseudoFeatureAggregator(BaseModule):
    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.channel_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels, eps=1e-6),
        )

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = self.channel_fusion(x)
        return x
    
    
@MODELS.register_module()
class RSFeatureAggregator(BaseModule):
    in_channels_dict = {
        'base': [768] * (12+1),
        'large': [1024] * (24+1),
        'huge': [1280] * (32+1),
    }

    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            select_layers=range(1, 12, 2),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, str)
        model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
        self.in_channels = self.in_channels_dict[model_arch]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x


@MODELS.register_module()
class SAMDet(BaseDetector):
    def __init__(
            self,
            detector,
            segmentor,
            data_preprocessor=None,
            test_cfg=None,
            init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.detector = MODELS.build(detector)
        self.segmentor = MODELS.build(segmentor)
        self.segmentor.eval()
        self.test_cfg = test_cfg
        for param in self.segmentor.parameters():
            param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor):
        pass

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        losses = self.detector.loss(batch_inputs, batch_data_samples)
        return losses

    def oracle_predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True):

        batch_data_samples = self.detector.predict(batch_inputs, batch_data_samples, rescale=rescale)
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        for input_img, data_sample, meta in zip(batch_inputs, batch_data_samples, batch_img_metas):
            pred_instance_data = InstanceData()
            pred_instance_data.bboxes = data_sample.gt_instances.bboxes
            pred_instance_data.labels = data_sample.gt_instances.labels
            pred_instance_data.scores = torch.ones_like(data_sample.gt_instances.labels, dtype=torch.float32, device=data_sample.gt_instances.labels.device)

            bboxes = pred_instance_data.bboxes
            ori_img_shape = data_sample.ori_shape
            if len(bboxes) == 0:
                mask_pred_binary = torch.zeros(
                    0,
                    ori_img_shape[0],
                    ori_img_shape[1],
                    device=batch_inputs.device,
                    dtype=torch.bool)
            else:
                scale_factor = data_sample.scale_factor
                repeat_num = bboxes.size(-1) // 2
                scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
                bboxes = bboxes * scale_factor

                input_img = input_img.unsqueeze(0)
                bboxes = bboxes.unsqueeze(0)
                outputs = self.segmentor(
                    pixel_values=input_img,
                    input_boxes=bboxes,
                    multimask_output=False,
                )
                mask_pred_result = outputs.pred_masks
                mask_pred_result = mask_pred_result[0]
                mask_pred_result = mask_pred_result.squeeze(1)

                ori_img_height, ori_img_width = meta['ori_shape'][:2]
                scale_factor = meta['scale_factor']
                ori_scaled_height = int(ori_img_height * scale_factor[1])
                ori_scaled_width = int(ori_img_width * scale_factor[0])

                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=meta['img_shape'],
                    mode='bilinear',
                    align_corners=False)[:, 0]

                mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]
                mask_pred_binary = (mask_pred_result > 0)
            pred_instance_data.masks = mask_pred_binary
            data_sample.pred_instances = pred_instance_data
        return batch_data_samples

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True):
        if self.test_cfg is not None and self.test_cfg.get('oracle_on', True):
            return self.oracle_predict(batch_inputs, batch_data_samples, rescale=rescale)

        batch_data_samples = self.detector.predict(batch_inputs, batch_data_samples, rescale=rescale)
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]

        for input_img, data_sample, meta in zip(batch_inputs, batch_data_samples, batch_img_metas):
            bboxes = data_sample.pred_instances.bboxes
            ori_img_shape = data_sample.ori_shape
            if len(bboxes) == 0:
                mask_pred_binary = torch.zeros(
                    0,
                    ori_img_shape[0],
                    ori_img_shape[1],
                    device=batch_inputs.device,
                    dtype=torch.bool)
            else:
                scale_factor = data_sample.scale_factor
                repeat_num = bboxes.size(-1) // 2
                scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
                bboxes = bboxes * scale_factor

                input_img = input_img.unsqueeze(0)
                bboxes = bboxes.unsqueeze(0)
                outputs = self.segmentor(
                    pixel_values=input_img,
                    input_boxes=bboxes,
                    multimask_output=False,
                )
                mask_pred_result = outputs.pred_masks
                mask_pred_result = mask_pred_result[0]
                mask_pred_result = mask_pred_result.squeeze(1)

                ori_img_height, ori_img_width = meta['ori_shape'][:2]
                scale_factor = meta['scale_factor']
                ori_scaled_height = int(ori_img_height * scale_factor[1])
                ori_scaled_width = int(ori_img_width * scale_factor[0])

                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=meta['img_shape'],
                    mode='bilinear',
                    align_corners=False)[:, 0]

                mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]
                mask_pred_binary = (mask_pred_result > 0)
            data_sample.pred_instances.masks = mask_pred_binary

        return batch_data_samples


@MODELS.register_module()
class SAMSegMaskRCNN(MaskRCNN):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)

        if peft_config is None:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs.last_hidden_state
            vision_hidden_states = vision_outputs.hidden_states
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError
        x = self.neck(vision_hidden_states)
        return x


@MODELS.register_module()
class SAMSegMask2Former(Mask2Former):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)

        if peft_config is None:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs.last_hidden_state
            vision_hidden_states = vision_outputs.hidden_states
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        x = self.neck(vision_hidden_states)
        return x


@MODELS.register_module()
class RSSimpleFPN(BaseModule):
    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel       
        self.in_channels = in_channels                
        self.out_channels = out_channels
        self.num_ins = len(in_channels)               
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple: 
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))    
        inputs.append(self.fpn2(input))     
        inputs.append(self.fpn3(input))   
        inputs.append(self.fpn4(input))   

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]  

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)

