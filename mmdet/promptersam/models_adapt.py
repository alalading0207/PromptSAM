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
from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding, Mask2Former_adapt, Mask2FormerHead, \
    MaskFormerFusionHead, BaseDetector, Mask2FormerHead_adapt
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances, multi_apply, \
    get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample, OptSampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptConfigType, OptMultiConfig, MultiConfig, ConfigType, InstanceList, reduce_mean
import torch.nn.functional as F
from mmdet.structures.mask import mask2bbox

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
class RSPrompterQuery_adapt(Mask2Former_adapt):
    def __init__(
            self,
            shared_image_embedding,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})   # get peft_config 
        self.stage = kwargs.get('stage', None)                            # get stage 
        self.prototype_path = kwargs.pop('prototype_path', None)          # prototype_path
        # call the constructor of the parent class Mask2Former
        super().__init__(*args, **kwargs)   
        self.decoder_freeze = decoder_freeze    # false
        self.with_mask2formerhead = False if isinstance(self.panoptic_head, RSMask2FormerHead_adapt) else True  # false
        self.shared_image_embedding = MODELS.build(shared_image_embedding)      # used for generation and sharing of image embeddings
        
        self.state = 'source'


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

        # initialize objective_vectors for prototype computation
        if self.stage == 'compute_prototype':
            self.class_numbers = 2   # uv/non-uv
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.objective_vectors = torch.zeros([self.class_numbers, 256], device=self.device)     

        # only teacher model need self.objective_vectors
        if self.stage == 'stage_train':
            if self.prototype_path is not None:
                self.objective_vectors = torch.load(self.prototype_path)    
            else:
                self.objective_vectors = None


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

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))     # 
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
             batch_data_samples: SampleList
             ) -> Dict[str, Tensor]:
        
        # input teacher_feat \ prototype
        if self.state == 'target':
            teacher_feat = batch_data_samples[0].teacher_feat
            objective_vectors = batch_data_samples[0].objective_vectors
            del batch_data_samples[0].teacher_feat
            del batch_data_samples[0].objective_vectors
        # input teacher_feat \ prototype \ Strong target
        elif self.state == 'targetS':
            teacher_feat = batch_data_samples[0].teacher_feat
            objective_vectors = batch_data_samples[0].objective_vectors
            del batch_data_samples[0].teacher_feat
            del batch_data_samples[0].objective_vectors
            # target
            batch_inputs_S_list = []
            for sample in batch_data_samples:
                if not hasattr(sample, 'img_aug'):
                    raise ValueError("Some batch_data_samples are missing 'img_aug'!")
                batch_inputs_S_list.append(sample.img_aug)
                del sample.img_aug
            batch_inputs_S = torch.stack(batch_inputs_S_list)      
                
        else:
            teacher_feat=None

       
        # # x muiti-scale features / image_embeddings  / image_positional_embeddings
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)  
        

        # source mode
        if self.state == 'source':
            losses, mask_features = self.panoptic_head.loss(x, batch_data_samples,  
                                                image_embeddings=image_embeddings, 
                                                image_positional_embeddings=image_positional_embeddings,
                                                state=self.state)
            return losses
        
        # target mode
        elif self.state == 'target':
            losses, mask_features = self.panoptic_head.loss(x, batch_data_samples,   
                                                image_embeddings=image_embeddings, 
                                                image_positional_embeddings=image_positional_embeddings,
                                                teacher_feat=teacher_feat,
                                                objective_vectors = objective_vectors, # initial prototype
                                                state=self.state)
            return losses

        # target mode
        elif self.state == 'targetS':
            x_S, image_embeddings_S, image_positional_embeddings_S = self.extract_feat(batch_inputs_S)  
            losses, mask_features = self.panoptic_head.loss(x, batch_data_samples,   
                                                image_embeddings=image_embeddings, 
                                                image_positional_embeddings=image_positional_embeddings,
                                                teacher_feat=teacher_feat,
                                                objective_vectors = objective_vectors, # initial prototype
                                                state=self.state,
                                                x_S=x_S,
                                                image_embeddings_S=image_embeddings_S, 
                                                image_positional_embeddings_S=image_positional_embeddings_S,
                                                )
            return losses

        # teachers mode
        elif self.state == 'teacher':
            prototype, mask_features = self.panoptic_head.loss(x, batch_data_samples,  
                                                image_embeddings=image_embeddings, 
                                                image_positional_embeddings=image_positional_embeddings,
                                                objective_vectors = self.objective_vectors, # source prototype
                                                state=self.state)
            return (prototype, mask_features)
        
        else:
            raise NotImplementedError

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        
        # x muiti-scale features / image_embeddings  / image_positional_embeddings
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)  


        # to RSMask2FormerHead_adapt
        if self.with_mask2formerhead:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(x, batch_data_samples)  
        else:
            if self.stage == 'compute_prototype':
                self.objective_vectors = self.panoptic_head.predict(
                    x, batch_data_samples,
                    image_embeddings=image_embeddings,
                    image_positional_embeddings=image_positional_embeddings
                    )
                return self.objective_vectors
            else:   # normal prediction
                mask_features, mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                    x, batch_data_samples,
                    image_embeddings=image_embeddings,
                    image_positional_embeddings=image_positional_embeddings,
                    stage=self.stage
                    )

        # generate and predct need normal prediction process 
        if self.stage == 'get_softlabel' or self.stage == 'stage_predict':    
            # to RSMaskFormerFusionHead_adapt   
            # results_list[]'ins_result'] contains bbpxes/labels/masks/metainfo/scores
            results_list = self.panoptic_fusion_head.predict(
                mask_cls_results,
                mask_pred_results,
                batch_data_samples,
                rescale=rescale)   
            results = self.add_pred_to_datasample(batch_data_samples, results_list)    

            return results
    


@MODELS.register_module()
class RSMask2FormerHead_adapt(Mask2FormerHead_adapt, BaseModule):
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
        stage = kwargs.pop('stage', None)
        super().__init__(*args, **kwargs)

        self.stage = stage
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
        
        self.proto_temperature = 1.0
        self.objective_vectors = None
        self.class_numbers = 2 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.objective_vectors_num = torch.zeros([self.class_numbers], device=self.device)  
        self.last_objective_vectors = None


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
            level_embed = self.level_embed.weight[i].view(1, 1, -1)     # # learnable level embedding for each multi-scale feature
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
                mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1) 
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
        return mask_features, cls_pred_list, mask_pred_list, mask_pred_plus_list

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        image_embeddings=None,
        image_positional_embeddings=None,
        teacher_feat=None,
        objective_vectors = None,
        state=None,
        x_S=None,    
        image_embeddings_S=None, 
        image_positional_embeddings_S=None,
    ) -> Dict[str, Tensor]:

        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)        # image augmentation parameters
            batch_gt_instances.append(data_sample.gt_instances) # image ground truth parameters
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg) 
            else:
                batch_gt_semantic_segs.append(None)
        
        # preprocess ground truth
        if state == 'target' or state == 'targetS':
            batch_gt_instances = self.preprocess_softgt(batch_gt_instances, batch_gt_semantic_segs)   
        elif state == 'source':
            batch_gt_instances = self.preprocess_gt(batch_gt_instances, batch_gt_semantic_segs)
        else:   # tearcher mode doesn't need gt
            pass


        # forward  7 layers of transformer decoder
        mask_features, all_cls_scores, all_mask_preds, all_mask_preds_plus = self(x, batch_data_samples, image_embeddings, image_positional_embeddings)
        
        # target mode
        # target: prototype rectify pesudo-label
        if state == 'target':
            for i in range(len(batch_gt_instances)):
                # calculate the weight for prototype rectification
                weight_,_ = self.get_prototype_weight(teacher_feat[i:i+1], objective_vectors, batch_img_metas[i]) 
                weight = F.interpolate(weight_, scale_factor=4.0, mode='bilinear', align_corners=True).flip(1) 
                # get pseudo-label
                masks = batch_gt_instances[i]['masks']  
                if masks.shape[0] == 0 or (masks.sum(dim=(-1, -2))==0).all():
                    continue
                # introduce background class
                masks_bg = 1 - masks
                masks_all =  torch.stack([masks_bg, masks], dim=1)  # [num_instances, 2, H, W]
                # rectify pseudo-label
                rectified_soft = masks_all * weight                    # torch.Size([num_instances, 2, H, W])
                # binary pseudo-label
                rectified_label = rectified_soft.max(1,  keepdim=True)[1]  # torch.Size([num_instances, 1, H, W])
                # normalize->confidence
                rectified_norm = rectified_soft / (rectified_soft.sum(1, keepdim=True) + 1e-6)  # torch.Size([num_instances, 2, H, W])
                # confidence value
                rectified_conf = rectified_norm.max(1,  keepdim=True)[0]    # torch.Size([num_instances, 1, H, W]) 
                # remove low-confidence pseudo-label
                rectified_label[rectified_conf < 0.5] = 250     # dice loss need to set low-confidence area to ignore index 250
                # updated pseudo-label
                batch_gt_instances[i]['masks'] = rectified_label.squeeze(1)
             
            # loss
            losses = self.loss_by_feat_adapt(all_cls_scores, all_mask_preds, all_mask_preds_plus,
                                        batch_gt_instances, batch_img_metas) 
            return losses, mask_features
            
            
        # Strong mode
        elif state == 'targetS':
            # strong augmentation data
            mask_features_S, _, _, _ = self(x_S, batch_data_samples, image_embeddings_S, image_positional_embeddings_S) #
            teacher_distances = []
            student_distances = []

            for i in range(len(batch_gt_instances)):
                weight_, teacher_distance= self.get_prototype_weight(teacher_feat[i:i+1], objective_vectors, batch_img_metas[i]) 
                weight = F.interpolate(weight_, scale_factor=4.0, mode='bilinear', align_corners=True).flip(1)  
                masks = batch_gt_instances[i]['masks'] 
                if masks.shape[0] == 0 or (masks.sum(dim=(-1, -2))==0).all():
                    pass
                else:
                    
                    masks_bg = 1 - masks
                    masks_all =  torch.stack([masks_bg, masks], dim=1)  #

                    rectified_soft = masks_all * weight                   
                    rectified_label = rectified_soft.max(1,  keepdim=True)[1]  
                    rectified_norm = rectified_soft / (rectified_soft.sum(1, keepdim=True) + 1e-6)  
                    rectified_conf = rectified_norm.max(1,  keepdim=True)[0]   
                    rectified_label[rectified_conf < 0.5] = 250  

                    batch_gt_instances[i]['masks'] = rectified_label.squeeze(1)


                # prototype distance consistency loss: distance between teacher feature and prototype, distance between student feature and prototype
                teacher_strong_distance = self.distance2strong(teacher_distance, batch_img_metas[i])        
                strong_distance = self.feat_prototype_distance(mask_features_S[i:i+1], objective_vectors)   
                teacher_distances.append(teacher_strong_distance)
                student_distances.append(strong_distance)

            teacher_distances = torch.cat(teacher_distances, dim=0)
            student_distances = torch.cat(student_distances, dim=0)


            # loss
            losses = self.loss_by_feat_adapt(all_cls_scores, all_mask_preds, all_mask_preds_plus,
                                        batch_gt_instances, batch_img_metas,
                                        teacher=-teacher_distances, student=-student_distances) 
            return losses, mask_features
        

        # teacher mode: real-time update of prototype vector
        elif state == 'teacher':

            if self.objective_vectors is None:
                self.objective_vectors = objective_vectors.to(self.device)
            
            # calculate the prototype for the current batch 
            current_prototypes = self.compute_prototype(mask_features, all_cls_scores[-1], all_mask_preds[-1]) # this batch's prototypes
            
            # fisrt step only store current_prototypes without updating global prototype self.objective_vectors
            if self.last_objective_vectors is None:
                self.last_objective_vectors = current_prototypes
                return self.objective_vectors, mask_features

            # other step: use last_objective_vectors update global prototype vector 
            for batch_id, batch_prototypes in enumerate(self.last_objective_vectors): 
                for class_id, prototype in enumerate(batch_prototypes):
                    
                    last_proto = self.last_objective_vectors[batch_id][class_id]    #  last_proto = prototype
                    if last_proto.sum().item() == 0:
                        continue
                    self.update_objective_SingleVector(class_id, last_proto)

            self.last_objective_vectors = current_prototypes

            return self.objective_vectors, mask_features
        
        # source mode
        elif state == 'source':
            losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_mask_preds_plus,
                                    batch_gt_instances, batch_img_metas)  # loss_cls/loss_mask/loss_dice/loss_mask_plus/loss_dice_plus
            return losses, mask_features
        
        else:
            warnings.warn('state is wrong', UserWarning)
            
    

    def compute_prototype(self, feat, mask_cls, mask_pred): 

        # transfer mask_cls class probability distribution, exclude background class
        outputs_softmax = F.softmax(mask_cls, dim=-1)  
       
        # calculate prototype for each category in the current batch
        prototypes = [] 
        for n in range(feat.size(0)):              
            class_prototypes = []                  
            for c in range(self.class_numbers):    
                

                # combine each query classprobability and spatial mask prediction 
                soft_mask_c = mask_pred[n].sigmoid() * outputs_softmax[n, :, c].unsqueeze(-1).unsqueeze(-1)  
                # all background pixels 
                if soft_mask_c.max() < 1e-3:
                    class_prototypes.append(torch.zeros(feat.shape[1])) 
                    continue
                mask = soft_mask_c.max(dim=0)[0]       

                # calculate the weighted average of the features
                weighted_feat = feat[n] * mask                  
                prototype = weighted_feat.sum(dim=(1, 2)) / (mask.sum() + 1e-6) 
                class_prototypes.append(prototype)
            
            prototypes.append(class_prototypes)
        return prototypes
    

    # update single category prototype vector: current category cumulative vector = (old vector * number of updates) + new vector
    def update_objective_SingleVector(self, class_id, last_proto): 
        
        # adaptation mode
        self.objective_vectors[class_id] = self.objective_vectors[class_id] * (1 - 0.001) + 0.001 * last_proto
        self.objective_vectors_num[class_id] += 1
        self.objective_vectors_num[class_id] = min(self.objective_vectors_num[class_id], 3000)
    

        # #  prototype generation mode
        # # accumulation
        # self.objective_vectors[class_id] = (self.objective_vectors[class_id] * self.objective_vectors_num[class_id] + last_proto)
        # self.objective_vectors_num[class_id] += 1
        # self.objective_vectors[class_id] = self.objective_vectors[class_id] / self.objective_vectors_num[class_id]
        # self.objective_vectors_num[class_id] = min(self.objective_vectors_num[class_id], 3000)


    def get_prototype_weight(self, teacher_feat, objective_vectors, target_weak_params=None): 

        # full_image features align to the scale of weakly augmented image, where fear is teacher_feat
        if target_weak_params is not None:
            feat = self.full2weak(teacher_feat, target_weak_params)    

        feat_proto_distance = self.feat_prototype_distance(feat, objective_vectors)

        # relative distance
        nearest_feat_proto_distance, _ = feat_proto_distance.min(dim=1, keepdim=True)        
        centered_feat_proto_distance = feat_proto_distance - nearest_feat_proto_distance     # centered_feat_proto_distance 

        weight = F.softmax(-centered_feat_proto_distance * self.proto_temperature, dim=1)
        
        return weight, feat_proto_distance

    def feat_prototype_distance(self, feat, objective_vectors):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, self.class_numbers, H, W)).to(self.device)     
        # calculate the distance from each pixel in the feature map to each category prototype
        for i in range(self.class_numbers):
            proto_vector = objective_vectors[i].reshape(-1, 1, 1).expand(-1, H, W)  
            feat_proto_distance[:, i, :, :] = torch.norm(proto_vector - feat, p=2, dim=1, keepdim=True)  

        return feat_proto_distance


    def full2weak(self, teacher_feat, target_weak_param=None):

        # random scale
        target_h, target_w = target_weak_param['scale']
        teacher_feat_ = F.interpolate(teacher_feat, size=[target_h // 4, target_w // 4], mode='bilinear', align_corners=True)
        
        # if smaller, pad
        pad_h = max(target_weak_param['batch_input_shape'][0] // 4 - teacher_feat_.size(2), 0)
        pad_w = max(target_weak_param['batch_input_shape'][1] // 4 - teacher_feat_.size(3), 0)
        if pad_h > 0 or pad_w > 0:
            teacher_feat_ = F.pad(teacher_feat_, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # crop feature map
        crop_y1, crop_y2 = target_weak_param['crop_place']['crop_y1'], target_weak_param['crop_place']['crop_y2']
        crop_x1, crop_x2 = target_weak_param['crop_place']['crop_x1'], target_weak_param['crop_place']['crop_x2']
        crop_y1, crop_h = crop_y1 // 4, (crop_y2 - crop_y1) // 4
        crop_x1, crop_w = crop_x1 // 4, (crop_x2 - crop_x1) // 4
        if target_h < target_weak_param['batch_input_shape'][0]:
            crop_h = target_weak_param['batch_input_shape'][0] // 4
            crop_w = target_weak_param['batch_input_shape'][0] // 4
        teacher_feat_ = teacher_feat_[:, :, crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
        
        # horizontal flip
        if target_weak_param['flip']:
            inv_idx = torch.arange(teacher_feat_.size(3)-1,-1,-1).long().to(teacher_feat_.device)
            teacher_feat_ = teacher_feat_.index_select(3,inv_idx)

        return teacher_feat_
    
    def distance2strong(self, teacher_distance, target_params, padding=-250, scale=4):     
        if teacher_distance.dim() == 3:
            teacher_distance = teacher_distance.unsqueeze(0)
        teacher_distance = teacher_distance + 1       
        if 'cutout' in target_params:
            x0, y0, x1, y1 = (int(target_params['cutout'][i] // scale) for i in range(4))
            teacher_distance[:, :, y0:y1, x0:x1] = 0   
        teacher_distance[teacher_distance == 0] = padding + 1  # for strong augmentation, constant padding 
        teacher_distance = teacher_distance - 1
        return teacher_distance    


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
    
    def loss_by_feat_adapt(self,
                     all_cls_scores: Tensor,
                     all_mask_preds: Tensor,
                     all_mask_preds_plus,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict],
                     teacher: Optional[Tensor] = None,
                     student: Optional[Tensor] = None) -> Dict[str, Tensor]:
    
        num_dec_layers = len(all_cls_scores)   
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_mask_plus, losses_dice_plus,\
            losses_rce, losses_regular, losses_rce_plus, losses_regular_plus = multi_apply(
            self._loss_by_feat_single_adapt,
            all_cls_scores, all_mask_preds,
            all_mask_preds_plus,
            batch_gt_instances_list, img_metas_list) #

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_rce'] = losses_rce[-1]
        loss_dict['loss_regular'] = losses_regular[-1]
        loss_dict['loss_mask_plus'] = losses_mask_plus[-1]
        loss_dict['loss_dice_plus'] = losses_dice_plus[-1]
        loss_dict['loss_rce_plus'] = losses_rce_plus[-1]
        loss_dict['loss_regular_plus'] = losses_regular_plus[-1]


        # prototype distance consistency loss between teacher and student
        if teacher is not None and student is not None:
            loss_consist = self.loss_consist_target(student, teacher)
            loss_dict['loss_consist'] = loss_consist

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_mask_plus_i, loss_dice_plus_i, \
            loss_rce_i, loss_regular_i, loss_rce_plus_i, loss_regular_plus_i in zip(
            losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_mask_plus[:-1], losses_dice_plus[:-1],\
            losses_rce[:-1], losses_regular[:-1], losses_rce_plus[:-1], losses_regular_plus[:-1]):
            
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_rce'] = loss_rce_i
            loss_dict[f'd{num_dec_layer}.loss_regular'] = loss_regular_i
            loss_dict[f'd{num_dec_layer}.loss_mask_plus'] = loss_mask_plus_i
            loss_dict[f'd{num_dec_layer}.loss_dice_plus'] = loss_dice_plus_i
            loss_dict[f'd{num_dec_layer}.loss_rce_plus'] = loss_rce_plus_i
            loss_dict[f'd{num_dec_layer}.loss_regular_plus'] = loss_regular_plus_i
            num_dec_layer += 1
        return loss_dict


    def _loss_by_feat_single_adapt(self,
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
        labels = torch.stack(labels_list, dim=0)    # num_queries
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)  
        mask_weights = torch.stack(mask_weights_list, dim=0) 

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)      
        labels = labels.flatten(0, 1)              
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)     
        loss_cls = self.loss_cls_target(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))  
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]      
        mask_preds_plus = mask_preds_plus[mask_weights > 0] 

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_rce = mask_preds.sum()
            loss_regular = mask_preds.sum()
            loss_dice_plus = mask_preds_plus.sum()
            loss_mask_plus = mask_preds_plus.sum()
            loss_rce_plus = mask_preds_plus.sum()
            loss_regular_plus = mask_preds_plus.sum()
            return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus,\
                    loss_rce, loss_regular, loss_rce_plus, loss_regular_plus
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(  # random sample num_points uncertain points from the predicted masks
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)      
            # points_coords = points_coords.to(mask_preds.dtype)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords).squeeze(1)   # torch.Size([5, 12544])
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1) 
        mask_point_preds_plus = point_sample(
            mask_preds_plus.unsqueeze(1), points_coords).squeeze(1)


        # Regular loss 
        loss_regular = self.loss_regular_target(mask_point_preds)
        loss_regular_plus = self.loss_regular_target(mask_point_preds_plus)


        # before computing the loss, add filtering logic
        mask_point_targets[mask_point_targets > 1] = 250.0
        valid_mask = (mask_point_targets != 250)  
        instance_valid = (valid_mask.sum(dim=1) > 0)
        num_total_masks = instance_valid.sum().item()
        if num_total_masks == 0:     
            zero_loss = 0.0 * mask_preds.sum()
            return loss_cls, zero_loss, zero_loss, zero_loss, zero_loss,\
                     zero_loss, loss_regular, zero_loss, loss_regular_plus

        # Dice loss 
        loss_dice = self.loss_dice_target(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)  
        loss_dice_plus = self.loss_dice_target(
            mask_point_preds_plus, mask_point_targets, avg_factor=num_total_masks)


        # Mask loss 
        mask_point_preds = mask_point_preds.reshape(-1) 
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_preds_plus = mask_point_preds_plus.reshape(-1)

        loss_mask = self.loss_mask_target(mask_point_preds, mask_point_targets)     
        loss_mask_plus = self.loss_mask_target(mask_point_preds_plus, mask_point_targets)
        
        # Rce loss
        mask_point_preds_reversed = -mask_point_preds
        mask_point_preds_plus_reversed = -mask_point_preds_plus
        loss_rce = self.loss_rce_target(mask_point_preds_reversed, mask_point_targets)
        loss_rce_plus = self.loss_rce_target(mask_point_preds_plus_reversed, mask_point_targets)


        return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus,\
                loss_rce, loss_regular, loss_rce_plus, loss_regular_plus



    def _loss_by_feat_single(self,
                             cls_scores: Tensor,   
                             mask_preds: Tensor,   
                             mask_preds_plus,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)   # B=1
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)] 
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]  
        mask_preds_plus_list = [mask_preds_plus[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_plus_list,   
                                        batch_gt_instances, batch_img_metas)       

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0) 
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
            points_coords = get_uncertain_point_coords_with_randomness(  # random sample num_points uncertain points from the predicted masks
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)       
            # points_coords = points_coords.to(mask_preds.dtype)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords).squeeze(1)   # torch.Size([5, 12544])
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

        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets)     
        loss_mask_plus = self.loss_mask(mask_point_preds_plus, mask_point_targets)
        
        return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus


    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                image_embeddings=None,
                image_positional_embeddings=None,
                stage=None
                ) -> Tuple[Tensor]:
        
        if stage is not None:
            self.stage = stage
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]       
        
        # to RSMask2FormerHead_adapt forward
        mask_features, all_cls_scores, all_mask_preds, all_mask_preds_plus = self(
                                        x, batch_data_samples, image_embeddings=image_embeddings,
                                        image_positional_embeddings=image_positional_embeddings)   
        mask_cls_results = all_cls_scores[-1]              
        mask_pred_results = all_mask_preds[-1]            
        mask_pred_plus_results = all_mask_preds_plus[-1]
        
        try:
            img_shape = batch_img_metas[0]['batch_input_shape']
        except:
            img_shape = batch_img_metas[0]['pad_shape']


        # upsample masks
        # generate soft label and compute prototype without interpolation
        if self.stage == 'get_softlabel':
            return mask_features, mask_cls_results, mask_pred_results
        
        elif self.stage == 'compute_prototype':  
            if self.objective_vectors is None:
                self.objective_vectors = torch.zeros([self.class_numbers, 256], device=self.device)  
            
            # calculate current batch propotypes
            current_prototypes = self.compute_prototype(mask_features, mask_cls_results, mask_pred_results)
            
            # use current batch prototypes to update global prototypes objective_vectors
            for batch_id, batch_prototypes in enumerate(current_prototypes):     
                for class_id, prototype in enumerate(batch_prototypes):          
                    if prototype.sum().item() == 0:
                        continue
                    self.update_objective_SingleVector(class_id, prototype)
            
            return self.objective_vectors
        
        elif self.stage == 'stage_predict': 
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(img_shape[0], img_shape[1]),
                mode='bilinear',align_corners=False)      

            return mask_features, mask_cls_results, mask_pred_results
        
        else:
            raise ValueError(f'Wrong stage {self.stage} provided')

@MODELS.register_module()
class RSMaskFormerFusionHead_adapt(MaskFormerFusionHead):
    def __init__(self,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 test_cfg: OptConfigType = None,
                 loss_panoptic: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs):
            stage = kwargs.pop('stage', None)
            super().__init__(
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            test_cfg=test_cfg,
            loss_panoptic=loss_panoptic,
            init_cfg=init_cfg,
            **kwargs)
            self.stage = stage
        
    def predict(self,
                mask_cls_results: Tensor,
                mask_pred_results: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> List[dict]:
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        instance_on = self.test_cfg.get('instance_on', False)
        instance_on_large = self.test_cfg.get('instance_on_large', False)

        results = []

        for mask_cls_result, mask_pred_result, meta in zip(mask_cls_results, mask_pred_results, batch_img_metas):   
            # remove padding
            img_height, img_width = meta['img_shape'][:2]          
            ori_img_height, ori_img_width = meta['ori_shape'][:2]  
            scale_factor = meta['scale_factor']
            ori_scaled_height = int(ori_img_height * scale_factor[1]) 
            ori_scaled_width = int(ori_img_width * scale_factor[0])   
            mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]  

            if rescale and self.stage != 'get_softlabel':       
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]  
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],    
                    size=(ori_height, ori_width),  
                    mode='bilinear', align_corners=False)[:, 0]    

            result = dict()
            if instance_on: # True
                ins_results = self.instance_postprocess_adapt(
                    mask_cls_result, mask_pred_result)  
                result['ins_results'] = ins_results     #  # ins_results comtains bboxes/labels/masks/metainfo/scores  

            if instance_on_large: # True
                ins_results = self.instance_postprocess_large(
                    mask_cls_result, mask_pred_result)  
                result['ins_results'] = ins_results 

            results.append(result)

        return results
    
    def instance_postprocess_adapt(self, mask_cls: Tensor,          
                             mask_pred: Tensor) -> InstanceData:    

        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]  

        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]    

        # high score
        labels = torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        # top_indices is the indices of the top max_per_image scores
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)        
        

        # get queries and classes
        labels_per_image = labels[top_indices] 
        query_indices = top_indices // self.num_classes     
        mask_pred = mask_pred[query_indices]    

        # extract things 
        is_thing = labels_per_image < self.num_things_classes    
        scores_per_image = scores_per_image[is_thing]            
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]       

        # binary mask
        mask_pred_binary = (mask_pred > 0).float()     

        if self.stage == 'get_softlabel':
            # get overall score
            mask_scores_per_image = (mask_pred.sigmoid() *           
                                        mask_pred_binary).flatten(1).sum(1) / (
                                            mask_pred_binary.flatten(1).sum(1) + 1e-6)  
            
            det_scores = scores_per_image * mask_scores_per_image  
            soft_mask = mask_pred.sigmoid() * scores_per_image[:, None, None]  


            results = InstanceData()
            results.labels = labels_per_image  
            results.scores = det_scores         
            results.masks = soft_mask         

        else:
            mask_scores_per_image = (mask_pred.sigmoid() *           
                                        mask_pred_binary).flatten(1).sum(1) / (
                                            mask_pred_binary.flatten(1).sum(1) + 1e-6) 
            
            
            det_scores = scores_per_image * mask_scores_per_image   
            mask_pred_binary = mask_pred_binary.bool()
            bboxes = mask2bbox(mask_pred_binary)

            results = InstanceData()
            results.bboxes = bboxes          
            results.labels = labels_per_image       
            results.scores = det_scores             
            results.masks = mask_pred_binary
        return results
