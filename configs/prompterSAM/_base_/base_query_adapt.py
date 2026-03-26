# work directory
work_dir = './work_dirs/prompterSAM/prompterSAM_degub'  
default_scope = 'mmdet' 
custom_imports = dict(imports=['mmdet.promptersam'], allow_failed_imports=False) 


# defalut hooks 
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),                    # save_best='coco/segm_mAP    'coco/bbox_mAP'
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best='coco/bbox_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'), 
)

# environment 
env_cfg = dict(
    cudnn_benchmark=False, 
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # multi-process config
    dist_cfg=dict(backend='nccl'))  # distributed config


# training status
stage = 'stage_train'
prototype_path = '/home/dyl/PromptSAM/domain_adapt/prototype/prototypes_on_0218/prototype.pth'


# visualization 
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer_adapt', vis_backends=vis_backends, name='visualizer') 


# logging 
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'


# load and resume 
load_from = None
resume = False


# model config
num_classes = 1
prompt_shape = (100, 5)  # (query count, point per query)
hf_sam_pretrain_name = "work_dirs/sam_cache/sam_vit_base"
hf_sam_pretrain_ckpt_path = "work_dirs/sam_cache/sam_vit_base/pytorch_model.bin"



# image crop and data preprocessing 
crop_size = (1024, 1024)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,  
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)



# model details
model = dict(
    type='RSPrompterQuery_adapt',
    stage=stage,
    prototype_path=prototype_path,
    data_preprocessor=data_preprocessor,
    decoder_freeze=False,   
    shared_image_embedding=dict(
        type='RSSamPositionalEmbedding',  
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),  
    ),
    backbone=dict(
        type='RSSamVisionEncoder',
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(output_hidden_states=True),
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
    ),
    neck=dict(
        type='RSFPN',
        feature_aggregator=dict(  
            type='RSFeatureAggregator',
            in_channels=hf_sam_pretrain_name,
            out_channels=256,
            hidden_channels=32,
            select_layers=range(1, 32+1, 2),
        ),
        feature_spliter=dict(  
            type='RSSimpleFPN',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)),  
    ),
    panoptic_head=dict(
        type='RSMask2FormerHead_adapt',   
        decoder_plus=True, 
        mask_decoder=dict(
            type='RSSamMaskDecoder',  
            hf_pretrain_name=hf_sam_pretrain_name,
            init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
        per_pointset_point=prompt_shape[1],  
        with_sincos=True,      
        multimask_output=False, 
        in_channels=[256, 256, 256, 256, 256],  
        feat_channels=128,
        out_channels=256,
        num_things_classes=num_classes, 
        num_stuff_classes=0,
        num_queries=prompt_shape[0],   
        num_transformer_feat_level=3,  
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            strides=[4, 8, 16, 32, 64],
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  
                num_layers=3,
                layer_cfg=dict( 
                    self_attn_cfg=dict(  
                        embed_dims=128,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=128,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=64, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=64, normalize=True),
        transformer_decoder=dict(  
            return_intermediate=True,
            num_layers=6,
            layer_cfg=dict(  
                self_attn_cfg=dict( 
                    embed_dims=128,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict( 
                    embed_dims=128,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=128,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_cls_target=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask_target=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            ignore_index=250,
            loss_weight=2.0),
        loss_dice_target=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            ignore_index=250,
            naive_dice=True,
            eps=1.0,
            loss_weight=2.0),
        loss_rce_target=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            ignore_index=250,
            loss_weight=0.1),
        loss_regular_target=dict(
            type='RegularLoss',
            regular_type='MRKLD',
            loss_weight=0.1),
        loss_consist_target=dict(
            type='KnowledgeDistillationKLDivLoss',
            reduction='none',
            ignore_index=250,
            T=1.0,
            loss_weight=1.5),
            ),
    panoptic_fusion_head=dict(
        type='RSMaskFormerFusionHead_adapt',
        num_things_classes=num_classes,
        num_stuff_classes=0,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544, 
        oversample_ratio=3.0,
        importance_sample_ratio=0.75, 
        assigner=dict(
            type='HungarianAssigner',   
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=prompt_shape[0],
        iou_thr=0.8,
        filter_low_score=True)
)


# dataset 
code_root = '/home/dyl/PromptSAM/'
source_dataset_type = 'WuhanUISDataset_2022'
source_data_root = '/home/dyl/PromptSAM/data/wuhanUIS_2022'
target_dataset_type = 'WuhanUISDataset_0519'
target_data_root = '/home/dyl/PromptSAM/data/wuhanUIS_0519'



# training data pipelines
backend_args = None
train_source_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', 
         scale=crop_size,
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 1.5),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

train_weak_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', 
         scale=crop_size,
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 1.5),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

train_strong_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', 
         scale=crop_size,
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 1.5),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='ColorJitter', 
         prob=0.5, 
         n=1, 
         m=10),
    dict(
        type='CutoutAbs', 
        divisor=4),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]


# test data pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor'))
]



# dataloader
batch_size = 2
num_workers = 8
persistent_workers = True
indices = None


source_train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=source_dataset_type,       # 'WuhanUISDataset_2022'
        indices=indices,
        data_root=source_data_root,    
        ann_file=code_root + '/data/wuhanUIS_2022/annotations/wuhanUIS_2022_train.json',  
        data_prefix=dict(img='imgs/train/image'), 
        pipeline=train_source_pipeline, 
        backend_args=backend_args))

# use soft label
target_train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=target_dataset_type,       # 'WuhanUISDataset_0519'
        indices=indices,
        data_root=target_data_root,    
        ann_file=code_root + '/data/wuhanUIS_0519/annotations_0218/wuhanUIS_0519_soft_train.json',    
        data_prefix=dict(img='imgs/train/image'),
        
        # pipeline=train_weak_pipeline,  # can be replaced with weak augmentation.
        pipeline=train_strong_pipeline, 
        backend_args=backend_args))


train_dataloader= {
    'source': source_train_dataloader,
    'target': target_train_dataloader,
    'batch_size': batch_size,
}

# target_val
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=target_dataset_type,
        indices=indices,
        data_root=target_data_root,
        ann_file=code_root + '/data/wuhanUIS_0519/annotations/wuhanUIS_0519_val.json',
        data_prefix=dict(img='imgs/val/image'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader= dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=target_dataset_type,
        indices=indices,
        data_root=target_data_root,
        ann_file=code_root + '/data/wuhanUIS_0519/annotations/wuhanUIS_0519_test.json',
        data_prefix=dict(img='imgs/test/image'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)



# evaluator
val_evaluator = dict(
    type='CocoMetric',  
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator


# training config
max_epochs = 100
train_cfg = dict(type='EpochBasedTrainLoop_adapt', max_epochs=max_epochs, val_interval=3) 
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 0.0001
find_unused_parameters = True  

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
]
auto_scale_lr = dict(enable=False, base_batch_size=batch_size)