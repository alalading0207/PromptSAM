'''3. Target inference config. Predict on WuhanUIS 0519 using source-trained model.'''


# work directory
_base_ = ['_base_/base_query.py'] 
work_dir = '/home/dyl/PromptSAM/work_dirs/prompterSAM'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),                     # save_best='coco/segm_mAP    'coco/bbox_mAP'
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best='coco/segm_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),  
)



# visualization 
vis_backends = [dict(type='LocalVisBackend'),
                # dict(type='WandbVisBackend', init_kwargs=dict(project='rsprompter-whu', group='rsprompter-query', name='rsprompter-query-whu'))
                ]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')



# load and resume 
resume = False      # True
load_from = None    # ‘/home/dyl/PromptSAM/work_dirs/rsprompter/epoch_300.pth‘  



# model config
num_classes = 1
prompt_shape = (30, 5)  # (query count, point per query)
hf_sam_pretrain_name = "work_dirs/sam_cache/sam_vit_base"
hf_sam_pretrain_ckpt_path = "work_dirs/sam_cache/sam_vit_base/pytorch_model.bin"



# image crop and data preprocessing 
crop_size = (512, 512)
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
    decoder_freeze=False,
    data_preprocessor=data_preprocessor,
    shared_image_embedding=dict(
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        _delete_=True,
        type='RSSamVisionEncoder',  
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
                peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        ),
        extra_config=dict( 
        output_hidden_states=True,
        image_size=crop_size[0],    
        ),
    ),
    neck=dict(
        type='RSFPN',
        feature_aggregator=dict(  
            type='RSFeatureAggregator',
            in_channels=hf_sam_pretrain_name,
            out_channels=256,
            hidden_channels=32,
            select_layers=range(1, 12+1, 2),
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
        decoder_plus=True,
        mask_decoder=dict(
            hf_pretrain_name=hf_sam_pretrain_name,
            init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
        ),
        per_pointset_point=prompt_shape[1],
        with_sincos=True,
        num_things_classes=num_classes,
        num_queries=prompt_shape[0],
        loss_cls=dict(
            class_weight=[1.0] * num_classes + [0.1])
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_classes
    ),
    test_cfg=dict(
        max_per_image=prompt_shape[0],
    )
)



# dataset 
dataset_type = 'WuhanUISDataset'    
code_root = '/home/dyl/PromptSAM'
data_root = '/home/dyl/PromptSAM/data/wuhanUIS_2022'
target_data_root = '/home/dyl/PromptSAM/data/wuhanUIS_0519'




# training data pipelines
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  
    dict(type='Resize', 
         scale=crop_size,
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
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
batch_size = 1
batch_size_per_gpu = 1
num_workers = 8
persistent_workers = True

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=code_root + '/data/wuhanUIS_2022/annotations/wuhanUIS_2022_train.json',
        data_prefix=dict(img='imgs/train/image'),
        pipeline=train_pipeline,
    ) 
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=code_root + '/data/wuhanUIS_2022/annotations/wuhanUIS_2022_val.json',
        data_prefix=dict(img='imgs/val/image'),
        pipeline=test_pipeline,
    )
)

test_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=target_data_root,
        ann_file=code_root + '/data/wuhanUIS_0519/annotations/wuhanUIS_0519_test.json',
        data_prefix=dict(img='imgs/test/image'),
        pipeline=test_pipeline,
    )
)



# training config
max_epochs = 300
base_lr = 0.0001
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
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




#### DeepSpeed Configs
runner_type = 'FlexibleRunner'
# Runner = 'FlexibleRunner'
strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        auto_cast=False,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    gradient_clipping=0.1,
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        allgather_bucket_size=2e8,
        reduce_scatter=True,
        reduce_bucket_size='auto',
        overlap_comm=True,
        contiguous_gradients=True,
    ),
)
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    )
)

# #### AMP training config
# runner_type = 'Runner'
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     dtype='float16',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05)
# )




