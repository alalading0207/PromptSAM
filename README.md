# LongConsistMap
This repository provides the implementation of our paper: "LongConsistMap: A SAM-based Long-term Consistency Mapping Framework for Urban Informal Settlement Mapping and Governance-oriented Analysis"

# Introduction
LongConsistMap aims to generate long-term consistent UIS maps from heterogeneous historical imagery and support governance-oriented spatiotemporal analysis.

LongConsistMap consists of three sequential components:

1. **Task Adaptation** – establishing transferable UIS prompt priors from annotated reference imagery;
   
2. **Temporal Adaptation** – progressively transferring prompt semantics to annotation-scarce historical imagery through prototype-constrained cross-temporal self-training;
 
3. **Temporal Correction** – refining multi-temporal predictions according to UIS evolutionary regularities to generate long-term consistent UIS maps.

This repository implements the first two components of LongConsistMap, namely **Task Adaptation** and **Temporal Adaptation**. The core model is referred to as **PromptSAM**, which serves as the foundation of LongConsistMap. Specifically, PromptSAM reconstructs the prompt encoder of SAM through task-adaptive prompt learning and prototype-constrained temporal adaptation, enabling both single-temporal UIS detection and cross-temporal UIS detection across heterogeneous historical imagery.

The third component, Temporal Correction, together with the subsequent governance-oriented spatiotemporal analysis, operate on the generated long-term consistent UIS maps and are therefore beyond the scope of the current code release.

Overall, PromptSAM provides the transferable prompt priors and cross-temporal adaptation capability that form the foundation of LongConsistMap for long-term consistent UIS mapping.

# Dataset Availability

The “WuhanUIS” dataset used in this project is currently not publicly available. As further research on long-term UIS is ongoing, the complete dataset will remain confidential at this stage. We fully acknowledge the importance of open and reproducible research, and we plan to release the dataset in the future.

# PromptSAM Workflow

This repository provides the workflow for PromptSAM-based domain adaptation on the WuhanUIS dataset, including source-domain training, inference, pseudo-label generation, prototype computation, and target-domain adaptation.


### 1. Source Domain Training (WuhanUIS 2022)

**Config file:** `./configs/prompterSAM/wuhanUIS_512.py`

**Command:**
```bash
python tools/train.py configs/prompterSAM/wuhanUIS_512.py
```

### 2. Source Domain Inference (2022) with Source Model

**Config file:** `./configs/prompterSAM/wuhanUIS_512-2022pred.py`

**Command:**
```bash
python demo/image_demo.py data/wuhanUIS_2022/imgs/test/image \
    configs/prompterSAM/wuhanUIS_512-2022pred.py \
    --weights work_dirs/prompterSAM/....pth \
    --out-dir work_dirs/output/...
```

### 3. Target Domain Inference (0519) with Source Model

**Config file:** `./configs/prompterSAM/wuhanUIS_512-0519pred.py`

**Command:**
```bash
python demo/image_demo.py data/wuhanUIS_0519/imgs/test/image \
    configs/prompterSAM/wuhanUIS_512-0519pred.py \
    --weights work_dirs/prompterSAM/....pth \
    --out-dir work_dirs/output/...
```


### 4. Soft Pseudo-Label Generation for Domain Adaptation

**Config file:** `./configs/prompterSAM/wuhanUIS_512_soft.py`

**Command:**
```bash
python demo/image_demo_adapt.py data/wuhanUIS_0519/imgs/train/image \
    configs/prompterSAM/wuhanUIS_512_soft.py \
    --weights work_dirs/prompterSAM/....pth \
    --out-dir domain_adapt/pseudo/soft_label
```


### 5. Prototype Computation for Domain Adaptation

**Config file:** `./configs/prompterSAM/wuhanUIS_512_prototype.py`

**Command:**
```bash
python demo/image_demo_adapt.py data/wuhanUIS_0519/imgs/train/image \
    configs/prompterSAM/wuhanUIS_512_prototype.py \
    --weights work_dirs/prompterSAM/....pth \
    --out-dir domain_adapt/prototype/prototypes
```


### Prepare Soft Pseudo-Labels for Domain Adaptation Training
To enable domain adaptation training, you need to generate a training-ready pseudo-label annotation file (for example, soft_train.json).

**Step 1. Move soft pseudo-label files into the target dataset**

  Move the files in: `./domain_adapt/pseudo/soft_label/vis` to: `data/wuhanUIS_0519/imgs/train/soft`
  
**Step 2. Modify the soft pseudo-label to COCO conversion script**

  Edit the following arguments in `./domain_adapt/whUISsoft2coco.py`:
  ```bash
  parser.add_argument('--gt-dir', default='imgs/train/0215_soft', type=str)
  parser.add_argument('-o', '--out-dir', default='/home/dyl/RSPrompter_UIS/data/wuhanUIS_0519/annotations_0215', help='output path')

```
**Step 3. Convert soft pseudo-labels to COCO format**

  Run: `python domain_adapt/whUISsoft2coco.py`

**Step 4. Update the domain adaptation training config**

  Modify the ann_file in target_train_dataloader to use the generated pseudo-label JSON file.


### 6. Domain Adaptation Training (Source + Target)

**Config file:** `./configs/prompterSAM/wuhanUIS_512_adapt.py`

**Command:**
```bash
python tools/train.py /home/dyl/RSPrompter_UIS/configs/prompterSAM/wuhanUIS_512_adapt.py
```
**Preparation**

Before training, make sure the following items are properly configured:

**Step 1. Prototype path: set the correct prototype file path**

**Step 2. Pretrained model: set the pretrained model in DeepSpeed format**

**Step 3. Pseudo-labels: set the ann_file in target_train_dataloader to the pseudo-label JSON file**

### 7. Target Domain Inference (0519) with Target Model after Domain Adaptation

**Config file:** `./configs/prompterSAM/wuhanUIS_512_adapt_0519pred.py`

**Command:**
```bash
python demo/image_demo_adapt.py data/wuhanUIS_0519/imgs/test/image \
    configs/prompterSAM/wuhanUIS_512_adapt.py \
    --weights work_dirs/prompterSAM/....pth \
    --out-dir work_dirs/output/...
```

### Notes
Please replace work_dirs/prompterSAM/....pth with the actual checkpoint path.

Please update all dataset paths according to your local environment.

Make sure the configuration files, prototype files, pretrained weights, and pseudo-label annotations are correctly prepared before running domain adaptation training.
