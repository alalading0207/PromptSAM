import argparse
import glob
import os
import os.path as osp
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmengine.fileio import dump
from mmengine.utils import (Timer, mkdir_or_exist, track_parallel_progress,
                            track_progress)


# Collect image and label paths
def collect_files(img_dir, gt_dir):
    files = []
    img_files = glob.glob(osp.join(img_dir, 'image/*.tif'))
    for img_file in img_files:   
        segm_file = osp.join(gt_dir, osp.splitext(osp.basename(img_file))[0] + '.npy')

        print('11111',gt_dir)
        # segm_file = gt_dir + '/label/' + os.path.basename(img_file)    
        files.append((img_file, segm_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


# Load annotation data with multiprocessing
def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = track_parallel_progress(load_img_info, files, nproc=nproc)
    else:
        images = track_progress(load_img_info, files)

    return images


# Extract information and annotations for a single image
def load_img_info(files):
    img_file, segm_file = files   

    parent_dir = osp.dirname(segm_file)  
    base_name = osp.basename(parent_dir) 
    output_mask_dir = osp.join(osp.dirname(parent_dir), f"{base_name}_instance") 
    os.makedirs(output_mask_dir, exist_ok=True) 
    print(segm_file)
    print(output_mask_dir)

    segm_img = np.load(segm_file)
    if segm_img.shape[0] == 1:
        segm_img = segm_img.squeeze(0) 
    assert segm_img.ndim == 2, f"Expected 2D soft mask, but got {segm_img.ndim}D in {segm_file}"
    assert segm_img.max() <= 1.0 and segm_img.min() >= 0.0, f"Soft mask values out of range in {segm_file}"

    # Calculate the number of connected components, thresholding at 0.3 to separate instances
    num_labels, instances, stats, centroids = cv2.connectedComponentsWithStats((segm_img > 0.3).astype(np.uint8), connectivity=4)
    
    anno_info = []
    for inst_id in range(1, num_labels):        

        category_id = 1
        mask = np.asarray(instances == inst_id, dtype=np.float32, order='F') * segm_img 
        if mask.max() < 1e-5:   
            print(f'Ignore empty instance: {inst_id} in {segm_file}')
            continue

        # calculate area and bounding box
        area = np.sum(mask)
        y_coords, x_coords = np.where(mask > 0)         
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

        # save mask as .npy file and store the path
        mask_file = f"{osp.splitext(osp.basename(segm_file))[0]}_mask_{inst_id}.npy"
        mask_path = osp.join(output_mask_dir, mask_file)
        np.save(mask_path, mask)

        anno = dict(
            iscrowd=0,                 
            category_id=category_id,    
            bbox=bbox,
            area=float(area),
            segmentation=mask_path)
        anno_info.append(anno)


    # Create image metadata
    img_info = dict(
        file_name=osp.basename(img_file),   
        height=segm_img.shape[0],
        width=segm_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.basename(segm_file))  

    return img_info

'''format:：
{
    'file_name': 'example.tif',
    'height': 480,
    'width': 640,
    'anno_info': [
        {
            'iscrowd': 0,
            'category_id': 1,
            'bbox': [10.0, 20.0, 100.0, 200.0],
            'area': 20000.0,
            'segmentation': [
                [0.0, 0.2, 0.4, ...]
            ]
        },
        ...
    ],
    'segm_file': 'example_label.tif'
}

'''


# Convert the extracted single image information and annotations to COCO format
def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []

    for image_info in image_infos:
        image_info['id'] = img_id                  
        anno_infos = image_info.pop('anno_info')   

        out_json['images'].append(image_info)       

        for anno_info in anno_infos:
            anno_info['image_id'] = img_id         
            anno_info['id'] = ann_id             
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1

    cat = dict(id=1, name='urbanv')  
    out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert wuhanUIS urbanv annotations to COCO format')
    parser.add_argument('--whu_path', default='/home/dyl/RSPrompter_UIS/data/wuhanUIS_0519', help='whu data path')
    parser.add_argument('--img-dir', default='imgs', type=str)
    parser.add_argument('--gt-dir', default='imgs/train/20260123_114157_220_soft', type=str)
    parser.add_argument('-o', '--out-dir', default='/home/dyl/RSPrompter_UIS/data/wuhanUIS_0519/annotations_20260123_114157_220', help='output path')
    parser.add_argument('--nproc', default=0, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    whu_path = args.whu_path
    out_dir = args.out_dir if args.out_dir else whu_path
    mkdir_or_exist(out_dir)

    img_dir = osp.join(whu_path, args.img_dir)
    gt_dir = osp.join(whu_path, args.gt_dir)

    # print('img_dir',img_dir)    # /home/dyl/RSPrompter_UIS/data/wuhanUIS_0519/imgs
    # print('gt_dir',gt_dir)      # /home/dyl/RSPrompter_UIS/data/wuhanUIS_0519/imgs/train/1004_soft

    set_name = dict(
        train='wuhanUIS_0519_soft_train.json'
    )

    # split: train
    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with Timer(print_tmpl='It took {}s to convert wuhanUIS annotation'):
            files = collect_files(osp.join(img_dir, split), osp.join(gt_dir))
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()



# python whuvsoft2coco.py --whu_path /path/to/data --img-dir imgs --gt-dir gt --out-dir /path/to/output