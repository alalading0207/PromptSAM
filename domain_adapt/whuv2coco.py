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
        segm_file = gt_dir + '/label/' + os.path.basename(img_file)    
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
    print(img_file)
    print(segm_file)


    # Process the annotation first
    segm_img = mmcv.imread(segm_file, flag='unchanged', backend='cv2')
    # Calculate the number of connected components 
    num_labels, instances, stats, centroids = cv2.connectedComponentsWithStats(segm_img, connectivity=4)

    anno_info = []
    for inst_id in range(1, num_labels):  

        category_id = 1
        mask = np.asarray(instances == inst_id, dtype=np.uint8, order='F') 
        if mask.max() < 1:
            print(f'Ignore empty instance: {inst_id} in {segm_file}')
            continue

        mask_rle = maskUtils.encode(mask[:, :, None])[0]
        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)       # bbox format[x_min, y_min, width, height]

        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()   


        anno = dict(
            iscrowd=0,                
            category_id=category_id,   
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)     
        anno_info.append(anno)


    # Create image metadata
    img_info = dict(
        file_name=osp.basename(img_file),   # image name
        height=segm_img.shape[0],
        width=segm_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.basename(segm_file))  # label name

    return img_info

'''format:
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
            'segmentation': {...}
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
    parser.add_argument('--gt-dir', default='imgs', type=str)
    parser.add_argument('-o', '--out-dir', default='/home/dyl/RSPrompter_UIS/data/wuhanUIS_0519/annotations', help='output path')
    parser.add_argument(
        '--nproc', default=0, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    whu_path = args.whu_path
    out_dir = args.out_dir if args.out_dir else whu_path
    mkdir_or_exist(out_dir)

    img_dir = osp.join(whu_path, args.img_dir)
    gt_dir = osp.join(whu_path, args.gt_dir)

    set_name = dict(
        train='wuhanUIS_0519_train.json',
        val='wuhanUIS_0519_val.json',
        test='wuhanUIS_0519_test.json'
    )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with Timer(print_tmpl='It took {}s to convert wuhanUIS annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()


# wuhanUIS_2022
# python whuvsoft2coco.py --whu_path /home/dyl/RSPrompter_UIS/data/wuhanUIS_2022 --out-dir /home/dyl/RSPrompter_UIS/data/wuhanUIS_2022/annotations

# wuhanUIS_0519
# python whuvsoft2coco.py --whu_path /home/dyl/RSPrompter_UIS/data/wuhanUIS_0519 --out-dir /home/dyl/RSPrompter_UIS/data/wuhanUIS_0519/annotations
