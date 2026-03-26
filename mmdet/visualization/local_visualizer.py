# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmengine.visualization import Visualizer

from ..evaluation import INSTANCE_OFFSET
from ..registry import VISUALIZERS       
from ..structures import DetDataSample
from ..structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from .palette import _get_adaptive_scales, get_palette, jitter_color  
from ..structures import SegDataSample 
from ..utils import  get_palette_seg, get_classes

import torch
import torch.nn.functional as F

@VISUALIZERS.register_module()
class DetLocalVisualizer(Visualizer):
    """MMDetection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet.structures import DetDataSample
        >>> from mmdet.visualization import DetLocalVisualizer

        >>> det_local_visualizer = DetLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.Tensor([[1, 2, 2, 5]])
        >>> gt_instances.labels = torch.randint(0, 2, (1,))
        >>> gt_det_data_sample = DetDataSample()
        >>> gt_det_data_sample.gt_instances = gt_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample)
        >>> det_local_visualizer.add_datasample(
        ...                       'image', image, gt_det_data_sample,
        ...                        out_file='out_file.jpg')
        >>> det_local_visualizer.add_datasample(
        ...                        'image', image, gt_det_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.bboxes = torch.Tensor([[2, 4, 4, 8]])
        >>> pred_instances.labels = torch.randint(0, 2, (1,))
        >>> pred_det_data_sample = DetDataSample()
        >>> pred_det_data_sample.pred_instances = pred_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample,
        ...                         pred_det_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        image[:] = 0
        self.set_image(image)

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks        
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()      
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)     

            palette = (255, 255, 255)      
            max_label = int(max(labels) if len(labels) > 0 else 0)
            # mask color （0，255，0）
            mask_color = palette if self.mask_color is None else self.mask_color 
            mask_palette = get_palette(mask_color, max_label + 1)               
            colors = [jitter_color(mask_palette[label]) for label in labels]    
            text_palette = get_palette(self.text_color, max_label + 1)         
            text_colors = [text_palette[label] for label in labels]            

            # draw segm
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)   

            if len(labels) > 0 and \
                    ('bboxes' not in instances or
                     instances.bboxes.sum() == 0):                            
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    if 'label_names' in instances:
                        label_text = instances.label_names[i]
                    else:
                        label_text = classes[
                            label] if classes is not None else f'class {label}'
                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f': {score}'

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])
        return self.get_image()
    
     
    def _draw_instances_large(self, image: np.ndarray, large_masks: torch.Tensor,
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
      
        image[:] = 0
        self.set_image(image)

        label = 0
        large_masks = large_masks.numpy()      
        large_masks = large_masks.astype(bool)      

        palette = (255, 255, 255)      
        mask_color = palette if self.mask_color is None else self.mask_color 
        mask_palette = get_palette(mask_color, label + 1)            

        self.draw_binary_masks(large_masks, colors=mask_palette[label], alphas=self.alpha)  

            
        return self.get_image()

    def _draw_panoptic_seg(self, image: np.ndarray,
                           panoptic_seg: ['PixelData'],
                           classes: Optional[List[str]],
                           palette: Optional[List]) -> np.ndarray:
        """Draw panoptic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            panoptic_seg (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            classes (List[str], optional): Category information.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        num_classes = len(classes)

        panoptic_seg_data = panoptic_seg.sem_seg[0]

        ids = np.unique(panoptic_seg_data)[::-1]

        if 'label_names' in panoptic_seg:
            # open set panoptic segmentation
            classes = panoptic_seg.metainfo['label_names']
            ignore_index = panoptic_seg.metainfo.get('ignore_index',
                                                     len(classes))
            ids = ids[ids != ignore_index]
        else:
            ids = ids[ids != num_classes]

        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (panoptic_seg_data[None] == ids[:, None, None])

        max_label = int(max(labels) if len(labels) > 0 else 0)

        mask_color = palette if self.mask_color is None \
            else self.mask_color
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]

        self.set_image(image)

        # draw segm
        polygons = []
        for i, mask in enumerate(segms):
            contours, _ = bitmap_to_polygon(mask)
            polygons.extend(contours)
        self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
        self.draw_binary_masks(segms, colors=colors, alphas=self.alpha)

        # draw label
        areas = []
        positions = []
        for mask in segms:
            _, _, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            max_id = np.argmax(stats[1:, -1]) + 1
            positions.append(centroids[max_id])
            areas.append(stats[max_id, -1])
        areas = np.stack(areas, axis=0)
        scales = _get_adaptive_scales(areas)

        text_palette = get_palette(self.text_color, max_label + 1)
        text_colors = [text_palette[label] for label in labels]

        for i, (pos, label) in enumerate(zip(positions, labels)):
            label_text = classes[label]

            self.draw_texts(
                label_text,
                pos,
                colors=text_colors[i],
                font_sizes=int(13 * scales[i]),
                bboxes=[{
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                }],
                horizontal_alignments='center')
        return self.get_image()

    def _draw_sem_seg(self, image: np.ndarray, sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List]) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        sem_seg_data = sem_seg.sem_seg
        if isinstance(sem_seg_data, torch.Tensor):
            sem_seg_data = sem_seg_data.numpy()

        # 0 ~ num_class, the value 0 means background
        ids = np.unique(sem_seg_data)
        ignore_index = sem_seg.metainfo.get('ignore_index', 255)
        ids = ids[ids != ignore_index]

        if 'label_names' in sem_seg:
            # open set semseg
            label_names = sem_seg.metainfo['label_names']
        else:
            label_names = classes

        labels = np.array(ids, dtype=np.int64)
        colors = [palette[label] for label in labels]

        self.set_image(image)

        # draw semantic masks
        for i, (label, color) in enumerate(zip(labels, colors)):
            masks = sem_seg_data == label
            self.draw_binary_masks(masks, colors=[color], alphas=self.alpha)
            label_text = label_names[label]
            _, _, stats, centroids = cv2.connectedComponentsWithStats(
                masks[0].astype(np.uint8), connectivity=8)
            if stats.shape[0] > 1:
                largest_id = np.argmax(stats[1:, -1]) + 1
                centroids = centroids[largest_id]

                areas = stats[largest_id, -1]
                scales = _get_adaptive_scales(areas)

                self.draw_texts(
                    label_text,
                    centroids,
                    colors=(255, 255, 255),
                    font_sizes=int(13 * scales),
                    horizontal_alignments='center',
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)        
        classes = self.dataset_meta.get('classes', None)   
        palette = self.dataset_meta.get('palette', None)  

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()   # alraedy move data to cpu

        if draw_gt and data_sample is not None:    
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)
            if 'gt_sem_seg' in data_sample:
                gt_img_data = self._draw_sem_seg(gt_img_data,
                                                 data_sample.gt_sem_seg,
                                                 classes, palette)

            if 'gt_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_panoptic_seg(
                    gt_img_data, data_sample.gt_panoptic_seg, classes, palette)

        if draw_pred and data_sample is not None:   # True
            pred_img_data = image
            if 'pred_instances' in data_sample:    # pre-draw data_sample.pred_instances 
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]     # threshold is 0.3
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)

            if 'pred_sem_seg' in data_sample:
                pred_img_data = self._draw_sem_seg(pred_img_data,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette)

            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes, palette)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = image

        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)


    @master_only
    def add_datasample_large(
            self,
            name: str,
            image: np.ndarray,
            # data_sample: Optional['DetDataSample'] = None,
            predictions: list,     
            large_size:tuple,
            patch_size: int,
            patch_stride : int,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and merge large."""
        image = image.clip(0, 255).astype(np.uint8)        
        classes = self.dataset_meta.get('classes', None)    
        palette = self.dataset_meta.get('palette', None)  

        gt_img_data = None
        pred_img_data = image


        all_masks = []
        for pred in predictions : 
            pred_instances = pred.pred_instances            # pre-draw data_sample.pred_instances 
            pred_instances = pred_instances[pred_instances.scores > pred_score_thr]     # threshold is 0.3
            
            # combine multiple masks into one channel
            if pred_instances['masks'].size(0) > 1:
                mask = torch.max(pred_instances['masks'], dim=0)[0].unsqueeze(0)
            elif pred_instances['masks'].size(0) == 0:
                mask = torch.full((1, patch_size, patch_size), fill_value = -10.0)
            else:
                mask = pred_instances['masks']

            all_masks.append(mask)  

        # concatenate all masks and fold to large size
        all_masks = torch.cat(all_masks, dim=0).unsqueeze(1)   # (num_patches, 1, patch_size, patch_size)
        all_masks = all_masks.permute(1, 2, 3, 0)             # (1, patch_size, patch_size, num_patches)
        all_masks = all_masks.contiguous().view(1, 1 * patch_size * patch_size, -1)  # (1, patch_size*patch_size, num_patches)
        large_masks = F.fold(all_masks, output_size=large_size, kernel_size=(patch_size, patch_size), stride=(patch_stride, patch_stride))
        # binary masks
        large_masks = (large_masks > 0).squeeze(0)          
        # large_masks = large_masks.permute(1, 2, 0).repeat(1, 1, 3)
        # draw large masks
        pred_img_data = self._draw_instances_large(image, large_masks, classes, palette)


        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = image

        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)


def random_color(seed):
    """Random a color according to the input seed."""
    if sns is None:
        raise RuntimeError('motmetrics is not installed,\
                 please install it by: pip install seaborn')
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color







@VISUALIZERS.register_module()
class DetLocalVisualizer_adapt(Visualizer):
    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 stage: str = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        self.stage=stage
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha
        self.dataset_meta = {}

    def _draw_instances_adapt(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:

        if self.stage == 'get_softlabel':
            # labels = instances.labels
            masks = instances.masks

            if masks.shape[0] == 0: 
                _, height, width = masks.shape
                soft_label = np.zeros((1, height, width), dtype=np.float32)
            else:
                if isinstance(masks, torch.Tensor):
                    masks = masks.numpy()       
                elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                    masks = masks.to_ndarray()

                # masks[masks<0.3] = 0
                # merge multiple masks and calculate the maximum value for each pixel
                soft_label = masks.max(axis=0)
                soft_label = soft_label[None, ...]          
                
            return soft_label


        else:
            image[:] = 0
            self.set_image(image)


            labels = instances.labels
            masks = instances.masks         
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()      
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)    

            palette = (255, 255, 255)      
            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None else self.mask_color 
            mask_palette = get_palette(mask_color, max_label + 1)             
            colors = [jitter_color(mask_palette[label]) for label in labels]   
            text_palette = get_palette(self.text_color, max_label + 1)          
            text_colors = [text_palette[label] for label in labels]           

            
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)    

            
            return self.get_image()

    @master_only
    def add_datasample_adapt(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0,
            ) -> None:

        image = image.clip(0, 255).astype(np.uint8)         # shape H,W,C
        classes = self.dataset_meta.get('classes', None)    # 0 ='building'
        palette = self.dataset_meta.get('palette', None)    # 0 =(0, 255, 0)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()     

        if draw_pred and data_sample is not None:  
            pred_img_data = image

            if 'pred_instances' in data_sample: # labels\scores\masks
                pred_instances = data_sample.pred_instances
                # draw score>0.3
                pred_instances = pred_instances[pred_instances.scores > pred_score_thr]

                if len(pred_instances.scores) == 1 :
                    print('numbers=1')
                if len(pred_instances.scores) > 1 :
                    print('numbers>1')
                # draw soft
                pred_img_data = self._draw_instances_adapt(image, pred_instances, classes, palette)


        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:    
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        if self.stage == 'get_softlabel':
            pass
        else:
            self.set_image(drawn_img)

        if out_file is not None:    # '/domain_adapt/pseudo/soft/vis/2016_5636.npy'
            if out_file.endswith('.npy'):
                np.save(out_file, drawn_img)
            else:
                mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)


def random_color(seed):
    """Random a color according to the input seed."""
    if sns is None:
        raise RuntimeError('motmetrics is not installed,\
                 please install it by: pip install seaborn')
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color










@VISUALIZERS.register_module()
class TrackLocalVisualizer(Visualizer):
    """Tracking Local Visualizer for the MOT, VIS tasks.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
                Defaults to 0.8.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(name, image, vis_backends, save_dir)
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `TrackLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray,
                        instances: InstanceData) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)
        classes = self.dataset_meta.get('classes', None)

        # get colors and texts
        # for the MOT and VIS tasks
        colors = [random_color(_id) for _id in instances.instances_id]
        categories = [
            classes[label] if classes is not None else f'cls{label}'
            for label in instances.labels
        ]
        if 'scores' in instances:
            texts = [
                f'{category_name}\n{instance_id} | {score:.2f}'
                for category_name, instance_id, score in zip(
                    categories, instances.instances_id, instances.scores)
            ]
        else:
            texts = [
                f'{category_name}\n{instance_id}' for category_name,
                instance_id in zip(categories, instances.instances_id)
            ]

        # draw bboxes and texts
        if 'bboxes' in instances:
            # draw bboxes
            bboxes = instances.bboxes.clone()
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)
            # draw texts
            if texts is not None:
                positions = bboxes[:, :2] + self.line_width
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0])
                scales = _get_adaptive_scales(areas.cpu().numpy())
                for i, pos in enumerate(positions):
                    self.draw_texts(
                        texts[i],
                        pos,
                        colors='black',
                        font_sizes=int(13 * scales[i]),
                        bboxes=[{
                            'facecolor': [c / 255 for c in colors[i]],
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

        # draw masks
        if 'masks' in instances:
            masks = instances.masks
            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: DetDataSample = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: int = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.
        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (OptTrackSampleList): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT TrackDataSample.
                Default to True.
            draw_pred (bool): Whether to draw Prediction TrackDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (int): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            assert 'gt_instances' in data_sample
            gt_img_data = self._draw_instances(image, data_sample.gt_instances)

        if draw_pred and data_sample is not None:
            assert 'pred_track_instances' in data_sample
            pred_instances = data_sample.pred_track_instances
            if 'scores' in pred_instances:
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr].cpu()
            pred_img_data = self._draw_instances(image, pred_instances)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

@VISUALIZERS.register_module()
class SegLocalVisualizer(Visualizer):
    """Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        classes (list, optional): Input classes for result rendering, as the
            prediction of segmentation model is a segment map with label
            indices, `classes` is a list which includes items responding to the
            label indices. If classes is not defined, visualizer will take
            `cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is
            a list of color palette responding to the classes. Defaults to None.
        dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_
            visulizer will use the meta information of the dataset i.e. classes
            and palette, but the `classes` and `palette` have higher priority.
            Defaults to None.
        alpha (int, float): The transparency of segmentation mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import PixelData
        >>> from mmseg.structures import SegDataSample
        >>> from mmseg.visualization import SegLocalVisualizer

        >>> seg_local_visualizer = SegLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_sem_seg_data = dict(data=torch.randint(0, 2, (1, 10, 12)))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> gt_seg_data_sample = SegDataSample()
        >>> gt_seg_data_sample.gt_sem_seg = gt_sem_seg
        >>> seg_local_visualizer.dataset_meta = dict(
        >>>     classes=('background', 'foreground'),
        >>>     palette=[[120, 120, 120], [6, 230, 230]])
        >>> seg_local_visualizer.add_datasample('visualizer_example',
        ...                         image, gt_seg_data_sample)
        >>> seg_local_visualizer.add_datasample(
        ...                        'visualizer_example', image,
        ...                         gt_seg_data_sample, show=True)
    """  # noqa

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs)
        self.alpha: float = alpha
        self.set_dataset_meta(palette, classes, dataset_name)

    def _get_center_loc(self, mask: np.ndarray) -> np.ndarray:
        """Get semantic seg center coordinate.

        Args:
            mask: np.ndarray: get from sem_seg
        """
        loc = np.argwhere(mask == 1)

        loc_sort = np.array(
            sorted(loc.tolist(), key=lambda row: (row[0], row[1])))
        y_list = loc_sort[:, 0]
        unique, indices, counts = np.unique(
            y_list, return_index=True, return_counts=True)
        y_loc = unique[counts.argmax()]
        y_most_freq_loc = loc[loc_sort[:, 0] == y_loc]
        center_num = len(y_most_freq_loc) // 2
        x = y_most_freq_loc[center_num][1]
        y = y_most_freq_loc[center_num][0]
        return np.array([x, y])

    def _draw_sem_seg(self,
                      image: np.ndarray,
                      sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List],
                      with_labels: Optional[bool] = True) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.
            with_labels(bool, optional): Add semantic labels in visualization
                result, Default to True.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        if with_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # (0,1] to change the size of the text relative to the image
            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            fontColor = (255, 255, 255)
            if image.shape[0] < 300 or image.shape[1] < 300:
                thickness = 1
                rectangleThickness = 1
            else:
                thickness = 2
                rectangleThickness = 2
            lineType = 2

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            masks = masks.astype(np.uint8)
            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(masks[mask_num])
                text = classes[classes_id]
                (label_width, label_height), baseline = cv2.getTextSize(
                    text, font, fontScale, thickness)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     classes_color, -1)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     (0, 0, 0), rectangleThickness)
                mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                                   font, fontScale, fontColor, thickness,
                                   lineType)
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg

    def _draw_depth_map(self, image: np.ndarray,
                        depth_map: PixelData) -> np.ndarray:
        """Draws a depth map on a given image.

        This function takes an image and a depth map as input,
        renders the depth map, and concatenates it with the original image.
        Finally, it updates the internal image state of the visualizer with
        the concatenated result.

        Args:
            image (np.ndarray): The original image where the depth map will
                be drawn. The array should be in the format HxWx3 where H is
                the height, W is the width.

            depth_map (PixelData): Depth map to be drawn. The depth map
                should be in the form of a PixelData object. It will be
                converted to a torch tensor if it is a numpy array.

        Returns:
            np.ndarray: The concatenated image with the depth map drawn.

        Example:
            >>> depth_map_data = PixelData(data=torch.rand(1, 10, 10))
            >>> image = np.random.randint(0, 256,
            >>>                           size=(10, 10, 3)).astype('uint8')
            >>> visualizer = SegLocalVisualizer()
            >>> visualizer._draw_depth_map(image, depth_map_data)
        """
        depth_map = depth_map.cpu().data
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        if depth_map.ndim == 2:
            depth_map = depth_map[None]

        depth_map = self.draw_featmap(depth_map, resize_shape=image.shape[:2])
        out_image = np.concatenate((image, depth_map), axis=0)
        self.set_image(out_image)
        return out_image

    def set_dataset_meta(self,
                         classes: Optional[List] = None,
                         palette: Optional[List] = None,
                         dataset_name: Optional[str] = None) -> None:
        """Set meta information to visualizer.

        Args:
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.
            dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_
                visulizer will use the meta information of the dataset i.e.
                classes and palette, but the `classes` and `palette` have
                higher priority. Defaults to None.
        """  # noqa
        # Set default value. When calling
        # `SegLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        if dataset_name is None:
            dataset_name = 'cityscapes'
        classes = classes if classes else get_classes(dataset_name)
        palette = palette if palette else get_palette_seg(dataset_name)
        assert len(classes) == len(
            palette), 'The length of classes should be equal to palette'
        self.dataset_meta: dict = {'classes': classes, 'palette': palette}

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
            with_labels(bool, optional): Add semantic labels in visualization
                result, Defaults to True.
        """
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_sem_seg(image, data_sample.gt_sem_seg,
                                                 classes, palette, with_labels)

            if 'gt_depth_map' in data_sample:
                gt_img_data = gt_img_data if gt_img_data is not None else image
                gt_img_data = self._draw_depth_map(gt_img_data,
                                                   data_sample.gt_depth_map)

        if draw_pred and data_sample is not None:

            if 'pred_sem_seg' in data_sample:

                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_sem_seg(image,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette,
                                                   with_labels)

            if 'pred_depth_map' in data_sample:
                pred_img_data = pred_img_data if pred_img_data is not None \
                    else image
                pred_img_data = self._draw_depth_map(
                    pred_img_data, data_sample.pred_depth_map)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)