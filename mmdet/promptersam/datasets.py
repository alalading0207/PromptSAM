from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS




@DATASETS.register_module()
class WuhanUISDataset(CocoDataset):
    METAINFO = {
        'classes': ['urbanv'],
        'palette': [(0, 255, 0)]
    }


@DATASETS.register_module()
class WuhanUISDataset_2022(CocoDataset):
    METAINFO = {
        'classes': ['urbanv','non-urbanv'],
        'palette': [(255, 255, 255),(0, 0, 0)]
    }


@DATASETS.register_module()
class WuhanUISDataset_0519(CocoDataset):
    METAINFO = {
        'classes': ['urbanv','non-urbanv'],
        'palette': [(255, 255, 255),(0, 0, 0)]
    }

