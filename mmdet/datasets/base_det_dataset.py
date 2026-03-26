# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import is_abs

from ..registry import DATASETS


@DATASETS.register_module()
class BaseDetDataset(BaseDataset):
    """Base dataset for detection.

    Args:
        proposal_file (str, optional): Proposals file path. Defaults to None.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        return_classes (bool): Whether to return class information
            for open vocabulary-based algorithms. Defaults to False.
    """

    def __init__(self,   # 配置文件没写关于这部分的配置，所以为默认值
                 *args,
                 seg_map_suffix: str = '.png',  # 分割图的文件后缀
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 return_classes: bool = False, 
                 **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes  # 不返回class信息
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(*args, **kwargs)   # 跳到LoadAnnotations

    def full_init(self) -> None:  # 一系列初始化操作
        """Load annotation file and set ``BaseDataset._fully_initialized`` to 
        True.加载注释文件, 并设置BaseDataset._fully_initialized = True
        """
        if self._fully_initialized:
            return
        # load data information     如何从注释文件中加载注释信息 包括 img\box\mask\cat
        self.data_list = self.load_data_list() 

        # filter illegal data, such as data that has no annotations.  过滤非法数据
        self.data_list = self.filter_data()

        # Get subset data according to indices. NONE
        # if self._indices is not None:
        #     self.data_list = self._get_unserialized_subset(self._indices)

        # data_bytes将data_list序列化为字节流   data_address计算每个数据项的大小和起始位置
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data() # 将字节流存放在data_address地址中，共享内存访问

        self._fully_initialized = True  # 全部初始化完


    def get_cat_ids(self, idx: int) -> List[int]:
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
