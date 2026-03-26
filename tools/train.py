# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append(sys.path[0] + '/..')  # 将系统路径调整为包含上一级目录，确保脚本能找到必要的模块
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

import domain_adapt.adapt_loops.adapt_epoch_based_train_loop



# 配置文件路径、工作目录、是否启用自动混合精度训练、是否自动调整学习率、是否从检查点恢复训练
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')  #如果指定了检查点路径则从该路径恢复；如果没有指定则尝试从工作目录中最新的检查点恢复
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')  # 覆盖使用的配置中的某些设置时，以 xxx=yyy 格式表示的键值对将被合并到配置文件中
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')  # 指定launcher类型
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve training speed.减少重复编译次数，提高训练速度。
    setup_cache_size_limit_of_dynamo()  # 用于设置 dynamo 缓存大小限制

    # load config
    cfg = Config.fromfile(args.config)  # 进TextToImageRetrievalInferencer
    cfg.launcher = args.launcher
    if args.cfg_options is not None: 
        cfg.merge_from_dict(args.cfg_options)  # 合并命令行中提供的配置选项

    # # work_dir is determined in this priority: CLI > segment in file > filename  设置工作目录
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get('work_dir', None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     cfg.work_dir = osp.join('./work_dirs',
    #                             osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training 是否开启AMP
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR 是否自动调整学习率
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume   是否resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config  根觉配置文件构建运行器
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)       # 这里只build model
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)  # 配置文件中 runner_type='FlexibleRunner'，于是从注册表中构建一个自定义运行器

    # start training
    runner.train()


if __name__ == '__main__':
    main()
