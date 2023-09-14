# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    else:
        return dataset


def build_dataset(image_set, cfg,multi_task_eval_id=4):
    if cfg.DATASET.DATASET_FILE == 'coco':
        return build_coco(image_set, cfg)
    if cfg.DATASET.DATASET_FILE == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, cfg)
    DAOD_dataset = [
        'cityscapes_to_foggy_cityscapes',
        'sim10k_to_cityscapes_caronly',
        'cityscapes_to_bdd_daytime',
        'pascal_to_clipart',
    ]
    if cfg.DATASET.DATASET_FILE in DAOD_dataset:
        from .DAOD import build
        return build(image_set, cfg, multi_task_eval_id=multi_task_eval_id)
    raise ValueError(f'dataset {cfg.DATASET.DATASET_FILE} not supported')
