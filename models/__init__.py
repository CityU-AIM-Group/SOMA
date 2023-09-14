# ------------------------------------------------------------------------
# Modified by Wuyang LI
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
def build_model(cfg):
    if cfg.AOOD.OW_DETR_ON:
        print('using ow detr!')
        from .ow_detr import build
    elif cfg.AOOD.MOTIF_ON:
        print('using motif detr!')
        from .motif_detr import build
    else:
        print('using def detr!')
        from .deformable_detr import build
    return build(cfg)


