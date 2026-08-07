"""Cascade Mask R-CNN + Swin-S, in MMDetection 3.x form.

Same as the Swin-T variant -- including the ConvFCBBoxHead (4 shared convs +
1 shared FC, ``reg_class_agnostic=False``) that the Swin repo's checkpoints
expect -- but with Swin-S depths ([2, 2, 18, 2]) and its larger
stochastic-depth rate.  See the Swin-T file for why the head matters.

The published full-precision result is 51.9 box AP / 45.0 mask AP.

Note the size: Swin-S has 24 transformer blocks, so the backbone yields 100
quantizable components (vs 52 for Swin-T). At 3 candidate bits that is 300
Omega measurements, so a mixed-precision run takes roughly twice as long as
the Swin-T one.
"""

_base_ = './cascade-mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'

model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3))
