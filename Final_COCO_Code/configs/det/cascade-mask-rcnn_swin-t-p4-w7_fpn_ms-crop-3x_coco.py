"""Cascade Mask R-CNN + Swin-T, in MMDetection 3.x form.

MMDetection 3.x ships Swin configs for Mask R-CNN and RetinaNet but not for
Cascade Mask R-CNN -- those checkpoints live in Microsoft's
Swin-Transformer-Object-Detection repo, which is mmdet 2.x.  This config
recreates that model on the 3.x side.

Two things differ from mmdet's stock ``cascade-mask-rcnn_r50_fpn``, and both
matter for loading the Swin-repo weights:

* **The bbox heads.**  Stock mmdet uses ``Shared2FCBBoxHead`` (two shared FCs,
  ``reg_class_agnostic=True`` so ``fc_reg`` emits 4 numbers).  The Swin repo
  uses ``ConvFCBBoxHead`` with 4 shared convs + 1 shared FC, BN, GIoU loss and
  ``reg_class_agnostic=False`` so ``fc_reg`` emits 80 x 4 = 320.  Using the
  stock head leaves ``shared_fcs.1`` and every ``fc_reg`` randomly
  initialised -- the model still loads, still runs, and reports nonsense.

* **The backbone**, swapped from ResNet to Swin-T with the FPN widened to
  Swin's channel dims.

Pair it with a checkpoint converted by ``tools/convert_swin_det_ckpt.py``.
The published full-precision result is 50.4 box AP / 43.7 mask AP --
reproducing that with --eval-fp-only is the only real proof that the config
and the conversion are both right.
"""

_base_ = 'mmdet::cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'

# The three cascade stages differ only in their regression target stds and
# (at train time) their IoU thresholds.
_bbox_head_common = dict(
    type='ConvFCBBoxHead',
    num_shared_convs=4,
    num_shared_fcs=1,
    in_channels=256,
    conv_out_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=80,
    reg_class_agnostic=False,
    reg_decoded_bbox=True,
    # BN, not SyncBN: identical in eval (both use running stats) and identical
    # key names, but SyncBN needs an initialised process group.
    norm_cfg=dict(type='BN', requires_grad=True),
    loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_bbox=dict(type='GIoULoss', loss_weight=10.0))


def _stage(target_stds):
    head = dict(_bbox_head_common)
    head['bbox_coder'] = dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[0., 0., 0., 0.],
        target_stds=target_stds)
    return head


model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=None),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=[
            _stage([0.1, 0.1, 0.2, 0.2]),
            _stage([0.05, 0.05, 0.1, 0.1]),
            _stage([0.033, 0.033, 0.067, 0.067]),
        ]))
