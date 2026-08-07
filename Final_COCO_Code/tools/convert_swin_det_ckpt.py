"""Convert a Swin-Transformer-Object-Detection (mmdet 2.x) checkpoint to mmdet 3.x.

Microsoft's Swin detection repo publishes the Cascade Mask R-CNN + Swin
checkpoints that the ViT-PTQ tables use, but that repo is built on mmdet 2.x
and the original Swin backbone implementation.  Two things differ from
mmdet 3.x:

* **Naming.**  ``layers.{i}`` -> ``stages.{i}``, ``patch_embed.proj`` ->
  ``patch_embed.projection``, ``attn.`` -> ``attn.w_msa.``,
  ``mlp.fc1`` -> ``ffn.layers.0.0``, ``mlp.fc2`` -> ``ffn.layers.1``.

* **Patch-merging weight order.**  The original Swin builds the 4x-channel
  input by slicing and concatenating; mmdet's ``PatchMerging`` uses
  ``nn.Unfold``, which emits the four sub-patches in a *different* order.  The
  ``reduction`` weight columns and the ``norm`` parameters therefore have to be
  permuted, not just renamed.  Getting this wrong yields a model that loads
  cleanly and predicts garbage.

Rather than re-derive that permutation, this script calls mmdet's own
``swin_converter`` -- the function mmdet uses for ``convert_weights=True`` --
so the tricky part is handled by tested upstream code.  Detector heads (RPN,
cascade bbox/mask heads) and the FPN keep the same key names between 2.x and
3.x and are copied through unchanged.

Usage
-----
    python tools/convert_swin_det_ckpt.py \
        --src ./checkpoints/det/cascade_mask_rcnn_swin_tiny_patch4_window7.pth \
        --dst ./checkpoints/det/cascade_mask_rcnn_swin_t_3x_mmdet3.pth \
        --config ./configs/det/cascade-mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py

``--config`` is optional but strongly recommended: it builds the target model
and diffs the key sets, so a bad conversion is caught here instead of showing
up as a mysteriously low AP.  The real proof is still an --eval-fp-only run
reproducing the published number (50.4 / 43.7 for Swin-T).
"""

import argparse
import os
from collections import OrderedDict

import torch


def load_state_dict(path):
    ckpt = torch.load(path, map_location='cpu')
    for key in ('state_dict', 'model'):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    return ckpt


def convert(state_dict):
    """Return an mmdet 3.x state dict from an mmdet 2.x / Swin-repo one."""
    try:
        from mmdet.models.backbones.swin import swin_converter
    except ImportError as e:
        raise ImportError(
            "Could not import mmdet's swin_converter "
            "(mmdet.models.backbones.swin.swin_converter). It handles the "
            "patch-merging weight reordering, so conversion without it would "
            "be silently wrong. Check your mmdet install. Original error: %s" % e)

    backbone, rest = OrderedDict(), OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            backbone[k[len('backbone.'):]] = v
        else:
            rest[k] = v

    if not backbone:
        raise ValueError(
            "No 'backbone.*' keys found in the source checkpoint -- this does "
            "not look like an mmdet detection checkpoint.")

    # swin_converter re-adds the 'backbone.' prefix itself.
    converted = swin_converter(backbone)

    out = OrderedDict()
    out.update(converted)
    out.update(rest)
    return out, len(backbone), len(rest)


def verify(out, config_path):
    """Diff the converted keys against what the target config's model wants."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS

    init_default_scope('mmdet')
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    expected = model.state_dict()

    exp_keys, got_keys = set(expected), set(out)
    missing = sorted(exp_keys - got_keys)
    unexpected = sorted(got_keys - exp_keys)
    mismatched = [(k, tuple(expected[k].shape), tuple(out[k].shape))
                  for k in sorted(exp_keys & got_keys)
                  if tuple(expected[k].shape) != tuple(out[k].shape)]

    print(f"\nVerification against {os.path.basename(config_path)}:")
    print(f"  model expects      : {len(exp_keys)} keys")
    print(f"  checkpoint provides: {len(got_keys)} keys")
    print(f"  missing            : {len(missing)}")
    print(f"  unexpected         : {len(unexpected)}")
    print(f"  shape mismatches   : {len(mismatched)}")

    for label, items in (('MISSING', missing), ('UNEXPECTED', unexpected)):
        for k in items[:15]:
            print(f"    {label}: {k}")
        if len(items) > 15:
            print(f"    ... and {len(items) - 15} more")
    for k, want, got in mismatched[:15]:
        print(f"    SHAPE: {k}  model={want}  ckpt={got}")

    ok = not missing and not mismatched
    if ok:
        print("  -> key sets match. Confirm with an --eval-fp-only run.")
    else:
        print("  -> MISMATCH. Do not trust this checkpoint; the AP would be wrong.")
    return ok


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='mmdet 2.x / Swin-repo .pth')
    p.add_argument('--dst', required=True, help='output mmdet 3.x .pth')
    p.add_argument('--config', default=None,
                   help='target mmdet 3.x config, for key verification')
    args = p.parse_args()

    print(f"Reading {args.src} ...")
    state = load_state_dict(args.src)
    print(f"  {len(state)} tensors")

    out, n_backbone, n_rest = convert(state)
    print(f"Converted {n_backbone} backbone tensors "
          f"(renamed + patch-merging reordered); "
          f"copied {n_rest} neck/head tensors unchanged.")

    if args.config and not verify(out, args.config):
        # Refuse to write. A checkpoint that fails verification still *loads*
        # -- mmdet only warns about mismatched keys and leaves them randomly
        # initialised -- so an unusable file on disk is worse than no file.
        raise SystemExit(
            "\nABORTED: not writing the checkpoint.\n"
            "The converted keys do not match the target config, so the model "
            "would load with randomly initialised layers and silently report "
            "wrong AP. Fix the config (or check you used the right source "
            "checkpoint) and re-run.")

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save({'state_dict': out, 'meta': {'converted_from': args.src}}, args.dst)
    print(f"\nWrote {args.dst}")


if __name__ == '__main__':
    main()
