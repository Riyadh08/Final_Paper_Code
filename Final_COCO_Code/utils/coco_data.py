"""COCO data plumbing for the mixed-precision detection experiments.

Three jobs:

1. Load an MMDetection config / checkpoint and build the detector.
2. Build the standard COCO ``val2017`` test dataloader + ``CocoMetric``
   evaluator, so the reported box AP / mask AP are exactly mmdet's numbers.
3. Build a small **fixed-size** calibration batch list.

Why fixed size (3)?  ``utils/calibrator.py`` caches each layer's inputs and
outputs across the whole calibration set and concatenates them along dim 0.
With mmdet's ``keep_ratio`` resizing every image has a different resolution,
so the per-image token counts differ and the concatenation fails for the patch
embedding, the FFNs and the patch-merging layers.  We therefore pad every
calibration image to one common ``(H, W)`` -- zero padding on the normalized
tensor, which is exactly what mmdet's own ``DetDataPreprocessor`` does when it
batches images of different sizes.  Only landscape images (``w >= h``) are
used so that the ``(800, 1344)`` default is always large enough.
"""

import copy
import glob
import logging
import os
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# mmdet / mmengine entry points (imported lazily)
# ----------------------------------------------------------------------

def _mm_imports():
    try:
        from mmengine.config import Config
        from mmengine.dataset import pseudo_collate
        from mmengine.evaluator import Evaluator
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner
        from mmdet.apis import init_detector
        from mmdet.registry import DATASETS
    except ImportError as e:  # pragma: no cover - depends on user environment
        raise ImportError(
            "The COCO detection pipeline needs mmdetection >= 3.0 "
            "(with mmcv >= 2.0 and mmengine). Original error: %s" % e
        )
    return (Config, pseudo_collate, Evaluator, init_default_scope, Runner,
            init_detector, DATASETS)


# ----------------------------------------------------------------------
# Config / model
# ----------------------------------------------------------------------

def _find_installed_config(config_name: str) -> Optional[str]:
    """Search the mmdet package's own bundled configs for ``config_name``.

    ``mim install mmdet`` copies the repo's ``configs/`` tree into
    ``<site-packages>/mmdet/.mim/configs/`` (or, on some older layouts,
    straight into ``<site-packages>/mmdet/configs/``). Either location makes
    every bare config name mmdet ships (e.g. what ``MODEL_ZOO`` in
    ``run_mixed_precision_det.py`` stores) resolvable without the user having
    to keep track of where ``mim download`` happened to place a copy.
    """
    import mmdet
    pkg_root = os.path.dirname(os.path.abspath(mmdet.__file__))
    for search_root in (os.path.join(pkg_root, '.mim', 'configs'),
                        os.path.join(pkg_root, 'configs')):
        if not os.path.isdir(search_root):
            continue
        matches = glob.glob(os.path.join(search_root, '**', config_name + '.py'),
                            recursive=True)
        if matches:
            return matches[0]
    return None


def resolve_config_path(config_name_or_path: str) -> str:
    """Resolve a bare mmdet config name or a path to an actual ``.py`` file.

    Tries, in order: the string as a literal path, the string + ``.py`` as a
    relative path (covers a config copy ``mim download`` dropped next to a
    checkpoint), then a search through the installed mmdet package's own
    config tree (covers every config mmdet ships, regardless of where it was
    downloaded to, or whether it was downloaded at all).
    """
    if os.path.isfile(config_name_or_path):
        return config_name_or_path
    py_path = config_name_or_path if config_name_or_path.endswith('.py') \
        else config_name_or_path + '.py'
    if os.path.isfile(py_path):
        return py_path
    bare_name = os.path.basename(config_name_or_path)
    if bare_name.endswith('.py'):
        bare_name = bare_name[:-3]
    found = _find_installed_config(bare_name)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"Could not resolve mmdet config '{config_name_or_path}' to a .py "
        "file. Pass --det-config pointing directly at one (mim download also "
        "drops a copy next to the checkpoint, e.g. "
        "./checkpoints/det/<name>.py), or first run "
        f"'mim download mmdet --config {bare_name} --dest <dir>'."
    )


def load_det_config(config_path, data_root=None):
    """Load an mmdet config and (optionally) repoint it at a local COCO root."""
    Config, _, _, init_default_scope, _, _, _ = _mm_imports()
    init_default_scope('mmdet')
    config_path = resolve_config_path(config_path)
    cfg = Config.fromfile(config_path)
    if data_root is not None:
        data_root = data_root.replace('\\', '/')
        if not data_root.endswith('/'):
            data_root += '/'
        cfg.data_root = data_root
        for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            if cfg.get(key) is not None:
                cfg[key].dataset.data_root = data_root
        for key in ('val_evaluator', 'test_evaluator'):
            ev = cfg.get(key)
            if ev is not None and ev.get('ann_file') is not None:
                ev.ann_file = data_root + 'annotations/instances_val2017.json'
    return cfg


def assert_checkpoint_matches(model, checkpoint):
    """Raise unless ``checkpoint`` fully populates ``model``.

    ``init_detector`` only *warns* when keys are missing or shapes disagree,
    leaving those tensors randomly initialised.  The model then loads, runs,
    and produces plausible-looking but wrong numbers -- the worst failure mode
    there is, because nothing downstream complains.  This turns that warning
    into a hard error.

    Extra keys in the checkpoint are reported but tolerated: they cost
    nothing, whereas a missing or mis-shaped tensor means part of the network
    is random.
    """
    from mmengine.runner.checkpoint import _load_checkpoint

    ckpt = _load_checkpoint(checkpoint, map_location='cpu')
    state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}

    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [(k, tuple(expected[k].shape), tuple(state[k].shape))
                  for k in sorted(set(expected) & set(state))
                  if tuple(expected[k].shape) != tuple(state[k].shape)]

    if unexpected:
        logger.warning("  checkpoint has %d key(s) the model does not use "
                       "(harmless), e.g. %s", len(unexpected), unexpected[0])

    if not missing and not mismatched:
        return

    lines = [
        f"Checkpoint does not match the model: {len(missing)} missing key(s), "
        f"{len(mismatched)} shape mismatch(es).",
        "Those tensors would stay RANDOMLY INITIALISED and every metric "
        "computed from this model would be meaningless.",
    ]
    for k in missing[:10]:
        lines.append(f"  missing : {k}")
    if len(missing) > 10:
        lines.append(f"  ... and {len(missing) - 10} more")
    for k, want, got in mismatched[:10]:
        lines.append(f"  shape   : {k}  model={want}  ckpt={got}")
    if len(mismatched) > 10:
        lines.append(f"  ... and {len(mismatched) - 10} more")
    lines.append(
        "This usually means the config's head architecture does not match the "
        "checkpoint's. Pass --allow-checkpoint-mismatch only if you are "
        "certain the difference is harmless.")
    raise RuntimeError("\n".join(lines))


def build_detector(cfg, checkpoint, device, strict=True):
    """Build the detector from ``cfg`` and load ``checkpoint``, in eval mode."""
    _, _, _, _, _, init_detector, _ = _mm_imports()
    model = init_detector(cfg, checkpoint, device=str(device))
    model.eval()
    if strict:
        assert_checkpoint_matches(model, checkpoint)
    return model


def assert_two_stage_with_mask(model):
    """Fail early if the detector is not a Mask R-CNN-style two-stage model."""
    for attr in ('backbone', 'neck', 'rpn_head', 'roi_head'):
        if not hasattr(model, attr):
            raise ValueError(
                f"Expected a two-stage detector with a '{attr}'; got "
                f"{type(model).__name__}. The detection pipeline currently "
                "supports Mask R-CNN / Cascade Mask R-CNN with a Swin backbone."
            )


# ----------------------------------------------------------------------
# Evaluation dataloader + COCO evaluator
# ----------------------------------------------------------------------

def build_val_dataloader(cfg, batch_size=1, num_workers=2):
    """Build the standard COCO ``val2017`` test dataloader."""
    _, _, _, _, Runner, _, _ = _mm_imports()
    loader_cfg = copy.deepcopy(cfg.test_dataloader)
    loader_cfg['batch_size'] = batch_size
    loader_cfg['num_workers'] = num_workers
    loader_cfg['persistent_workers'] = num_workers > 0
    return Runner.build_dataloader(loader_cfg)


def build_evaluator(cfg, dataloader):
    """Build mmdet's ``CocoMetric`` evaluator bound to the val dataset meta."""
    _, _, Evaluator, _, _, _, _ = _mm_imports()
    evaluator = Evaluator(copy.deepcopy(cfg.test_evaluator))
    evaluator.dataset_meta = dataloader.dataset.metainfo
    return evaluator


# ----------------------------------------------------------------------
# Calibration set
# ----------------------------------------------------------------------

def _override_resize_scale(pipeline, scale):
    """Set the ``Resize`` scale of a (copied) test pipeline."""
    pipeline = copy.deepcopy(pipeline)
    for step in pipeline:
        if step.get('type') in ('Resize', 'mmdet.Resize'):
            step['scale'] = tuple(scale)
    return pipeline


def _build_calib_dataset(cfg, split, scale):
    """Build a ``test_mode`` COCO dataset over ``split`` with the test pipeline."""
    _, _, _, _, _, _, DATASETS = _mm_imports()
    ds_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    ds_cfg['test_mode'] = True
    ds_cfg['pipeline'] = _override_resize_scale(ds_cfg['pipeline'], scale)
    if split == 'train':
        ds_cfg['ann_file'] = 'annotations/instances_train2017.json'
        ds_cfg['data_prefix'] = dict(img='train2017/')
    elif split != 'val':
        raise ValueError(f"--calib-split must be 'train' or 'val', got {split!r}")
    return DATASETS.build(ds_cfg)


@torch.no_grad()
def build_calib_batches(
    model,
    cfg,
    num,
    device,
    split='train',
    scale=(1333, 800),
    pad_hw=(800, 1344),
    seed=5,
):
    """Return ``num`` preprocessed, equally-sized calibration batches.

    Each element is ``{'inputs': FloatTensor[1, 3, H, W] (CPU),
    'data_samples': [DetDataSample]}`` -- already normalized by the detector's
    own ``data_preprocessor`` and zero-padded to ``pad_hw``.

    Images are drawn from a seeded permutation of ``split`` and filtered to
    landscape orientation so that ``pad_hw`` always fits; a portrait image
    resized with ``keep_ratio`` would be up to 1333 px tall.
    """
    _, pseudo_collate, _, _, _, _, _ = _mm_imports()
    from torch.utils.data import DataLoader, Subset

    pad_h, pad_w = int(pad_hw[0]), int(pad_hw[1])
    dataset = _build_calib_dataset(cfg, split, scale)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(dataset))
    picked: List[int] = []
    for idx in order:
        info = dataset.get_data_info(int(idx))
        if info.get('width', 0) >= info.get('height', 0):
            picked.append(int(idx))
        if len(picked) == num:
            break
    if len(picked) < num:
        raise RuntimeError(
            f"Only found {len(picked)} usable calibration images in '{split}'.")
    logger.info("  Calibration images (%s split, seed %d): %s",
                split, seed, picked)

    loader = DataLoader(Subset(dataset, picked), batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=pseudo_collate)

    batches = []
    for data_batch in loader:
        data = model.data_preprocessor(data_batch, False)
        inputs = data['inputs']
        _, _, h, w = inputs.shape
        if h > pad_h or w > pad_w:
            raise RuntimeError(
                f"Calibration image of size {h}x{w} does not fit the fixed "
                f"padding {pad_h}x{pad_w}; raise --calib-pad-hw.")
        padded = inputs.new_zeros((inputs.shape[0], inputs.shape[1], pad_h, pad_w))
        padded[:, :, :h, :w] = inputs
        for sample in data['data_samples']:
            sample.set_metainfo(dict(batch_input_shape=(pad_h, pad_w)))
        batches.append({
            'inputs': padded.cpu(),
            'data_samples': [s.cpu() for s in data['data_samples']],
        })

    logger.info("  Built %d calibration batches at %dx%d (resize scale %s)",
                len(batches), pad_h, pad_w, tuple(scale))
    return batches


def batch_to_device(batch, device):
    """Move one calibration batch onto ``device``."""
    return {
        'inputs': batch['inputs'].to(device),
        'data_samples': [s.to(device) for s in batch['data_samples']],
    }
