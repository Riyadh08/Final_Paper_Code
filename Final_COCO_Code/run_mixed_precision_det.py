"""Entry point for mixed-precision AdaLog PTQ on COCO detection / segmentation.

Loads an MMDetection Mask R-CNN with a Swin backbone, runs the KL-fragility +
MCKP bit allocation over the backbone (the same allocator as the ImageNet
pipeline, with the drift measured on the detector's own output distributions),
builds the final mixed-precision model and evaluates box AP / mask AP on COCO
``val2017`` with mmdet's own ``CocoMetric``.

Examples
--------
Full-precision reference row::

    python run_mixed_precision_det.py --model mask_rcnn_swin_t \
        --checkpoint ./checkpoints/det/mask_rcnn_swin_t.pth \
        --data-root D:/coco --eval-fp-only

Uniform W4/A4 AdaLog baseline row::

    python run_mixed_precision_det.py --model mask_rcnn_swin_t \
        --checkpoint ./checkpoints/det/mask_rcnn_swin_t.pth \
        --data-root D:/coco --config ./configs/4bit_det.py --uniform

Mixed precision 4MP/4MP row::

    python run_mixed_precision_det.py --model mask_rcnn_swin_t \
        --checkpoint ./checkpoints/det/mask_rcnn_swin_t.pth \
        --data-root D:/coco --config ./configs/4bit_det.py \
        --candidate-bits 3 4 5 --target-avg-bit 4.0
"""

import argparse
import importlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

# Required for deterministic cuBLAS GEMMs under use_deterministic_algorithms.
# Must be set before the first CUDA/cuBLAS call, so set it at import time.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from mixed_precision_det import (
    build_final_backbone,
    extract_components_det,
    log_detection_compression_report,
    prepare_backend_backbone,
    run_mixed_precision_det_pipeline,
)
from utils.coco_data import (
    assert_two_stage_with_mask,
    build_calib_batches,
    build_detector,
    build_evaluator,
    build_val_dataloader,
    load_det_config,
)
from make_det_table import format_run_summary


# Short name -> mmdet config name.  Fetch config + checkpoint with, e.g.:
#   mim download mmdet --config mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco --dest ./checkpoints/det
#
# NOTE on which schedule to use.  The published ViT-PTQ detection tables quote
# full-precision Mask R-CNN Swin-T at 46.0 box / 41.6 mask AP, which is the
# **3x multi-scale-crop** schedule.  The 1x checkpoint is a different (weaker)
# model at ~42.7 / 39.3, so rows produced with it are not comparable to those
# tables.  Use the ``*_3x`` entries for anything that goes in the paper.
MODEL_ZOO = {
    # --- comparable to the published tables ---
    'mask_rcnn_swin_t_3x': 'mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco',
    'mask_rcnn_swin_s_3x': 'mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco',
    # Cascade Mask R-CNN is not in mmdet 3.x's zoo; these are local configs
    # paired with a checkpoint converted by tools/convert_swin_det_ckpt.py.
    'cascade_mask_rcnn_swin_t_3x':
        './configs/det/cascade-mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py',
    'cascade_mask_rcnn_swin_s_3x':
        './configs/det/cascade-mask-rcnn_swin-s-p4-w7_fpn_ms-crop-3x_coco.py',
    # --- quick smoke-test model, NOT for the paper table ---
    'mask_rcnn_swin_t':    'mask-rcnn_swin-t-p4-w7_fpn_1x_coco',
}


def precision_label(args, cfg=None):
    """The 'Prec. (W/A)' cell for this run: 32/32, 4/4, or MP4/MP4."""
    if args.eval_fp_only:
        return '32/32'
    if args.uniform:
        return f'{cfg.w_bit}/{cfg.a_bit}'
    target = args.target_avg_bit
    target_str = f'{target:g}'
    return f'MP{target_str}/MP{target_str}'


def get_args_parser():
    p = argparse.ArgumentParser(add_help=False)
    # ---- model / data ----
    p.add_argument("--model", default="mask_rcnn_swin_t", choices=list(MODEL_ZOO.keys()),
                   help="shorthand for a bundled mmdet config name")
    p.add_argument("--det-config", type=str, default=None,
                   help="explicit path to an mmdet config .py (overrides --model)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="path to the pretrained detector checkpoint")
    p.add_argument("--data-root", type=str, default=None,
                   help="COCO root containing train2017/, val2017/, annotations/")
    p.add_argument("--config", type=str, default="./configs/4bit_det.py",
                   help="AdaLog backend config (Config class)")
    # ---- calibration ----
    p.add_argument("--calib-size", default=4, type=int,
                   help="number of calibration images")
    p.add_argument("--calib-split", default="train", choices=["train", "val"])
    p.add_argument("--calib-scale", type=int, nargs=2, default=[1333, 800],
                   metavar=('W', 'H'), help="keep-ratio resize scale for calibration")
    p.add_argument("--calib-pad-hw", type=int, nargs=2, default=[800, 1344],
                   metavar=('H', 'W'), help="fixed padded calibration input size")
    p.add_argument("--chunk-elems", type=int, default=4_000_000,
                   help="activation elements per search chunk (lower on small GPUs)")
    # ---- mixed precision ----
    p.add_argument("--candidate-bits", type=int, nargs='+', default=[3, 4, 5])
    p.add_argument("--target-avg-bit", type=float, default=4.0)
    p.add_argument("--no-refine", action="store_true",
                   help="disable the in-context refinement round")
    p.add_argument("--refine-top-k", type=int, default=None)
    p.add_argument("--no-reparam", action="store_true",
                   help="disable the LayerNorm channel->layer reparameterization")
    p.add_argument("--uniform", action="store_true",
                   help="skip the bit search; quantize uniformly at cfg.w_bit/a_bit "
                        "(the plain-AdaLog baseline row)")
    p.add_argument("--eval-fp-only", action="store_true",
                   help="evaluate the full-precision detector and exit")
    p.add_argument("--allow-checkpoint-mismatch", action="store_true",
                   help="proceed even if the checkpoint leaves some model "
                        "weights randomly initialised (almost never right)")
    # ---- drift probe ----
    p.add_argument("--drift-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   metavar=('RPN', 'ROI', 'MASK'),
                   help="weights of the RPN / ROI / mask KL terms in Omega")
    p.add_argument("--drift-temperature", type=float, default=1.0)
    p.add_argument("--n-proposals", type=int, default=100,
                   help="full-precision proposals per image used to align the probe")
    p.add_argument("--no-mask-drift", action="store_true",
                   help="drop the mask term from Omega (detection-only allocation)")
    # ---- runtime ----
    p.add_argument("--val-batch-size", default=1, type=int)
    p.add_argument("--max-eval-images", type=int, default=None,
                   help="evaluate on only the first N val images (smoke tests; "
                        "AP is then computed over that subset, not the full 5k)")
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=5, type=int)
    p.add_argument("--print-freq", default=200, type=int)
    p.add_argument("--out-dir", default="./checkpoints/mixed_precision_det", type=str)
    return p


def seed_all(seed):
    """Seed every RNG and force deterministic kernels for reproducible runs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:  # pragma: no cover - very old torch
        logging.warning("Could not enable deterministic algorithms: %s", e)


@torch.no_grad()
def evaluate_coco(model, dataloader, evaluator, print_freq=200, max_images=None):
    """Run mmdet's standard COCO evaluation and return the metric dict.

    ``max_images`` truncates the run for smoke tests.  ``CocoMetric`` restricts
    ``COCOeval`` to the image ids it actually saw, so the AP is a valid number
    over that subset -- just not comparable to a full val2017 run.
    """
    model.eval()
    t0 = time.time()
    seen = 0
    total = len(dataloader.dataset) if max_images is None \
        else min(max_images, len(dataloader.dataset))
    for i, data_batch in enumerate(dataloader):
        outputs = model.test_step(data_batch)
        evaluator.process(data_samples=outputs, data_batch=data_batch)
        seen += len(outputs)
        if i % print_freq == 0:
            logging.info("  eval [%d/%d]  %.1f s", seen, total, time.time() - t0)
        if max_images is not None and seen >= max_images:
            logging.info("  stopping early at %d images (--max-eval-images)", seen)
            break
    metrics = evaluator.evaluate(seen)
    logging.info("  evaluation took %.1f s", time.time() - t0)
    for k, v in metrics.items():
        logging.info("  %-24s = %s", k, v)
    return metrics


def save_run_meta(root_path, args, prec, metrics, actual_avg_bit=None,
                  wall_time=None):
    """Write ``run_meta.json`` -- the one file ``make_table.py`` consumes.

    Keeping every field needed to place this run in the results table (model,
    precision cell, whether refinement ran, the achieved average bit-width)
    next to the metrics means the table can be rebuilt from the output tree
    alone, without re-deriving anything from directory names.
    """
    meta = {
        'model': args.model,
        'det_config': args.det_config or MODEL_ZOO.get(args.model),
        'checkpoint': args.checkpoint,
        'precision': prec,
        'mode': 'fp32' if args.eval_fp_only else ('uniform' if args.uniform else 'mp'),
        'quant_config': None if args.eval_fp_only else args.config,
        'candidate_bits': None if args.eval_fp_only or args.uniform else args.candidate_bits,
        'target_avg_bit': None if args.eval_fp_only or args.uniform else args.target_avg_bit,
        'actual_avg_bit': actual_avg_bit,
        'refined': None if args.eval_fp_only or args.uniform else (not args.no_refine),
        'reparam': not args.no_reparam,
        'calib_size': None if args.eval_fp_only else args.calib_size,
        'calib_split': None if args.eval_fp_only else args.calib_split,
        'drift_weights': None if args.eval_fp_only or args.uniform else args.drift_weights,
        'seed': args.seed,
        'max_eval_images': args.max_eval_images,
        'partial_eval': args.max_eval_images is not None,
        'wall_time_s': wall_time,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'metrics': metrics,
    }
    path = os.path.join(root_path, 'run_meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote run metadata to %s", path)
    return meta


def load_quant_config(path, args):
    """Import the ``Config`` class from an AdaLog backend config file."""
    dir_path = os.path.dirname(os.path.abspath(path))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(path))[0]
    Config = getattr(importlib.import_module(module_name), 'Config')
    cfg = Config()
    cfg.calib_size = args.calib_size
    return cfg


def main(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    root_path = os.path.join(args.out_dir, ts)
    os.makedirs(root_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.FileHandler(f'{root_path}/output.log'),
                  logging.StreamHandler()],
    )
    logging.info("Args: %s", args)

    if args.device.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[1]
        args.device = 'cuda:0'
    device = torch.device(args.device)
    seed_all(args.seed)

    # ---- detector ----
    det_config = args.det_config or MODEL_ZOO[args.model]
    logging.info("Loading detector config %s ...", det_config)
    det_cfg = load_det_config(det_config, data_root=args.data_root)
    model = build_detector(det_cfg, args.checkpoint, device,
                           strict=not args.allow_checkpoint_mismatch)
    assert_two_stage_with_mask(model)

    val_loader = build_val_dataloader(det_cfg, args.val_batch_size, args.num_workers)

    if args.eval_fp_only:
        logging.info("Evaluating the full-precision detector ...")
        metrics = evaluate_coco(model, val_loader, build_evaluator(det_cfg, val_loader),
                                args.print_freq, args.max_eval_images)
        with open(os.path.join(root_path, 'metrics_fp32.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        save_run_meta(root_path, args, precision_label(args), metrics)
        logging.info("%s", format_run_summary(
            args.model, precision_label(args), metrics=metrics,
            method='Full-Precision (this run)',
            partial_eval=args.max_eval_images is not None))
        return

    # ---- AdaLog backend config ----
    cfg = load_quant_config(args.config, args)
    for k, v in vars(cfg).items():
        logging.info("cfg.%s = %s", k, v)

    # ---- calibration data ----
    logging.info("Building calibration set ...")
    calib_batches = build_calib_batches(
        model, det_cfg, num=args.calib_size, device=device,
        split=args.calib_split, scale=tuple(args.calib_scale),
        pad_hw=tuple(args.calib_pad_hw), seed=args.seed)

    reparam = not args.no_reparam
    t0 = time.time()
    prec = precision_label(args, cfg)

    if args.uniform:
        # ---- baseline: plain AdaLog at the uniform cfg bit-widths ----
        logging.info("Uniform W%d/A%d AdaLog baseline (no bit search) ...",
                     cfg.w_bit, cfg.a_bit)
        fp_backbone = prepare_backend_backbone(model.backbone, device)
        components = extract_components_det(fp_backbone)
        assigned_bits = {c.component_id: cfg.w_bit for c in components}
        model.backbone = build_final_backbone(
            model.backbone, components, assigned_bits, cfg, calib_batches,
            device, reparam=reparam, chunk_elems=args.chunk_elems)
        model.to(device).eval()
        info = {'components': components, 'assigned_bits': assigned_bits,
                'actual_avg_bit': float(cfg.w_bit)}
    else:
        # ---- mixed precision ----
        model, assigned_bits, info = run_mixed_precision_det_pipeline(
            model=model, cfg=cfg, calib_batches=calib_batches, device=device,
            candidate_bits=args.candidate_bits,
            target_avg_bit=args.target_avg_bit,
            probe_kwargs=dict(
                n_proposals=args.n_proposals,
                weights=tuple(args.drift_weights),
                temperature=args.drift_temperature,
                use_mask=not args.no_mask_drift,
            ),
            refine=not args.no_refine,
            refine_top_k=args.refine_top_k,
            reparam=reparam,
            chunk_elems=args.chunk_elems,
        )
        fp_backbone = info['fp_backbone']

    wall_time = time.time() - t0
    logging.info("Pipeline wall-time: %.1f s", wall_time)
    logging.info("Actual average backbone bit-width: %.4f", info['actual_avg_bit'])

    # ---- save the assignment + the quantized backbone ----
    tag = f"{args.model}_{'uniform' if args.uniform else 'mp'}" \
          f"_avg{info['actual_avg_bit']:.2f}"
    with open(os.path.join(root_path, f'{tag}_bits.json'), 'w') as f:
        json.dump({c.name: info['assigned_bits'][c.component_id]
                   for c in info['components']}, f, indent=2)
    save_path = os.path.join(root_path, f'{tag}_backbone.pth')
    torch.save(model.backbone.state_dict(), save_path)
    logging.info("Saved quantized backbone to %s", save_path)

    # ---- COCO evaluation ----
    logging.info("Evaluating the quantized detector on COCO val2017 ...")
    metrics = evaluate_coco(model, val_loader, build_evaluator(det_cfg, val_loader),
                            args.print_freq, args.max_eval_images)
    with open(os.path.join(root_path, f'{tag}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    save_run_meta(root_path, args, prec, metrics,
                  actual_avg_bit=info['actual_avg_bit'], wall_time=wall_time)

    # ---- compression report (runs last, after everything) ----
    log_detection_compression_report(model, fp_backbone, info, metrics=metrics)

    if os.path.exists(save_path):
        disk_mb = os.path.getsize(save_path) / (1024 * 1024)
        logging.info("Saved checkpoint on disk: %.2f MB (simulation: stores FP "
                     "weights + quantizer params, not the packed size above)",
                     disk_mb)

    # ---- paper-format result row (last thing on screen) ----
    method = 'AdaLog repro (this run)' if args.uniform else 'Ours (this run)'
    if not args.uniform and args.no_refine:
        method = 'Ours, no refine (this run)'
    logging.info("%s", format_run_summary(
        args.model, prec, metrics=metrics, method=method,
        actual_avg_bit=info['actual_avg_bit'],
        partial_eval=args.max_eval_images is not None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    main(parser.parse_args())
