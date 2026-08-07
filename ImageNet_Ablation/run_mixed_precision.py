"""
Entry point for the mixed-precision AdaLog PTQ pipeline.

Loads a full-precision timm ViT, runs the KL-fragility + DP bit allocation
(backed by the complete AdaLog quantizer), builds the final mixed-precision
model, and validates it on ImageNet.

Example
-------
    python run_mixed_precision.py --model deit_tiny \
        --config ./configs/3bit.py --dataset /dataset/imagenet/ \
        --target-avg-bit 3.0 --candidate-bits 2 3 4
"""

import os
import sys
import time
import copy
import random
import argparse
import importlib
import logging
from datetime import datetime

# Required for deterministic cuBLAS GEMMs under use_deterministic_algorithms.
# Must be set before the first CUDA/cuBLAS call, so set it at import time.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch import nn
import numpy as np
import timm

import utils.datasets as mydatasets
from utils.test_utils import validate
from mixed_precision import (
    run_mixed_precision_pipeline,
    compute_compression_report,
    log_compression_report,
)


MODEL_ZOO = {
    'vit_tiny':   'vit_tiny_patch16_224',
    'vit_small':  'vit_small_patch16_224',
    'vit_base':   'vit_base_patch16_224',
    'vit_large':  'vit_large_patch16_224',
    'deit_tiny':  'deit_tiny_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'deit_base':  'deit_base_patch16_224',
    'swin_tiny':  'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base':  'swin_base_patch4_window7_224',
    'swin_base_384': 'swin_base_patch4_window12_384',
}


def get_args_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", default="deit_tiny", choices=list(MODEL_ZOO.keys()))
    p.add_argument('--config', type=str, default="./configs/3bit.py",
                   help="AdaLog backend config (Config class).")
    p.add_argument('--dataset', default="/dataset/imagenet/", help='path to dataset')
    p.add_argument("--calib-size", default=32, type=int, help="calibration set size")
    p.add_argument("--calib-batch-size", default=32, type=int)
    p.add_argument("--val-batch-size", default=200, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=5, type=int)
    # mixed-precision specific
    p.add_argument("--candidate-bits", type=int, nargs='+', default=[2, 3, 4],
                   help="per-component bit options")
    p.add_argument("--target-avg-bit", type=float, default=3.0,
                   help="target average weight bit-width")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax temperature for KL fragility")
    p.add_argument("--allocation-strategy", type=str, default="mckp",
                   choices=["mckp", "greedy", "random"],
                   help="bit allocation strategy for the ablation study")
    p.add_argument("--no-refine", action="store_true",
                   help="disable the LAMPQ-inspired in-context refinement round")
    p.add_argument("--refine-top-k", type=int, default=None,
                   help="number of most-sensitive components re-measured in "
                        "context during refinement (default: components // 4)")
    p.add_argument("--print-freq", default=10, type=int)
    return p


def seed_all(seed):
    """Seed every RNG and force deterministic kernels for reproducible runs.

    ``warn_only=True`` keeps the run alive if some backend op lacks a
    deterministic implementation (it warns instead of raising), so this is
    safe to enable unconditionally while still pinning everything that can be
    pinned -- crucially the cuDNN convolution (patch_embed) and cuBLAS GEMMs
    that otherwise vary run-to-run on the GPU.
    """
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


def main(args):
    # ---- logging ----
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    root_path = f'./checkpoints/mixed_precision/{ts}'
    os.makedirs(root_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.FileHandler(f'{root_path}/output.log'),
                  logging.StreamHandler()],
    )
    logging.info("Args: %s", args)

    # ---- config ----
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    Config = getattr(importlib.import_module(module_name), 'Config')
    cfg = Config()
    cfg.calib_size = args.calib_size
    cfg.calib_batch_size = args.calib_batch_size
    for k, v in vars(cfg).items():
        logging.info("cfg.%s = %s", k, v)

    if args.device.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[1]
        args.device = 'cuda:0'
    device = torch.device(args.device)
    seed_all(args.seed)

    # ---- model ----
    logging.info("Building model %s ...", args.model)
    try:
        fp_model = timm.create_model(
            MODEL_ZOO[args.model],
            checkpoint_path=f'./checkpoints/vit_raw/{MODEL_ZOO[args.model]}.bin')
    except Exception:
        fp_model = timm.create_model(MODEL_ZOO[args.model], pretrained=True)
    fp_model.to(device).eval()

    # ---- data ----
    g = mydatasets.ViTImageNetLoaderGenerator(
        args.dataset, args.val_batch_size, args.num_workers, kwargs={"model": fp_model})
    val_loader = g.val_loader()
    calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
    criterion = nn.CrossEntropyLoss().to(device)

    # ---- mixed-precision pipeline ----
    t0 = time.time()
    final_model, assigned_bits, info = run_mixed_precision_pipeline(
        fp_model=fp_model, cfg=cfg, calib_loader=calib_loader, device=device,
        candidate_bits=args.candidate_bits,
        target_avg_bit=args.target_avg_bit,
        temperature=args.temperature,
        allocation_strategy=args.allocation_strategy,
        refine=not args.no_refine,
        refine_top_k=args.refine_top_k,
    )
    logging.info("Pipeline wall-time: %.1f s", time.time() - t0)
    logging.info("Actual average bit-width: %.4f", info['actual_avg_bit'])

    # ---- save + validate ----
    save_path = os.path.join(
        root_path, f"{args.model}_mp_avg{info['actual_avg_bit']:.2f}.pth")
    torch.save(final_model.state_dict(), save_path)
    logging.info("Saved final model to %s", save_path)

    final_model.to(device).eval()
    logging.info("Validating mixed-precision model ...")
    _, top1, _ = validate(val_loader, final_model, criterion,
                          print_freq=args.print_freq, device=device)

    # ---- compression report (runs last, after everything) ----
    report = compute_compression_report(
        fp_model, info['components'], info['assigned_bits'])
    log_compression_report(report, top1=top1)

    # on-disk note: the saved .pth is a simulation checkpoint (FP weights)
    if os.path.exists(save_path):
        disk_mb = os.path.getsize(save_path) / (1024 * 1024)
        logging.info("Saved checkpoint on disk: %.2f MB (simulation: stores FP "
                     "weights + quantizer params, not the packed size above)",
                     disk_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    main(parser.parse_args())
