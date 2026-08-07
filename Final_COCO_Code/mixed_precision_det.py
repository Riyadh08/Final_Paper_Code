"""Mixed-Precision Post-Training Quantization for COCO detection / segmentation.

This is the detection counterpart of ``mixed_precision.py``.  The *method* is
unchanged -- per-component forward-KL fragility (Omega), an MCKP dynamic
program over the bit budget, and one in-context refinement round -- and the
quantization backend is still the complete AdaLog implementation in
``quant_layers/`` + ``quantizers/``.  Only two things are task-specific:

1. **Where the components live.**  The quantized encoder is the MMDetection
   Swin backbone (``utils/wrap_net_det.py``); the FPN neck, RPN and the ROI /
   mask heads stay full precision, which is the convention in the ViT-PTQ
   detection literature and keeps the reported AP comparable.

2. **What Omega measures.**  Instead of the classifier posterior, the drift is
   measured on the detector's task-level output distributions -- dense RPN
   objectness, the ROI class posterior and the per-pixel mask posterior --
   evaluated under full-precision proposals so the distributions being
   compared are aligned (``utils/det_probe.py``).

Everything else (the DP table, the MCKP solver, the marginal in-context
refinement, the compression report) is imported from the ImageNet pipeline, so
both experiment tables are produced by literally the same allocator.
"""

import copy
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from dp_solver import solve_mckp
from mixed_precision import (
    ComponentSpec,
    build_dp_table,
    _enforce_monotonic,
    compute_compression_report,
    log_compression_report,
)
from utils.det_calibrator import DetQuantCalibrator
from utils.det_probe import DetectionDriftProbe
from utils.wrap_net import get_module_by_name, set_module_by_name
from utils.wrap_net_det import (
    extract_block_index_det,
    finish_bias_reparam,
    identify_component_type_det,
    inject_matmuls_det,
    make_quant_module_det,
    reparam_layernorm_for_det,
    wrap_backbone_in_net,
    wrap_reparamed_modules_in_backbone,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Step 1 -- Component extraction
# ======================================================================

def extract_components_det(backbone: nn.Module) -> List[ComponentSpec]:
    """Enumerate the quantizable Swin-backbone components, in tree order.

    One component per QKV projection, attention output projection, FFN FC1,
    FFN FC2 and patch-merging reduction, plus the patch-embedding Conv2d.
    Names are relative to the backbone.  The ordering is deterministic and must
    stay fixed -- the DP solver indexes into it.
    """
    components: List[ComponentSpec] = []
    for name, module in backbone.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        ctype = identify_component_type_det(name, module)
        if ctype is None:
            continue
        components.append(ComponentSpec(
            component_id=len(components),
            name=name,
            component_type=ctype,
            block_index=extract_block_index_det(name),
            param_count=module.weight.numel(),
            weight_shape=tuple(module.weight.shape),
            module_type=type(module).__name__,
        ))
    return components


def prepare_backend_backbone(backbone: nn.Module, device: torch.device) -> nn.Module:
    """Full-precision working copy of the backbone with explicit MatMuls.

    Numerically equivalent to the input (matmul injection is FP-equivalent and
    is verified against mmdet's own ``WindowMSA.forward`` on the way in).
    """
    if backbone is None:
        # Most likely cause: a caller wrote `x = something.eval()`. mmdet's
        # SwinTransformer.train() does not return self, so .eval() is None.
        raise ValueError(
            "prepare_backend_backbone() got None instead of a backbone. Note "
            "that .eval() on an mmdet Swin backbone returns None -- call it as "
            "a statement, never chained into an assignment.")
    working = copy.deepcopy(backbone)
    inject_matmuls_det(working)
    working.to(device)
    working.eval()
    return working


# ======================================================================
# Steps 3-4 -- Per-component fragility Omega
# ======================================================================

@contextmanager
def temporary_adalog_component_det(
    backbone: nn.Module,
    spec: ComponentSpec,
    bit: int,
    cfg,
    calib_batches,
    device: torch.device,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
):
    """Temporarily quantize **only** ``spec`` with the full AdaLog pipeline.

    Mirrors ``mixed_precision.temporary_adalog_component``: install the AdaLog
    module for this one component at ``w_bit = a_bit = bit``, calibrate just
    that module (everything else in the backbone is plain FP and is skipped),
    then restore the original module and roll back any LayerNorm the
    channel-wise reparameterization rewrote.
    """
    name = spec.name
    orig_module = get_module_by_name(backbone, name)

    ln = reparam_layernorm_for_det(backbone, name) if reparam else None
    ln_backup = None
    if ln is not None:
        ln_backup = (ln.weight.data.clone(),
                     ln.bias.data.clone() if ln.bias is not None else None)

    quant_module = make_quant_module_det(
        backbone, name, orig_module, cfg, reparam=reparam,
        bit_assignment={name: bit},
    )
    quant_module.to(orig_module.weight.device)
    set_module_by_name(backbone, name, quant_module)

    try:
        DetQuantCalibrator(backbone, calib_batches, device,
                           chunk_elems=chunk_elems).batching_quant_calib()
        yield backbone
    finally:
        set_module_by_name(backbone, name, orig_module)
        if ln is not None and ln_backup is not None:
            ln.weight.data.copy_(ln_backup[0])
            if ln_backup[1] is not None:
                ln.bias.data.copy_(ln_backup[1])


def compute_all_omegas_det(
    model,
    backbone: nn.Module,
    components: List[ComponentSpec],
    candidate_bits: List[int],
    cfg,
    probe: DetectionDriftProbe,
    calib_batches,
    device: torch.device,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
) -> Dict[int, Dict[int, float]]:
    """Compute ``omega[component_id][bit]`` for every (component, bit) pair.

    ``backbone`` must be ``model.backbone`` -- the component is swapped inside
    it and the drift is then read off the whole detector.
    """
    omega: Dict[int, Dict[int, float]] = {}
    total = len(components) * len(candidate_bits)
    with tqdm(total=total, desc="Computing Omega") as pbar:
        for comp in components:
            omega[comp.component_id] = {}
            for bit in candidate_bits:
                pbar.set_description(f"Omega({comp.name}, {bit}b)")
                with temporary_adalog_component_det(
                        backbone, comp, bit, cfg, calib_batches, device,
                        reparam=reparam, chunk_elems=chunk_elems):
                    val = probe.drift(model)
                omega[comp.component_id][bit] = val
                pbar.update(1)
                logger.info("Omega(%s, %d-bit) = %.6e   [rpn %.3e | roi %.3e | mask %.3e]",
                            comp.name, bit, val, probe.last_terms['rpn'],
                            probe.last_terms['roi'], probe.last_terms['mask'])
    return omega


# ======================================================================
# Step 7 -- Final mixed-precision backbone
# ======================================================================

def build_final_backbone(
    fp_backbone: nn.Module,
    components: List[ComponentSpec],
    assigned_bits: Dict[int, int],
    cfg,
    calib_batches,
    device: torch.device,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
) -> nn.Module:
    """Build the genuine mixed-precision AdaLog backbone.

    Wraps a fresh copy of the FP backbone at the assigned per-component bits,
    runs the standard full AdaLog calibration once (so every reparameterization
    happens jointly), folds the reparameterized Linears back and applies the
    post-GELU bias reparameterization.  Attention matmuls are quantized
    globally at ``cfg.a_bit`` / ``cfg.s_bit``.
    """
    name_to_bit = {c.name: assigned_bits[c.component_id] for c in components}

    final = copy.deepcopy(fp_backbone)
    final.to(device)
    final.eval()
    wrap_backbone_in_net(final, cfg, reparam=reparam, bit_assignment=name_to_bit)
    final.to(device)

    DetQuantCalibrator(final, calib_batches, device,
                       chunk_elems=chunk_elems).batching_quant_calib()
    final = wrap_reparamed_modules_in_backbone(final)
    final.to(device)
    finish_bias_reparam(final)
    return final


# ======================================================================
# Step 8 -- Refinement (single in-context feedback round)
# ======================================================================

def _build_quant_background_det(
    fp_backbone: nn.Module,
    components: List[ComponentSpec],
    assigned_bits: Dict[int, int],
    cfg,
    calib_batches,
    device: torch.device,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
) -> nn.Module:
    """Fully-quantized backbone left in a *swappable* state.

    Identical to :func:`build_final_backbone` up to but excluding the folding
    and bias-reparameterization steps, so qkv / fc1 / reduction keep their
    ``prev_layer`` pointer and a single component can be re-quantized against
    this background later.
    """
    name_to_bit = {c.name: assigned_bits[c.component_id] for c in components}
    bg = copy.deepcopy(fp_backbone)
    bg.to(device)
    bg.eval()
    wrap_backbone_in_net(bg, cfg, reparam=reparam, bit_assignment=name_to_bit)
    bg.to(device)
    DetQuantCalibrator(bg, calib_batches, device,
                       chunk_elems=chunk_elems).batching_quant_calib()
    bg.to(device)
    return bg


@contextmanager
def temporary_requant_in_background_det(
    bg: nn.Module,
    fp_source: nn.Module,
    spec: ComponentSpec,
    bit: int,
    cfg,
    calib_batches,
    device: torch.device,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
):
    """Re-quantize only ``spec`` at ``bit`` inside an already-quantized backbone.

    The surrounding backbone is quantized at the current assignment, so the
    measured fragility includes inter-layer quantization interactions.  The
    component's LayerNorm is reset to full precision before the fresh reparam
    (otherwise the reparam would compound on top of the existing one) and the
    weights are sourced from ``fp_source`` rather than from the background
    module, whose weights its own reparam has already rewritten.
    """
    name = spec.name
    bg_module = get_module_by_name(bg, name)
    fp_module = get_module_by_name(fp_source, name)

    ln = reparam_layernorm_for_det(bg, name) if reparam else None
    fp_ln = reparam_layernorm_for_det(fp_source, name) if reparam else None
    ln_backup = None
    if ln is not None:
        ln_backup = (ln.weight.data.clone(),
                     ln.bias.data.clone() if ln.bias is not None else None)
        ln.weight.data.copy_(fp_ln.weight.data)
        if ln.bias is not None and fp_ln.bias is not None:
            ln.bias.data.copy_(fp_ln.bias.data)

    fresh = make_quant_module_det(
        bg, name, fp_module, cfg, reparam=reparam, bit_assignment={name: bit})
    fresh.to(next(bg.parameters()).device)
    set_module_by_name(bg, name, fresh)

    try:
        DetQuantCalibrator(bg, calib_batches, device,
                           chunk_elems=chunk_elems).batching_quant_calib()
        yield bg
    finally:
        set_module_by_name(bg, name, bg_module)
        if ln is not None and ln_backup is not None:
            ln.weight.data.copy_(ln_backup[0])
            if ln_backup[1] is not None:
                ln.bias.data.copy_(ln_backup[1])


@contextmanager
def temporary_fp_in_background_det(
    bg: nn.Module,
    fp_source: nn.Module,
    spec: ComponentSpec,
    reparam: bool = True,
):
    """Temporarily restore ``spec`` to full precision inside the quantized bg.

    Used to measure the background baseline -- the drift of the whole quantized
    backbone with *this one* component de-quantized -- so it can be subtracted
    off, leaving the component's marginal in-context damage on the same scale
    as the isolated first-pass Omega.
    """
    name = spec.name
    bg_module = get_module_by_name(bg, name)
    fp_module = get_module_by_name(fp_source, name)

    ln = reparam_layernorm_for_det(bg, name) if reparam else None
    fp_ln = reparam_layernorm_for_det(fp_source, name) if reparam else None
    ln_backup = None
    if ln is not None:
        ln_backup = (ln.weight.data.clone(),
                     ln.bias.data.clone() if ln.bias is not None else None)
        ln.weight.data.copy_(fp_ln.weight.data)
        if ln.bias is not None and fp_ln.bias is not None:
            ln.bias.data.copy_(fp_ln.bias.data)

    fp_clone = copy.deepcopy(fp_module).to(next(bg.parameters()).device)
    set_module_by_name(bg, name, fp_clone)
    try:
        yield bg
    finally:
        set_module_by_name(bg, name, bg_module)
        if ln is not None and ln_backup is not None:
            ln.weight.data.copy_(ln_backup[0])
            if ln_backup[1] is not None:
                ln.bias.data.copy_(ln_backup[1])


def compute_refined_omega_marginal_det(
    model,
    bg: nn.Module,
    fp_source: nn.Module,
    spec: ComponentSpec,
    candidate_bits: List[int],
    cfg,
    probe: DetectionDriftProbe,
    calib_batches,
    device: torch.device,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
) -> Dict[int, float]:
    """Marginal in-context Omega of ``spec`` across all candidate bits.

    ``omega(bit) = max(0, drift(bg[spec=bit]) - drift(bg[spec=FP]))`` -- the
    subtracted baseline is the damage from every *other* quantized component,
    so what remains is comparable with the first-pass isolated Omega and the
    refined and un-refined entries can share one DP table.
    """
    candidate_bits = sorted(candidate_bits)

    with temporary_fp_in_background_det(bg, fp_source, spec, reparam=reparam):
        base = probe.drift(model)

    out: Dict[int, float] = {}
    for bit in candidate_bits:
        with temporary_requant_in_background_det(
                bg, fp_source, spec, bit, cfg, calib_batches, device,
                reparam=reparam, chunk_elems=chunk_elems):
            out[bit] = max(0.0, probe.drift(model) - base)
    return out


def refinement_pass_det(
    model,
    fp_backbone: nn.Module,
    fp_source: nn.Module,
    components: List[ComponentSpec],
    omega: Dict[int, Dict[int, float]],
    assigned_bits: Dict[int, int],
    candidate_bits: List[int],
    cfg,
    probe: DetectionDriftProbe,
    calib_batches,
    device: torch.device,
    budget: int,
    top_k: int,
    cost_scale: Optional[int] = None,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
) -> Tuple[Dict[int, int], Dict[int, Dict[int, float]], List[int]]:
    """Single feedback refinement round, in the fully-quantized context.

    Same three moves as the ImageNet version: build a quantized background at
    the current assignment, re-measure the ``top_k`` most sensitive components
    marginally inside it (clamped monotonic in bit-width to discard noise), and
    re-solve the MCKP with the refined table.

    ``model.backbone`` is swapped to the background for the duration and
    restored afterwards.
    """
    candidate_bits = sorted(candidate_bits)

    logger.info("  Building quantized background for refinement ...")
    bg = _build_quant_background_det(
        fp_backbone, components, assigned_bits, cfg, calib_batches, device,
        reparam=reparam, chunk_elems=chunk_elems)

    ranked = sorted(components,
                    key=lambda c: omega[c.component_id][assigned_bits[c.component_id]],
                    reverse=True)
    top = ranked[:max(0, top_k)]
    logger.info("  Re-measuring %d most-sensitive components in context:", len(top))
    for c in top:
        logger.info("    [%2d] %-52s (assigned %d-bit, Omega=%.3e)",
                    c.component_id, c.name, assigned_bits[c.component_id],
                    omega[c.component_id][assigned_bits[c.component_id]])

    refined: Dict[int, Dict[int, float]] = {
        c.component_id: dict(omega[c.component_id]) for c in components
    }

    saved_backbone = model.backbone
    model.backbone = bg
    try:
        with tqdm(total=len(top), desc="Refining Omega") as pbar:
            for c in top:
                pbar.set_description(f"Refine Omega({c.name})")
                vals = compute_refined_omega_marginal_det(
                    model, bg, fp_source, c, candidate_bits, cfg, probe,
                    calib_batches, device, reparam=reparam,
                    chunk_elems=chunk_elems)
                vals = _enforce_monotonic(vals, candidate_bits)
                for bit in candidate_bits:
                    logger.info("    refined Omega(%s, %d-bit) = %.6e  (was %.6e)",
                                c.name, bit, vals[bit], omega[c.component_id][bit])
                    refined[c.component_id][bit] = vals[bit]
                pbar.update(1)
    finally:
        model.backbone = saved_backbone

    new_table = build_dp_table(components, refined, candidate_bits)
    new_assigned = solve_mckp(new_table, budget, cost_scale)
    return new_assigned, refined, [c.component_id for c in top]


# ======================================================================
# Full pipeline orchestrator
# ======================================================================

def run_mixed_precision_det_pipeline(
    model,
    cfg,
    calib_batches,
    device: torch.device,
    candidate_bits: Optional[List[int]] = None,
    target_avg_bit: float = 4.0,
    probe_kwargs: Optional[dict] = None,
    cost_scale: Optional[int] = None,
    refine: bool = True,
    refine_top_k: Optional[int] = None,
    reparam: bool = True,
    chunk_elems: int = 4_000_000,
) -> Tuple[nn.Module, Dict[int, int], dict]:
    """End-to-end mixed-precision AdaLog quantization of a detector backbone.

    ``model`` is mutated in place: on return ``model.backbone`` is the
    calibrated mixed-precision backbone and the detector is ready to evaluate.

    Returns ``(model, assigned_bits, info)``.
    """
    if candidate_bits is None:
        candidate_bits = [3, 4, 5]
    candidate_bits = sorted(candidate_bits)
    probe_kwargs = probe_kwargs or {}

    # ---- FP references: a pristine backbone copy and a working copy ----
    logger.info("Preparing AdaLog backend backbone ...")
    # NB: do not chain .to(device).eval() here. mmdet's SwinTransformer
    # overrides train() to re-freeze stages and does not return self, so
    # .eval() -- which is just train(False) -- evaluates to None.
    fp_backbone = copy.deepcopy(model.backbone)   # never mutated
    fp_backbone.to(device)
    fp_backbone.eval()
    working = prepare_backend_backbone(fp_backbone, device)
    model.backbone = working

    # ---- Step 1: components ----
    logger.info("Step 1: Extracting quantizable backbone components ...")
    components = extract_components_det(working)
    if not components:
        raise RuntimeError(
            "No quantizable components found. The detection pipeline expects "
            "an mmdet Swin backbone (stages/blocks/attn.w_msa/ffn).")
    total_params = sum(c.param_count for c in components)
    logger.info("  %d components, %s total weight params",
                len(components), f"{total_params:,}")
    for c in components:
        logger.info("  [%2d] %-52s %-8s params=%12s  block=%s  type=%s",
                    c.component_id, c.name, c.module_type,
                    f"{c.param_count:,}", c.block_index, c.component_type)

    # ---- Step 2: full-precision detector outputs (+ proposal alignment) ----
    logger.info("Step 2: Caching full-precision detector outputs ...")
    probe = DetectionDriftProbe(model, calib_batches, device, **probe_kwargs)
    probe.cache_reference()

    # ---- Steps 3-4: fragility Omega ----
    logger.info("Steps 3-4: Computing quantization fragility Omega ...")
    omega = compute_all_omegas_det(
        model, working, components, candidate_bits, cfg, probe,
        calib_batches, device, reparam=reparam, chunk_elems=chunk_elems)

    # ---- Step 5: DP table ----
    logger.info("Step 5: Building DP cost/value table ...")
    dp_table = build_dp_table(components, omega, candidate_bits)
    budget = int(total_params * target_avg_bit)
    logger.info("  Budget = %s  (target %.2f-bit avg)", f"{budget:,}", target_avg_bit)

    # ---- Step 6: MCKP ----
    logger.info("Step 6: Solving Multiple Choice Knapsack ...")
    assigned_bits = solve_mckp(dp_table, budget, cost_scale)
    logger.info("  Initial bit assignments:")
    for c in components:
        logger.info("    %-52s -> %d-bit", c.name, assigned_bits[c.component_id])

    # ---- Step 6.5: refinement ----
    assigned_bits_pre_refine = dict(assigned_bits)
    refined_omega, refined_ids = None, []
    if refine and len(candidate_bits) > 1:
        if refine_top_k is None:
            refine_top_k = max(1, len(components) // 4)
        logger.info("Step 6.5: Refinement pass (top-%d sensitive components) ...",
                    refine_top_k)
        assigned_bits, refined_omega, refined_ids = refinement_pass_det(
            model=model, fp_backbone=fp_backbone, fp_source=fp_backbone,
            components=components, omega=omega, assigned_bits=assigned_bits,
            candidate_bits=candidate_bits, cfg=cfg, probe=probe,
            calib_batches=calib_batches, device=device, budget=budget,
            top_k=refine_top_k, cost_scale=cost_scale, reparam=reparam,
            chunk_elems=chunk_elems)
        changed = {cid: (assigned_bits_pre_refine[cid], assigned_bits[cid])
                   for cid in assigned_bits
                   if assigned_bits_pre_refine[cid] != assigned_bits[cid]}
        if changed:
            name_by_id = {c.component_id: c.name for c in components}
            logger.info("  Refinement changed %d assignment(s):", len(changed))
            for cid, (old, new) in changed.items():
                logger.info("    %-52s %d-bit -> %d-bit", name_by_id[cid], old, new)
        else:
            logger.info("  Refinement left all assignments unchanged.")
    else:
        logger.info("Step 6.5: Refinement pass skipped.")

    actual_cost = sum(c.param_count * assigned_bits[c.component_id] for c in components)
    actual_avg = actual_cost / total_params
    logger.info("  Final bit assignments:")
    for c in components:
        logger.info("    %-52s -> %d-bit", c.name, assigned_bits[c.component_id])
    logger.info("  Actual average: %.4f bits", actual_avg)

    # ---- Step 7: final mixed-precision backbone ----
    logger.info("Step 7: Building + calibrating final mixed-precision backbone ...")
    final_backbone = build_final_backbone(
        fp_backbone, components, assigned_bits, cfg, calib_batches, device,
        reparam=reparam, chunk_elems=chunk_elems)
    model.backbone = final_backbone
    model.to(device).eval()

    info = {
        'components': components,
        'omega': omega,
        'assigned_bits': assigned_bits,
        'assigned_bits_pre_refine': assigned_bits_pre_refine,
        'refined': bool(refine and len(candidate_bits) > 1),
        'refined_omega': refined_omega,
        'refined_component_ids': refined_ids,
        'dp_table': dp_table,
        'fp_backbone': fp_backbone,
        'total_params': total_params,
        'budget': budget,
        'actual_cost': actual_cost,
        'actual_avg_bit': actual_avg,
    }
    logger.info("Pipeline complete.")
    return model, assigned_bits, info


# ======================================================================
# Compression report
# ======================================================================

def log_detection_compression_report(model, fp_backbone, info, metrics=None):
    """Backbone compression report plus a whole-detector size line.

    The backbone table comes from the shared ``compute_compression_report``;
    the extra lines account for the FP32 neck and heads, which dominate the
    residual size once the encoder is at ~4 bits.
    """
    report = compute_compression_report(
        fp_backbone, info['components'], info['assigned_bits'])
    log_compression_report(report)

    MB = 1024 * 1024
    detector_params = sum(p.numel() for p in model.parameters())
    backbone_params = report['total_params']
    rest_params = detector_params - backbone_params
    fp_detector_mb = detector_params * 4 / MB
    mp_detector_mb = report['mp_full_mb'] + rest_params * 4 / MB

    logger.info("Full detector (FP32 neck + RPN + ROI/mask heads):")
    logger.info("  Backbone params    : %s", f"{backbone_params:,}")
    logger.info("  Neck + head params : %s", f"{rest_params:,}")
    logger.info("  FP32 detector      : %8.2f MB", fp_detector_mb)
    logger.info("  Mixed-prec detector: %8.2f MB", mp_detector_mb)
    logger.info("  Detector compression: %7.2fx", fp_detector_mb / mp_detector_mb)
    if metrics:
        logger.info("-" * 64)
        for k, v in metrics.items():
            logger.info("  %-30s : %s", k, v)
    logger.info("=" * 64)
    return report
