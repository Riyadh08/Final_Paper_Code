"""
Mixed-Precision Post-Training Quantization Pipeline for Vision Transformers.

The quantization **backend is the complete AdaLog method** as implemented in
this repository (``quant_layers/*`` + ``quantizers/logarithm.py`` +
``utils/wrap_net.py`` + ``utils/calibrator.py``).  This module does NOT
reimplement quantization -- it *drives* that backend:

    * weight quantization with FPCS (Fast Progressive Combination Search)
    * adaptive log-base search                (AdaLogQuantizer.q)
    * LayerNorm channel->layer reparameterization (qkv / fc1)
    * post-GELU AdaLog quantization           (fc2)
    * post-Softmax AdaLog quantization        (attn @ v, global)
    * bias reparameterization                 (finish_training)
    * the standard calibration procedure      (QuantCalibrator)

The *bit allocation* is performed independently by a forward KL-divergence
fragility metric (Omega) plus the allocation solvers in ``dp_solver.py``.

Pipeline
--------
1. extract_components()        -- enumerate quantizable sub-layers
2. compute_reference_logits()  -- cache full-precision logits
3-4. compute_all_omegas()      -- per-(component, bit) KL via full AdaLog calib
5. build_dp_table()            -- prepare DP cost/value inputs
6. solve_bit_allocation()      -- MCKP / greedy / random allocation
7. build_final_model()         -- full-model AdaLog calibration at chosen bits
8. refinement_pass()           -- (future) single feedback round

Design notes
------------
* Fragility uses **on-demand per-component calibration**: for each
  (component, bit) only that module is wrapped + calibrated with the full
  AdaLog pipeline; everything else stays full precision; the module (and any
  LayerNorm it reparameterized) is restored afterwards.
* The attention score / probability matmuls (Q.K, post-Softmax attn.V) are a
  **global** backend setting (cfg.a_bit / cfg.s_bit), not DP variables; they
  are quantized only in the final model and stay FP during per-Linear
  fragility so each Linear's contribution is isolated.
"""

import copy
import re
import logging
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.calibrator import QuantCalibrator
from utils.wrap_net import (
    inject_matmuls,
    make_quant_module,
    wrap_modules_in_net,
    wrap_reparamed_modules_in_net,
    get_module_by_name,
    set_module_by_name,
    _parent_name,
)
from dp_solver import DPEntry, solve_bit_allocation

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class ComponentSpec:
    """Specification for one quantizable component in the model."""
    component_id: int          # sequential index (deterministic ordering)
    name: str                  # dotted path (e.g. "blocks.3.mlp.fc1")
    component_type: str        # patch_embed | attn_qkv | attn_proj | mlp_fc1 | mlp_fc2 | head
    block_index: int           # transformer block index (-1 for patch_embed / head)
    param_count: int           # number of weight elements (numel)
    weight_shape: tuple        # e.g. (192, 768) or (192, 3, 16, 16)
    module_type: str           # "Linear" or "Conv2d"


# ======================================================================
# Step 1 -- Component extraction
# ======================================================================

def _identify_component_type(name: str, module: nn.Module) -> Optional[str]:
    """Classify a leaf module as one of the six quantizable types, or None."""
    if 'patch_embed' in name and isinstance(module, nn.Conv2d):
        return 'patch_embed'
    if isinstance(module, nn.Linear):
        if name in ('head', 'head.fc') \
                or name.endswith('.head') or name.endswith('.head.fc'):
            return 'head'
    if name.endswith('.qkv'):
        return 'attn_qkv'
    if name.endswith('.proj') and 'attn' in name and 'patch_embed' not in name:
        return 'attn_proj'
    if name.endswith('.fc1'):
        return 'mlp_fc1'
    if name.endswith('.fc2'):
        return 'mlp_fc2'
    return None


def _extract_block_index(name: str) -> int:
    """Derive a unique block index from the dotted module path.

    * Standard ViT / DeiT:  ``blocks.{i}``  --> i
    * Swin:  ``layers.{s}.blocks.{i}``       --> s*100 + i
    * Non-block modules (patch_embed, head)  --> -1
    """
    stage_m = re.search(r'layers\.(\d+)', name)
    block_m = re.search(r'blocks\.(\d+)', name)
    if block_m is None:
        return -1
    idx = int(block_m.group(1))
    if stage_m is not None:
        idx += int(stage_m.group(1)) * 100
    return idx


def extract_components(model: nn.Module) -> List[ComponentSpec]:
    """Extract every quantizable component at block sub-layer granularity.

    Per transformer block: QKV projection, attention output projection,
    MLP FC1, MLP FC2.  Plus the global patch embedding (Conv2d) and the
    classification head (Linear).

    The ordering is deterministic (``named_modules`` order) and must remain
    fixed -- the DP solver depends on it.
    """
    components: List[ComponentSpec] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        ctype = _identify_component_type(name, module)
        if ctype is None:
            continue
        components.append(ComponentSpec(
            component_id=len(components),
            name=name,
            component_type=ctype,
            block_index=_extract_block_index(name),
            param_count=module.weight.numel(),
            weight_shape=tuple(module.weight.shape),
            module_type=type(module).__name__,
        ))
    return components


# ======================================================================
# Backend model preparation
# ======================================================================

def prepare_backend_model(fp_model: nn.Module, device: torch.device) -> nn.Module:
    """Return a full-precision working copy with explicit MatMul modules.

    The returned model is numerically equivalent to ``fp_model`` (matmul
    injection is FP-equivalent) and provides the architecture against which
    every component will be measured.  ``fp_model`` is left untouched.
    """
    backend = copy.deepcopy(fp_model)
    inject_matmuls(backend)
    backend.to(device)
    backend.eval()
    return backend


# ======================================================================
# Step 2 -- Baseline full-precision logits
# ======================================================================

@torch.no_grad()
def compute_reference_logits(
    model: nn.Module,
    calib_loader,
    device: torch.device,
) -> torch.Tensor:
    """Run calibration data through the FP model and cache the logits on CPU.

    Computed once; the single reference for every KL-divergence computation.
    """
    model.eval()
    parts: List[torch.Tensor] = []
    for inp, _ in calib_loader:
        parts.append(model(inp.to(device)).cpu())
    return torch.cat(parts, dim=0)


# ======================================================================
# Step 3 -- Temporary single-component AdaLog quantization
# ======================================================================

def _reparam_layernorm_for(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Return the LayerNorm that AdaLog's channel-wise reparam will modify
    for component *name* (norm1 for qkv, norm2 for fc1), else None.

    Only qkv / fc1 trigger the RepQ-ViT-style LN reparameterization, so only
    those carry a cross-module side effect that must be restored.
    """
    parent = _parent_name(name)
    grandparent = _parent_name(parent)
    try:
        gp_mod = get_module_by_name(model, grandparent)
    except AttributeError:
        return None
    if name.endswith('.qkv') and hasattr(gp_mod, 'norm1'):
        return gp_mod.norm1
    if name.endswith('.fc1') and hasattr(gp_mod, 'norm2'):
        return gp_mod.norm2
    return None


@contextmanager
def temporary_adalog_component(
    model: nn.Module,
    spec: ComponentSpec,
    bit: int,
    cfg,
    calib_loader,
):
    """Temporarily quantize **only** ``spec`` with the full AdaLog pipeline.

    On enter:
      1. The component is replaced by the appropriate AdaLog quant module
         (channel-wise LN-reparam for qkv/fc1, post-GELU AdaLog for fc2,
         asymmetric uniform otherwise) at ``w_bit = a_bit = bit``.
      2. That single module is calibrated via the standard ``QuantCalibrator``
         (FPCS + adaptive log-base search + LN reparam), then switched to
         ``quant_forward``.  Everything else stays full precision.

    On exit: the original FP module is restored and any LayerNorm that was
    reparameterized is rolled back, so ``model`` returns to full precision.
    """
    name = spec.name
    orig_module = get_module_by_name(model, name)

    # Back up cross-module side effects (LayerNorm affine for qkv / fc1).
    ln = _reparam_layernorm_for(model, name)
    ln_backup = None
    if ln is not None:
        ln_backup = (ln.weight.data.clone(),
                     ln.bias.data.clone() if ln.bias is not None else None)

    # Build + install the AdaLog quant module for this one component.
    quant_module = make_quant_module(
        model, name, orig_module, cfg, reparam=True,
        bit_assignment={name: bit},
    )
    quant_module.to(orig_module.weight.device)
    set_module_by_name(model, name, quant_module)

    try:
        # Calibrate ONLY this module (the rest are plain FP layers, skipped).
        QuantCalibrator(model, calib_loader).batching_quant_calib()
        yield model
    finally:
        set_module_by_name(model, name, orig_module)
        if ln is not None and ln_backup is not None:
            ln.weight.data.copy_(ln_backup[0])
            if ln_backup[1] is not None:
                ln.bias.data.copy_(ln_backup[1])


# ======================================================================
# Step 4 -- Quantization fragility  Omega
# ======================================================================

def _kl_divergence(
    fp_logits: torch.Tensor,
    q_logits: torch.Tensor,
    temperature: float,
) -> float:
    """KL( softmax(fp / T)  ||  softmax(q / T) ), averaged over samples."""
    p = F.softmax(fp_logits / temperature, dim=-1)
    log_q = F.log_softmax(q_logits / temperature, dim=-1)
    return F.kl_div(log_q, p, reduction='batchmean').item()


def compute_component_omega(
    model: nn.Module,
    spec: ComponentSpec,
    bit: int,
    cfg,
    ref_logits: torch.Tensor,
    calib_loader,
    device: torch.device,
    temperature: float = 1.0,
) -> float:
    """Compute  Omega(component, bit)  via the full AdaLog backend.

    Temporarily quantize only this component with AdaLog at *bit*, run forward
    inference on the calibration set, and return the KL divergence against the
    cached full-precision logits.

    No labels, no accuracy, no gradients, no reconstruction.
    """
    with temporary_adalog_component(model, spec, bit, cfg, calib_loader):
        with torch.no_grad():
            parts = [model(inp.to(device)).cpu() for inp, _ in calib_loader]
    q_logits = torch.cat(parts, dim=0)
    return _kl_divergence(ref_logits, q_logits, temperature)


def compute_all_omegas(
    model: nn.Module,
    components: List[ComponentSpec],
    candidate_bits: List[int],
    cfg,
    ref_logits: torch.Tensor,
    calib_loader,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[int, Dict[int, float]]:
    """Compute Omega for every (component, bit) pair.

    Returns ``omega[component_id][bit] = float``.
    """
    omega: Dict[int, Dict[int, float]] = {}
    total = len(components) * len(candidate_bits)
    with tqdm(total=total, desc="Computing Omega") as pbar:
        for comp in components:
            omega[comp.component_id] = {}
            for bit in candidate_bits:
                pbar.set_description(f"Omega({comp.name}, {bit}b)")
                val = compute_component_omega(
                    model, comp, bit, cfg, ref_logits,
                    calib_loader, device, temperature,
                )
                omega[comp.component_id][bit] = val
                pbar.update(1)
                logger.info("Omega(%s, %d-bit) = %.6e", comp.name, bit, val)
    return omega


# ======================================================================
# Step 5 -- DP table construction
# ======================================================================

def build_dp_table(
    components: List[ComponentSpec],
    omega: Dict[int, Dict[int, float]],
    candidate_bits: List[int],
) -> List[List[DPEntry]]:
    """Prepare the cost / value table consumed by the MCKP solver.

    For each component *c* and candidate bit *b*:

    * ``Cost(c, b)  = param_count(c) * b``
    * ``Value(c, b) = Omega(c, min_bit) - Omega(c, b)``

    The solver receives *only* this table.
    """
    min_bit = min(candidate_bits)
    table: List[List[DPEntry]] = []
    for comp in components:
        base_omega = omega[comp.component_id][min_bit]
        group = [
            DPEntry(
                component_id=comp.component_id,
                bit=bit,
                cost=comp.param_count * bit,
                value=base_omega - omega[comp.component_id][bit],
            )
            for bit in candidate_bits
        ]
        table.append(group)
    return table


# ======================================================================
# Step 7 -- Final mixed-precision model (full AdaLog calibration)
# ======================================================================

def _finish_training(model: nn.Module) -> None:
    """Apply AdaLog bias reparameterization to post-GELU layers."""
    for _, module in model.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'reparam_bias'):
            module.reparam_bias()


def build_final_model(
    fp_model: nn.Module,
    components: List[ComponentSpec],
    assigned_bits: Dict[int, int],
    cfg,
    calib_loader,
    device: torch.device,
) -> nn.Module:
    """Build the genuine mixed-precision AdaLog model.

    Wraps a fresh copy of the FP model with per-component bit-widths, runs the
    **standard full AdaLog calibration** once (so all reparameterizations are
    done jointly and correctly), then converts reparam'd layers and applies
    bias reparameterization.  Matmuls are quantized globally at
    ``cfg.a_bit`` / ``cfg.s_bit``.
    """
    name_to_bit = {c.name: assigned_bits[c.component_id] for c in components}

    final = copy.deepcopy(fp_model)
    final.to(device)
    final.eval()
    wrap_modules_in_net(final, cfg, reparam=True, bit_assignment=name_to_bit)
    final.to(device)

    QuantCalibrator(final, calib_loader).batching_quant_calib()
    final = wrap_reparamed_modules_in_net(final)
    final.to(device)
    _finish_training(final)
    return final


# ======================================================================
# Step 8 -- Refinement (single LAMPQ-inspired feedback round)
# ======================================================================

def _build_quant_background(
    fp_model: nn.Module,
    components: List[ComponentSpec],
    assigned_bits: Dict[int, int],
    cfg,
    calib_loader,
    device: torch.device,
) -> nn.Module:
    """Build a fully-quantized model held in a *swappable* state.

    Identical to ``build_final_model`` up to (but **not** including) the
    finishing steps: ``wrap_reparamed_modules_in_net`` and the bias
    reparameterization are deliberately skipped so that

    * qkv / fc1 stay ``AsymmetricallyChannelWiseBatchingQuantLinear`` (they
      keep their ``prev_layer`` pointer, so a single component can be
      re-quantized later without baking the reparam into a plain Linear), and
    * fc2 keeps its post-GELU shift un-folded.

    After calibration every quant module is in ``quant_forward`` mode, so any
    component re-quantized against this background sees genuinely quantized
    activations -- the "partially / fully quantized context" the refinement
    measures in.
    """
    name_to_bit = {c.name: assigned_bits[c.component_id] for c in components}
    bg = copy.deepcopy(fp_model)
    bg.to(device)
    bg.eval()
    wrap_modules_in_net(bg, cfg, reparam=True, bit_assignment=name_to_bit)
    bg.to(device)
    QuantCalibrator(bg, calib_loader).batching_quant_calib()
    bg.to(device)
    return bg


@contextmanager
def temporary_requant_in_background(
    bg: nn.Module,
    fp_source: nn.Module,
    spec: ComponentSpec,
    bit: int,
    cfg,
    calib_loader,
):
    """Re-quantize **only** ``spec`` at ``bit`` inside a quantized background.

    Mirrors :func:`temporary_adalog_component`, but the surrounding model is
    already quantized at the assigned bits (rather than full precision), so the
    measured fragility includes inter-layer quantization interactions.

    Steps on enter:
      1. Stash the component's current (calibrated) quant module and, for
         qkv / fc1, the background LayerNorm it reparameterized.
      2. Reset that LayerNorm to its full-precision affine params (from
         ``fp_source``) so the fresh reparam starts from FP -- otherwise the
         reparam would compound on top of the already-reparamed LN.
      3. Install a fresh AdaLog quant module for this component, sourcing the
         **original FP weights** from ``fp_source`` (the background module's
         weights may have been mutated by its own reparam).
      4. Calibrate that single module via ``QuantCalibrator``; every other
         module is already calibrated, so only this one is (re)searched, and it
         sees the quantized background.

    On exit: the original quant module is reinstated and the background
    LayerNorm is restored, leaving ``bg`` byte-for-byte ready for the next
    measurement.
    """
    name = spec.name
    bg_module = get_module_by_name(bg, name)
    fp_module = get_module_by_name(fp_source, name)

    ln = _reparam_layernorm_for(bg, name)
    fp_ln = _reparam_layernorm_for(fp_source, name)
    ln_bg_backup = None
    if ln is not None:
        ln_bg_backup = (ln.weight.data.clone(),
                        ln.bias.data.clone() if ln.bias is not None else None)
        # Reset background LN to FP before the fresh reparam.
        ln.weight.data.copy_(fp_ln.weight.data)
        if ln.bias is not None and fp_ln.bias is not None:
            ln.bias.data.copy_(fp_ln.bias.data)

    fresh = make_quant_module(
        bg, name, fp_module, cfg, reparam=True, bit_assignment={name: bit},
    )
    fresh.to(fp_module.weight.device)
    set_module_by_name(bg, name, fresh)

    try:
        QuantCalibrator(bg, calib_loader).batching_quant_calib()
        yield bg
    finally:
        set_module_by_name(bg, name, bg_module)
        if ln is not None and ln_bg_backup is not None:
            ln.weight.data.copy_(ln_bg_backup[0])
            if ln_bg_backup[1] is not None:
                ln.bias.data.copy_(ln_bg_backup[1])


def compute_refined_omega(
    bg: nn.Module,
    fp_source: nn.Module,
    spec: ComponentSpec,
    bit: int,
    cfg,
    ref_logits: torch.Tensor,
    calib_loader,
    device: torch.device,
    temperature: float = 1.0,
) -> float:
    """KL fragility of ``spec`` at ``bit`` measured in the quantized context.

    NOTE: this returns the *absolute* KL of the whole quantized background,
    which is dominated by the other (already-quantized) components and is not
    on the same scale as the first-pass isolated Omega.  Prefer
    :func:`compute_refined_omega_marginal`, which subtracts the background
    baseline so the value is the component's marginal in-context damage.
    """
    with temporary_requant_in_background(bg, fp_source, spec, bit, cfg, calib_loader):
        with torch.no_grad():
            parts = [bg(inp.to(device)).cpu() for inp, _ in calib_loader]
    q_logits = torch.cat(parts, dim=0)
    return _kl_divergence(ref_logits, q_logits, temperature)


@contextmanager
def temporary_fp_in_background(
    bg: nn.Module,
    fp_source: nn.Module,
    spec: ComponentSpec,
):
    """Temporarily restore ``spec`` to full precision inside the quantized bg.

    Used to measure the background baseline: the KL of the whole quantized
    model with *this one* component de-quantized.  Subtracting that baseline
    from the per-bit in-context KL isolates the component's marginal effect and
    removes the large constant offset contributed by the other ~49 quantized
    layers.

    The component's reparameterized LayerNorm (qkv / fc1) is also reset to FP
    so the baseline is a genuine "this component is not quantized" state, then
    restored on exit.
    """
    name = spec.name
    bg_module = get_module_by_name(bg, name)
    fp_module = get_module_by_name(fp_source, name)

    ln = _reparam_layernorm_for(bg, name)
    fp_ln = _reparam_layernorm_for(fp_source, name)
    ln_bg_backup = None
    if ln is not None:
        ln_bg_backup = (ln.weight.data.clone(),
                        ln.bias.data.clone() if ln.bias is not None else None)
        ln.weight.data.copy_(fp_ln.weight.data)
        if ln.bias is not None and fp_ln.bias is not None:
            ln.bias.data.copy_(fp_ln.bias.data)

    # Install a fresh FP copy so the background is never mutated by forward.
    fp_clone = copy.deepcopy(fp_module).to(fp_module.weight.device)
    set_module_by_name(bg, name, fp_clone)
    try:
        yield bg
    finally:
        set_module_by_name(bg, name, bg_module)
        if ln is not None and ln_bg_backup is not None:
            ln.weight.data.copy_(ln_bg_backup[0])
            if ln_bg_backup[1] is not None:
                ln.bias.data.copy_(ln_bg_backup[1])


def _enforce_monotonic(
    omega_c: Dict[int, float],
    candidate_bits: List[int],
) -> Dict[int, float]:
    """Clamp Omega to be non-increasing in bit-width.

    More bits can never *increase* quantization damage; any apparent rise is
    measurement noise (e.g. refined 4-bit scoring worse than 3-bit).  Walking
    ascending bits, each value is clamped to be <= the previous (lower-bit)
    value, discarding such noise.
    """
    out = dict(omega_c)
    sb = sorted(candidate_bits)
    for i in range(1, len(sb)):
        out[sb[i]] = min(out[sb[i]], out[sb[i - 1]])
    return out


def compute_refined_omega_marginal(
    bg: nn.Module,
    fp_source: nn.Module,
    spec: ComponentSpec,
    candidate_bits: List[int],
    cfg,
    ref_logits: torch.Tensor,
    calib_loader,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[int, float]:
    """Marginal in-context Omega of ``spec`` across all candidate bits.

    Returns ``{bit: omega}`` where

        omega(bit) = max(0, KL(FP || bg[spec=bit]) - KL(FP || bg[spec=FP]))

    The subtracted baseline ``KL(FP || bg[spec=FP])`` is the damage from *every
    other* quantized component; removing it leaves only what quantizing
    ``spec`` at ``bit`` adds, on the same (small) scale as the first-pass
    isolated Omega -- so the refined and un-refined entries can share one DP
    table.  Negative results (noise) are floored to 0.
    """
    candidate_bits = sorted(candidate_bits)

    # Background baseline: this component de-quantized, everything else quantized.
    with temporary_fp_in_background(bg, fp_source, spec):
        with torch.no_grad():
            parts = [bg(inp.to(device)).cpu() for inp, _ in calib_loader]
    base_kl = _kl_divergence(ref_logits, torch.cat(parts, dim=0), temperature)

    out: Dict[int, float] = {}
    for bit in candidate_bits:
        with temporary_requant_in_background(bg, fp_source, spec, bit, cfg, calib_loader):
            with torch.no_grad():
                parts = [bg(inp.to(device)).cpu() for inp, _ in calib_loader]
        kl = _kl_divergence(ref_logits, torch.cat(parts, dim=0), temperature)
        out[bit] = max(0.0, kl - base_kl)
    return out


def refinement_pass(
    fp_model: nn.Module,
    fp_source: nn.Module,
    components: List[ComponentSpec],
    omega: Dict[int, Dict[int, float]],
    assigned_bits: Dict[int, int],
    candidate_bits: List[int],
    cfg,
    ref_logits: torch.Tensor,
    calib_loader,
    device: torch.device,
    budget: int,
    top_k: int,
    temperature: float = 1.0,
    cost_scale: Optional[int] = None,
    allocation_strategy: str = "mckp",
) -> Tuple[Dict[int, int], Dict[int, Dict[int, float]], List[int]]:
    """Single feedback refinement round (LAMPQ-inspired).

    The first-pass Omega measures each component in isolation against the FP
    model, ignoring how components interact once the whole model is quantized.
    This round corrects the most sensitive ones in realistic context:

    1. Build a fully-quantized background at the current ``assigned_bits``.
    2. Rank components by first-pass sensitivity at their assigned bit and take
       the ``top_k`` most sensitive.
    3. For each such component re-measure its *marginal* Omega in the quantized
       background (KL with the component quantized minus KL with it at FP), so
       the value is on the same scale as the first-pass isolated Omega and free
       of the large constant background offset.  Results are clamped monotonic
       in bit-width to discard measurement noise.  Others keep their first-pass
       Omega.
    4. Re-run the selected allocation solver with the refined table.

    No reconstruction, no Hessian, no Fisher -- purely the same forward
    KL-divergence metric, just measured in a quantized context.

    Returns ``(new_assigned_bits, refined_omega, refined_component_ids)``.
    """
    candidate_bits = sorted(candidate_bits)

    # 1. fully-quantized swappable background at the current assignment
    logger.info("  Building quantized background for refinement ...")
    bg = _build_quant_background(
        fp_model, components, assigned_bits, cfg, calib_loader, device,
    )

    # 2. rank by first-pass sensitivity at the assigned bit
    ranked = sorted(
        components,
        key=lambda c: omega[c.component_id][assigned_bits[c.component_id]],
        reverse=True,
    )
    top = ranked[:max(0, top_k)]
    logger.info("  Re-measuring %d most-sensitive components in context:", len(top))
    for c in top:
        logger.info("    [%2d] %-45s (assigned %d-bit, Omega=%.3e)",
                    c.component_id, c.name, assigned_bits[c.component_id],
                    omega[c.component_id][assigned_bits[c.component_id]])

    # 3. recompute marginal in-context Omega for the top-K (all bits at once)
    refined: Dict[int, Dict[int, float]] = {
        c.component_id: dict(omega[c.component_id]) for c in components
    }
    with tqdm(total=len(top), desc="Refining Omega") as pbar:
        for c in top:
            pbar.set_description(f"Refine Omega({c.name})")
            vals = compute_refined_omega_marginal(
                bg, fp_source, c, candidate_bits, cfg, ref_logits,
                calib_loader, device, temperature,
            )
            vals = _enforce_monotonic(vals, candidate_bits)
            for bit in candidate_bits:
                logger.info("    refined Omega(%s, %d-bit) = %.6e  (was %.6e)",
                            c.name, bit, vals[bit], omega[c.component_id][bit])
                refined[c.component_id][bit] = vals[bit]
            pbar.update(1)

    # 4. rebuild the DP table with refined values and re-solve
    new_table = build_dp_table(components, refined, candidate_bits)
    new_assigned = solve_bit_allocation(
        new_table,
        budget,
        strategy=allocation_strategy,
        cost_scale=cost_scale,
    )
    refined_ids = [c.component_id for c in top]
    return new_assigned, refined, refined_ids


# ======================================================================
# Full pipeline orchestrator
# ======================================================================

def run_mixed_precision_pipeline(
    fp_model: nn.Module,
    cfg,
    calib_loader,
    device: torch.device,
    candidate_bits: Optional[List[int]] = None,
    target_avg_bit: float = 3.0,
    temperature: float = 1.0,
    cost_scale: Optional[int] = None,
    allocation_strategy: str = "mckp",
    refine: bool = True,
    refine_top_k: Optional[int] = None,
) -> Tuple[nn.Module, Dict[int, int], dict]:
    """End-to-end mixed-precision AdaLog quantization.

    Parameters
    ----------
    fp_model : nn.Module
        Pretrained full-precision ViT (any timm vision transformer).  Not
        mutated.
    cfg : Config
        AdaLog backend config (calib_batch_size, eq_n, search_round, fpcs,
        steps, post_gelu_quantizer, post_softmax_quantizer, qconv_a_bit,
        qhead_a_bit, a_bit, s_bit, w_bit, ...).  ``a_bit`` / ``s_bit`` set the
        global matmul precision in the final model.
    calib_loader : DataLoader
        Yields ``(images, labels)``; labels are ignored.
    candidate_bits : list[int] or None
        Per-component bit options (default ``[2, 3, 4]``).
    target_avg_bit : float
        Target average weight bit-width.
    temperature : float
        Softmax temperature for the KL divergence.
    cost_scale : int or None
        DP cost granularity (auto-detected when ``None``).
    allocation_strategy : str
        Bit-allocation strategy: ``"mckp"``, ``"greedy"``, or ``"random"``.
    refine : bool
        Run a single LAMPQ-inspired refinement round after the first DP
        solution (re-measures the most sensitive components in the quantized
        context and re-solves).  Default ``True``.
    refine_top_k : int or None
        Number of most-sensitive components re-measured during refinement.
        Defaults to ``max(1, len(components) // 4)``.

    Returns
    -------
    final_model : nn.Module
        The mixed-precision AdaLog model (fully calibrated).
    assigned_bits : dict[int, int]
        ``{component_id: bit}``.
    info : dict
        Diagnostics (components, omega, dp_table, budget, actual_avg_bit, ...).
    """
    if candidate_bits is None:
        candidate_bits = [2, 3, 4]
    candidate_bits = sorted(candidate_bits)

    # ---- Backend working model (FP, matmuls injected) ----
    logger.info("Preparing AdaLog backend working model ...")
    backend = prepare_backend_model(fp_model, device)

    # ---- Step 1: extract components ----
    logger.info("Step 1: Extracting quantizable components ...")
    components = extract_components(backend)
    total_params = sum(c.param_count for c in components)
    logger.info("  %d components, %s total weight params",
                len(components), f"{total_params:,}")
    for c in components:
        logger.info("  [%2d] %-45s %-8s params=%12s  block=%s  type=%s",
                    c.component_id, c.name, c.module_type,
                    f"{c.param_count:,}", c.block_index, c.component_type)

    # ---- Step 2: reference logits (FP) ----
    logger.info("Step 2: Computing reference logits ...")
    ref_logits = compute_reference_logits(backend, calib_loader, device)
    logger.info("  Cached %d samples, %d classes", *ref_logits.shape)

    # ---- Steps 3-4: fragility Omega (full AdaLog calibration per component) ----
    logger.info("Steps 3-4: Computing quantization fragility Omega ...")
    omega = compute_all_omegas(
        backend, components, candidate_bits, cfg,
        ref_logits, calib_loader, device, temperature,
    )

    # ---- Step 5: DP table ----
    logger.info("Step 5: Building DP cost/value table ...")
    dp_table = build_dp_table(components, omega, candidate_bits)
    budget = int(total_params * target_avg_bit)
    logger.info("  Budget = %s  (target %.2f-bit avg)", f"{budget:,}", target_avg_bit)

    # ---- Step 6: solve allocation ----
    logger.info("Step 6: Solving bit allocation with strategy '%s' ...",
                allocation_strategy)
    assigned_bits = solve_bit_allocation(
        dp_table,
        budget,
        strategy=allocation_strategy,
        cost_scale=cost_scale,
    )
    logger.info("  Initial bit assignments:")
    for c in components:
        logger.info("    %-45s -> %d-bit", c.name, assigned_bits[c.component_id])

    # ---- Step 6.5: refinement (single in-context feedback round) ----
    assigned_bits_pre_refine = dict(assigned_bits)
    refined_omega = None
    refined_ids: List[int] = []
    if refine and len(candidate_bits) > 1:
        if refine_top_k is None:
            refine_top_k = max(1, len(components) // 4)
        logger.info("Step 6.5: Refinement pass (top-%d sensitive components) ...",
                    refine_top_k)
        assigned_bits, refined_omega, refined_ids = refinement_pass(
            fp_model=fp_model, fp_source=backend, components=components,
            omega=omega, assigned_bits=assigned_bits, candidate_bits=candidate_bits,
            cfg=cfg, ref_logits=ref_logits, calib_loader=calib_loader, device=device,
            budget=budget, top_k=refine_top_k, temperature=temperature,
            cost_scale=cost_scale, allocation_strategy=allocation_strategy,
        )
        changed = {cid: (assigned_bits_pre_refine[cid], assigned_bits[cid])
                   for cid in assigned_bits
                   if assigned_bits_pre_refine[cid] != assigned_bits[cid]}
        if changed:
            name_by_id = {c.component_id: c.name for c in components}
            logger.info("  Refinement changed %d assignment(s):", len(changed))
            for cid, (old, new) in changed.items():
                logger.info("    %-45s %d-bit -> %d-bit", name_by_id[cid], old, new)
        else:
            logger.info("  Refinement left all assignments unchanged.")
    else:
        logger.info("Step 6.5: Refinement pass skipped.")

    actual_cost = sum(c.param_count * assigned_bits[c.component_id] for c in components)
    actual_avg = actual_cost / total_params
    logger.info("  Final bit assignments:")
    for c in components:
        logger.info("    %-45s -> %d-bit", c.name, assigned_bits[c.component_id])
    logger.info("  Actual average: %.4f bits", actual_avg)

    # ---- Step 7: final mixed-precision model ----
    logger.info("Step 7: Building + calibrating final mixed-precision model ...")
    final_model = build_final_model(
        fp_model, components, assigned_bits, cfg, calib_loader, device,
    )

    info = {
        'components': components,
        'omega': omega,
        'assigned_bits': assigned_bits,
        'assigned_bits_pre_refine': assigned_bits_pre_refine,
        'refined': bool(refine and len(candidate_bits) > 1),
        'refined_omega': refined_omega,
        'refined_component_ids': refined_ids,
        'dp_table': dp_table,
        'total_params': total_params,
        'budget': budget,
        'allocation_strategy': allocation_strategy,
        'actual_cost': actual_cost,
        'actual_avg_bit': actual_avg,
    }
    logger.info("Pipeline complete.")
    return final_model, assigned_bits, info


# ======================================================================
# Compression report
# ======================================================================

def compute_compression_report(
    fp_model: nn.Module,
    components: List[ComponentSpec],
    assigned_bits: Dict[int, int],
    fp_baseline_bit: int = 32,
) -> dict:
    """Compute the theoretical weight-memory compression of the final model.

    The reported sizes are *deployment* sizes -- low-bit weights packed at
    their assigned bit-width.  They are NOT the size of the saved ``.pth``,
    which stores full-precision weights plus quantizer parameters (it is a
    simulation checkpoint, so on-disk it is roughly FP32-sized).

    Components carry the quantized weights (qkv / proj / fc1 / fc2 /
    patch_embed / head).  Everything else in the model (LayerNorm affines,
    biases, class token, positional / relative-position embeddings) has no DP
    bit and is assumed kept at ``fp_baseline_bit`` precision.

    Returns a dict with byte counts, MB, average bit-width, compression
    ratios and a per-bit-width breakdown.
    """
    quant_params = sum(c.param_count for c in components)
    total_params = sum(p.numel() for p in fp_model.parameters())
    other_params = total_params - quant_params

    quant_weight_bits = sum(
        c.param_count * assigned_bits[c.component_id] for c in components
    )
    avg_bit = quant_weight_bits / quant_params if quant_params else 0.0

    # theoretical sizes in bytes
    fp_quant_bytes = quant_params * fp_baseline_bit / 8          # quantizable weights @ FP
    mp_quant_bytes = quant_weight_bits / 8                       # quantizable weights @ mixed bits
    other_bytes = other_params * fp_baseline_bit / 8            # kept-FP params

    fp_full_bytes = total_params * fp_baseline_bit / 8
    mp_full_bytes = mp_quant_bytes + other_bytes

    MB = 1024 * 1024

    # per-bit-width breakdown
    breakdown = {}
    for bit in sorted(set(assigned_bits.values())):
        comps_b = [c for c in components if assigned_bits[c.component_id] == bit]
        params_b = sum(c.param_count for c in comps_b)
        breakdown[bit] = {
            'n_components': len(comps_b),
            'params': params_b,
            'weight_mb': params_b * bit / 8 / MB,
        }

    return {
        'quant_params': quant_params,
        'other_params': other_params,
        'total_params': total_params,
        'avg_bit': avg_bit,
        'fp_quant_mb': fp_quant_bytes / MB,
        'mp_quant_mb': mp_quant_bytes / MB,
        'fp_full_mb': fp_full_bytes / MB,
        'mp_full_mb': mp_full_bytes / MB,
        'weight_compression_ratio': fp_quant_bytes / mp_quant_bytes if mp_quant_bytes else 0.0,
        'model_compression_ratio': fp_full_bytes / mp_full_bytes if mp_full_bytes else 0.0,
        'breakdown': breakdown,
    }


def log_compression_report(report: dict, top1: Optional[float] = None) -> None:
    """Pretty-print a compression report via the module logger."""
    logger.info("=" * 64)
    logger.info("COMPRESSION REPORT")
    logger.info("=" * 64)
    logger.info("Per bit-width breakdown (quantizable weights):")
    logger.info("  %-6s %-12s %-16s %-10s", "bit", "#components", "params", "weight MB")
    for bit, d in report['breakdown'].items():
        logger.info("  %-6d %-12d %-16s %-10.3f",
                    bit, d['n_components'], f"{d['params']:,}", d['weight_mb'])
    logger.info("-" * 64)
    logger.info("Quantizable weights : %s params  (avg %.4f bit)",
                f"{report['quant_params']:,}", report['avg_bit'])
    logger.info("  FP%-2d size          : %8.2f MB", 32, report['fp_quant_mb'])
    logger.info("  Mixed-prec size    : %8.2f MB", report['mp_quant_mb'])
    logger.info("  Weight compression : %8.2fx", report['weight_compression_ratio'])
    logger.info("-" * 64)
    logger.info("Full model (incl. norms / bias / embeddings kept FP32):")
    logger.info("  Total params       : %s", f"{report['total_params']:,}")
    logger.info("  FP32 size          : %8.2f MB", report['fp_full_mb'])
    logger.info("  Mixed-prec size    : %8.2f MB", report['mp_full_mb'])
    logger.info("  Model compression  : %8.2fx", report['model_compression_ratio'])
    if top1 is not None:
        logger.info("-" * 64)
        logger.info("  Top-1 accuracy     : %8.3f %%", top1)
    logger.info("=" * 64)
