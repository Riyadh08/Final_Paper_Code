"""Bridge between an **MMDetection Swin backbone** and the AdaLog quant layers.

``utils/wrap_net.py`` targets timm ViT / Swin.  MMDetection's Swin has a
different module tree, so the naming rules, the LayerNorm reparameterization
map and the attention rewrite all need their own version:

    timm                             mmdet
    ----------------------------     ---------------------------------------
    patch_embed.proj  (Conv2d)       patch_embed.projection  (Conv2d)
    blocks.{i}.attn.qkv              stages.{s}.blocks.{i}.attn.w_msa.qkv
    blocks.{i}.attn.proj             stages.{s}.blocks.{i}.attn.w_msa.proj
    blocks.{i}.mlp.fc1               stages.{s}.blocks.{i}.ffn.layers.0.0
    blocks.{i}.mlp.fc2               stages.{s}.blocks.{i}.ffn.layers.1
    layers.{s}.downsample.reduction  stages.{s}.downsample.reduction

Everything downstream of the backbone (FPN neck, RPN, ROI / mask heads) is
left in full precision -- the convention used by the ViT-PTQ detection
literature, so the reported AP is comparable.

The quantization mechanics themselves are unchanged: the same AdaLog quant
modules from ``quant_layers/`` are installed, with the same per-module
selection rules (channel-wise LN reparam for qkv / fc1 / reduction, post-GELU
AdaLog for fc2, post-Softmax AdaLog for attn@V).
"""

import logging
import re
from types import MethodType

import torch
import torch.nn as nn

from quant_layers.linear import (
    AsymmetricallyBatchingQuantLinear,
    AsymmetricallyChannelWiseBatchingQuantLinear,
    PostGeluLogBasedBatchingQuantLinear,
    PostGeluTwinUniformBatchingQuantLinear,
)
from quant_layers.conv import AsymmetricallyBatchingQuantConv2d
from quant_layers.matmul import (
    AsymmetricallyBatchingQuantMatMul,
    PostSoftmaxAsymmetricallyBatchingQuantMatMul,
)
from utils.wrap_net import MatMul, get_module_by_name, set_module_by_name, _parent_name

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Lazy mmdet imports (mmdet is only needed for the detection experiments)
# ----------------------------------------------------------------------

def get_mmdet_classes():
    """Return ``(WindowMSA, PatchMerging, SwinBlock)`` from mmdet.

    Imported lazily so the ImageNet pipeline never needs mmdet installed.
    """
    try:
        from mmdet.models.backbones.swin import SwinBlock, WindowMSA
        from mmcv.cnn.bricks.transformer import PatchMerging
    except ImportError as e:  # pragma: no cover - depends on user environment
        raise ImportError(
            "The COCO detection pipeline needs mmdetection >= 3.0 "
            "(with mmcv >= 2.0 and mmengine). Original error: %s" % e
        )
    return WindowMSA, PatchMerging, SwinBlock


# ----------------------------------------------------------------------
# Attention rewrite: expose Q.K and attn.V as addressable MatMul modules
# ----------------------------------------------------------------------

def window_msa_forward(self, x, mask=None):
    """``mmdet.models.backbones.swin.WindowMSA.forward`` with explicit matmuls.

    Numerically identical to the original -- the only change is that the two
    ``@`` operators become ``self.matmul1`` / ``self.matmul2`` modules so they
    can be quantized.  :func:`inject_matmuls_det` verifies this equivalence at
    injection time, which also guards against mmdet API drift.
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                              C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B // nW, nW, self.num_heads, N,
                         N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)

    attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


@torch.no_grad()
def _verify_window_msa_forward(module, atol=1e-5):
    """Assert the rewritten WindowMSA forward matches mmdet's own.

    Runs one random window batch through the original bound method and through
    :func:`window_msa_forward`, both with and without an attention mask.  A
    mismatch means the installed mmdet version changed ``WindowMSA.forward``
    and the rewrite above needs updating -- far better to fail here than to
    silently report wrong AP.
    """
    original_forward = module.forward          # bound, un-patched
    device = next(module.parameters()).device
    dtype = next(module.parameters()).dtype

    n = module.window_size[0] * module.window_size[1]
    c = module.qkv.in_features
    n_windows = 4
    x = torch.randn(n_windows, n, c, device=device, dtype=dtype)
    mask = torch.zeros(n_windows, n, n, device=device, dtype=dtype)
    mask[1] = -100.0                            # a non-trivial mask pattern

    module.matmul1 = MatMul()
    module.matmul2 = MatMul()
    patched = MethodType(window_msa_forward, module)

    was_training = module.training
    module.eval()
    try:
        for m in (None, mask):
            ref = original_forward(x, m)
            new = patched(x, m)
            if not torch.allclose(ref, new, atol=atol, rtol=1e-4):
                raise RuntimeError(
                    "window_msa_forward does not reproduce this mmdet version's "
                    "WindowMSA.forward (max abs diff %.3e). Update "
                    "utils/wrap_net_det.py:window_msa_forward to match "
                    "mmdet/models/backbones/swin.py."
                    % (ref - new).abs().max().item()
                )
    finally:
        del module.matmul1, module.matmul2
        if was_training:
            module.train()


def inject_matmuls_det(backbone, verify=True):
    """Replace the fused attention math in every WindowMSA with MatMul modules.

    A full-precision, numerically equivalent structural change; must run before
    any quantization so Q.K and attn.V become addressable modules.
    """
    WindowMSA, _, _ = get_mmdet_classes()
    count = 0
    for _, module in backbone.named_modules():
        if not isinstance(module, WindowMSA):
            continue
        if verify and count == 0:
            _verify_window_msa_forward(module)
        module.matmul1 = MatMul()
        module.matmul2 = MatMul()
        module.forward = MethodType(window_msa_forward, module)
        count += 1
    logger.info("  Injected explicit matmuls into %d WindowMSA blocks", count)
    return backbone


# ----------------------------------------------------------------------
# Component identification (names are relative to the *backbone*)
# ----------------------------------------------------------------------

def identify_component_type_det(name, module):
    """Classify a leaf backbone module, or return ``None`` if not quantizable."""
    if isinstance(module, nn.Conv2d):
        return 'patch_embed' if 'patch_embed' in name else None
    if not isinstance(module, nn.Linear):
        return None
    if name.endswith('.qkv'):
        return 'attn_qkv'
    if name.endswith('.w_msa.proj'):
        return 'attn_proj'
    if name.endswith('.ffn.layers.0.0'):
        return 'mlp_fc1'
    if name.endswith('.ffn.layers.1'):
        return 'mlp_fc2'
    if name.endswith('.downsample.reduction'):
        return 'patch_merging'
    return None


def extract_block_index_det(name):
    """``stages.{s}.blocks.{i}`` -> ``s * 100 + i``; stage-level / stem -> -1.

    ``stages.{s}.downsample`` is given ``s * 100 + 99`` so patch-merging layers
    sort after the blocks of their own stage.
    """
    stage_m = re.search(r'stages\.(\d+)', name)
    block_m = re.search(r'blocks\.(\d+)', name)
    if stage_m is None:
        return -1
    stage = int(stage_m.group(1))
    if block_m is not None:
        return stage * 100 + int(block_m.group(1))
    if 'downsample' in name:
        return stage * 100 + 99
    return -1


def reparam_layernorm_for_det(backbone, name):
    """Return the LayerNorm that AdaLog's channel-wise reparam will rewrite.

    Only three module kinds carry this cross-module side effect:

    * ``...blocks.{i}.attn.w_msa.qkv``      <- ``...blocks.{i}.norm1``
    * ``...blocks.{i}.ffn.layers.0.0``      <- ``...blocks.{i}.norm2``
    * ``...downsample.reduction``           <- ``...downsample.norm``

    Returns ``None`` for everything else.
    """
    if name.endswith('.qkv'):
        block = _parent_name(_parent_name(_parent_name(name)))   # strip w_msa.attn
        block_mod = get_module_by_name(backbone, block)
        return getattr(block_mod, 'norm1', None)
    if name.endswith('.ffn.layers.0.0'):
        block = _parent_name(_parent_name(_parent_name(_parent_name(name))))
        block_mod = get_module_by_name(backbone, block)
        return getattr(block_mod, 'norm2', None)
    if name.endswith('.downsample.reduction'):
        ds_mod = get_module_by_name(backbone, _parent_name(name))
        return getattr(ds_mod, 'norm', None)
    return None


# ----------------------------------------------------------------------
# Quant module construction
# ----------------------------------------------------------------------

def resolve_module_bits_det(name, module, cfg, bit_assignment=None):
    """Determine ``(w_bit, a_bit)`` for one backbone module.

    Mirrors ``utils.wrap_net.resolve_module_bits``: a DP assignment overrides
    the weight bit and, for Linear layers, the activation bit as well (so a
    "4MP" model is mixed precision in both weights and activations).  The patch
    embedding Conv keeps ``cfg.qconv_a_bit`` on its input, which is the raw
    image.
    """
    assigned = bit_assignment.get(name) if bit_assignment else None
    if isinstance(module, nn.Conv2d):
        return (assigned if assigned is not None else cfg.w_bit), cfg.qconv_a_bit
    if assigned is not None:
        return assigned, assigned
    return cfg.w_bit, cfg.a_bit


def make_quant_module_det(backbone, name, module, cfg, reparam, bit_assignment=None):
    """Build the AdaLog quant replacement for one backbone leaf module.

    Same selection logic as ``utils.wrap_net.make_quant_module``, retargeted to
    mmdet's names.  Returns ``None`` for modules that are not quantized.
    """
    # ---- patch embedding Conv2d ----
    if isinstance(module, nn.Conv2d):
        if 'patch_embed' not in name:
            return None
        w_bit, a_bit = resolve_module_bits_det(name, module, cfg, bit_assignment)
        new_module = AsymmetricallyBatchingQuantConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            mode='raw',
            w_bit=w_bit,
            a_bit=a_bit,
            calib_batch_size=cfg.calib_batch_size,
            search_round=cfg.search_round,
            eq_n=cfg.eq_n,
            fpcs=cfg.fpcs,
            steps=cfg.steps,
        )
        new_module.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        return new_module

    # ---- attention score / probability matmuls (global bits) ----
    if isinstance(module, MatMul):
        w_msa = get_module_by_name(backbone, _parent_name(name))
        matmul_kwargs = {
            'B_bit': cfg.a_bit,
            'mode': 'raw',
            'calib_batch_size': cfg.calib_batch_size,
            'search_round': cfg.search_round,
            'eq_n': cfg.eq_n,
            'head_channel_wise': cfg.matmul_head_channel_wise,
            'num_heads': w_msa.num_heads,
            'fpcs': cfg.fpcs,
            'steps': cfg.steps,
        }
        if name.endswith('matmul2'):
            return PostSoftmaxAsymmetricallyBatchingQuantMatMul(
                A_bit=cfg.s_bit, **matmul_kwargs,
                quantizer=cfg.post_softmax_quantizer,
            )
        return AsymmetricallyBatchingQuantMatMul(A_bit=cfg.a_bit, **matmul_kwargs)

    # ---- Linear ----
    if isinstance(module, nn.Linear):
        ctype = identify_component_type_det(name, module)
        if ctype is None:
            return None
        w_bit, a_bit = resolve_module_bits_det(name, module, cfg, bit_assignment)
        linear_kwargs = {
            'in_features': module.in_features,
            'out_features': module.out_features,
            'bias': module.bias is not None,
            'mode': 'raw',
            'w_bit': w_bit,
            'a_bit': a_bit,
            'calib_batch_size': cfg.calib_batch_size,
            'search_round': cfg.search_round,
            'eq_n': cfg.eq_n,
            'n_V': 3 if ctype == 'attn_qkv' else 1,
            'fpcs': cfg.fpcs,
            'steps': cfg.steps,
        }

        ln = reparam_layernorm_for_det(backbone, name) if reparam else None
        if w_bit == a_bit and ln is not None:
            new_module = AsymmetricallyChannelWiseBatchingQuantLinear(**linear_kwargs)
            new_module.prev_layer = ln
        elif ctype == 'mlp_fc2' and cfg.post_gelu_quantizer in (
                'adalog', 'log2', 'logsqrt2', 'ptq4vit'):
            if cfg.post_gelu_quantizer == 'ptq4vit':
                new_module = PostGeluTwinUniformBatchingQuantLinear(**linear_kwargs)
            else:
                new_module = PostGeluLogBasedBatchingQuantLinear(
                    **linear_kwargs, quantizer=cfg.post_gelu_quantizer)
        else:
            new_module = AsymmetricallyBatchingQuantLinear(**linear_kwargs)

        new_module.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        return new_module

    return None


def wrap_backbone_in_net(backbone, cfg, reparam=False, bit_assignment=None):
    """Replace every quantizable backbone module with its AdaLog version.

    ``bit_assignment`` maps backbone-relative module names to weight bits (the
    mixed-precision assignment).  When ``None`` the uniform ``cfg`` bit-widths
    are used, which reproduces plain AdaLog on the detector backbone.
    """
    inject_matmuls_det(backbone)
    for name, module in list(backbone.named_modules()):
        if not isinstance(module, (nn.Conv2d, nn.Linear, MatMul)):
            continue
        new_module = make_quant_module_det(
            backbone, name, module, cfg, reparam, bit_assignment)
        if new_module is not None:
            set_module_by_name(backbone, name, new_module)
    return backbone


def wrap_reparamed_modules_in_backbone(backbone):
    """Fold reparameterized channel-wise Linears back into plain quant Linears.

    Same purpose as ``utils.wrap_net.wrap_reparamed_modules_in_net`` (after
    ``reparam()`` the channel-wise activation quantizer has been collapsed to a
    single layer-wise scale, so the module can be swapped for the cheaper
    layer-wise class), but it snapshots ``named_modules()`` first instead of
    mutating the tree mid-iteration.
    """
    named = list(backbone.named_modules())
    for name, module in named:
        if not isinstance(module, AsymmetricallyChannelWiseBatchingQuantLinear):
            continue
        linear_kwargs = {
            'in_features': module.in_features,
            'out_features': module.out_features,
            'bias': module.bias is not None,
            'mode': module.mode,
            'w_bit': module.w_quantizer.n_bits,
            'a_bit': module.a_quantizer.n_bits,
            'calib_batch_size': module.calib_batch_size,
            'search_round': module.search_round,
            'eq_n': module.eq_n,
            'n_V': module.n_V,
            'fpcs': module.fpcs,
            'steps': module.steps,
        }
        new_module = AsymmetricallyBatchingQuantLinear(**linear_kwargs)
        new_module.load_state_dict(module.state_dict())
        new_module.calibrated = True
        new_module.a_quantizer.inited = True
        new_module.w_quantizer.inited = True
        new_module.to(module.weight.device)
        set_module_by_name(backbone, name, new_module)
    return backbone


def finish_bias_reparam(backbone):
    """Apply AdaLog's bias reparameterization to the post-GELU layers."""
    for _, module in backbone.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'reparam_bias'):
            module.reparam_bias()
