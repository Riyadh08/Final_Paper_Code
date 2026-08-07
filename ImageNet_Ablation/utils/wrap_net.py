import torch
from torch import nn
from quant_layers.linear import *
from quant_layers.matmul import *
from quant_layers.conv import *
from functools import partial
import timm
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from types import MethodType
from tqdm import tqdm


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def vit_attn_forward(self, x, attn_mask=None):
    B, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q, k = self.q_norm(q), self.k_norm(k)
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def swin_attn_forward(self, x, mask=None):
    B_, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ----------------------------------------------------------------------
# Module-tree navigation helpers
# ----------------------------------------------------------------------

def get_module_by_name(root, name):
    """Return the submodule addressed by a dotted ``name`` ('' -> root)."""
    if name == '':
        return root
    obj = root
    for part in name.split('.'):
        obj = getattr(obj, part)
    return obj


def set_module_by_name(root, name, value):
    """Replace the submodule addressed by a dotted ``name`` with ``value``."""
    parts = name.split('.')
    parent = get_module_by_name(root, '.'.join(parts[:-1]))
    setattr(parent, parts[-1], value)


def _parent_name(name):
    idx = name.rfind('.')
    return name[:idx] if idx != -1 else ''


# ----------------------------------------------------------------------
# Pass 1: inject FP MatMul modules into attention blocks
# ----------------------------------------------------------------------

def inject_matmuls(model):
    """Replace the fused attention math with explicit ``MatMul`` modules.

    This is a *full-precision* structural change (matmul1 = Q@K, matmul2 =
    attn@V).  It is numerically equivalent to the original attention and is
    required before any quantization so the score / probability matmuls
    become addressable modules.
    """
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attn_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(swin_attn_forward, module)
    return model


# ----------------------------------------------------------------------
# Bit resolution + single-module construction (shared by whole-model and
# per-component wrapping)
# ----------------------------------------------------------------------

def resolve_module_bits(name, module, cfg, bit_assignment=None):
    """Determine (w_bit, a_bit) for one module.

    * ``bit_assignment`` (dict name->bit) overrides the *weight* bit for the
      listed components, and the *activation* bit too for Linear layers.
    * Conv (patch_embed) keeps ``cfg.qconv_a_bit`` for its activations.
    * Matmuls are not handled here -- they are global (cfg.a_bit / cfg.s_bit).
    """
    assigned = bit_assignment.get(name) if bit_assignment else None
    if isinstance(module, nn.Conv2d):
        w_bit = assigned if assigned is not None else cfg.w_bit
        return w_bit, cfg.qconv_a_bit
    # nn.Linear
    if assigned is not None:
        return assigned, assigned
    a_bit = cfg.qhead_a_bit if 'head' in name else cfg.a_bit
    return cfg.w_bit, a_bit


def make_quant_module(model, name, module, cfg, reparam, bit_assignment=None):
    """Build the AdaLog quant replacement for a single leaf module.

    Mirrors the original ``wrap_modules_in_net`` selection logic exactly, so
    every module automatically receives the AdaLog mechanism appropriate to
    it (channel-wise LN reparam for qkv/fc1, post-GELU AdaLog for fc2,
    post-Softmax AdaLog for matmul2, asymmetric uniform otherwise).

    Returns the new module (weights copied, ``prev_layer`` set) or ``None``
    if the
    module is not quantizable.
    """
    # ---- Conv2d (patch embedding) ----
    if isinstance(module, nn.Conv2d):
        w_bit, a_bit = resolve_module_bits(name, module, cfg, bit_assignment)
        new_module = AsymmetricallyBatchingQuantConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
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
        new_module.bias.data.copy_(module.bias.data)
        return new_module

    # ---- attention score / probability matmuls (global bits) ----
    if isinstance(module, MatMul):
        father = get_module_by_name(model, _parent_name(name))
        matmul_kwargs = {
            'B_bit': cfg.a_bit,
            'mode': 'raw',
            'calib_batch_size': cfg.calib_batch_size,
            'search_round': cfg.search_round,
            'eq_n': cfg.eq_n,
            'head_channel_wise': cfg.matmul_head_channel_wise,
            'num_heads': father.num_heads,
            'fpcs': cfg.fpcs,
            'steps': cfg.steps,
        }
        if 'matmul2' in name:
            return PostSoftmaxAsymmetricallyBatchingQuantMatMul(
                A_bit=cfg.s_bit, **matmul_kwargs,
                quantizer=cfg.post_softmax_quantizer,
            )
        return AsymmetricallyBatchingQuantMatMul(A_bit=cfg.a_bit, **matmul_kwargs)

    # ---- Linear ----
    if isinstance(module, nn.Linear):
        w_bit, a_bit = resolve_module_bits(name, module, cfg, bit_assignment)
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
            'n_V': 3 if 'qkv' in name else 1,
            'fpcs': cfg.fpcs,
            'steps': cfg.steps,
        }
        father_name = _parent_name(name)
        father = get_module_by_name(model, father_name)

        if w_bit == a_bit and reparam and ('qkv' in name or 'reduction' in name or 'fc1' in name):
            grandfather = get_module_by_name(model, _parent_name(father_name))
            new_module = AsymmetricallyChannelWiseBatchingQuantLinear(**linear_kwargs)
            if 'qkv' in name:
                new_module.prev_layer = grandfather.norm1
            if 'fc1' in name:
                new_module.prev_layer = grandfather.norm2
            if 'reduction' in name:
                new_module.prev_layer = father.norm
        elif 'fc2' in name and cfg.post_gelu_quantizer in ['adalog', 'log2', 'logsqrt2', 'ptq4vit']:
            if cfg.post_gelu_quantizer in ['adalog', 'log2', 'logsqrt2']:
                new_module = PostGeluLogBasedBatchingQuantLinear(
                    **linear_kwargs, quantizer=cfg.post_gelu_quantizer,
                )
            else:  # ptq4vit
                new_module = PostGeluTwinUniformBatchingQuantLinear(**linear_kwargs)
        else:
            new_module = AsymmetricallyBatchingQuantLinear(**linear_kwargs)

        new_module.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        return new_module

    return None


# ----------------------------------------------------------------------
# Whole-model wrapping
# ----------------------------------------------------------------------

def wrap_modules_in_net(model, cfg, reparam=False, bit_assignment=None):
    """Replace every quantizable module in the model with its AdaLog version.

    Parameters
    ----------
    bit_assignment : dict[str, int] or None
        Optional per-module-name bit override (used for mixed precision).
        When ``None`` the uniform ``cfg`` bit-widths are used (original
        behaviour).
    """
    inject_matmuls(model)
    for name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Conv2d, nn.Linear, MatMul)):
            continue
        new_module = make_quant_module(model, name, module, cfg, reparam, bit_assignment)
        if new_module is not None:
            set_module_by_name(model, name, new_module)
    return model


def wrap_reparamed_modules_in_net(model):
    module_dict={}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, AsymmetricallyChannelWiseBatchingQuantLinear):
            idx = idx + 1 if idx != 0 else idx
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
            setattr(father_module, name[idx:], new_module)
    return model
