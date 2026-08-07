"""
Gradio demo interface for the mixed-precision AdaLog thesis defense.

Scans ./checkpoints/mixed_precision/*/ for saved runs (a .pth checkpoint +
its output.log), rebuilds the quantized model exactly as the pipeline built
it (same wrapping + per-layer bit assignment parsed from the log), loads the
saved quantizer state, and serves a simple web UI:

    pick a quantized model -> upload / camera photo -> side-by-side
    comparison of the FP32 baseline vs. the proposed mixed-precision model
    (top-1 prediction, confidence, model size, average bit-width, latency).

Run:
    python demo_app.py
then open http://127.0.0.1:7860 in a browser.
"""

import gc
import os
import re
import glob
import importlib
import sys
import traceback

import torch
from torch import nn
import timm
from timm.data import resolve_data_config, create_transform

from utils.wrap_net import (
    wrap_modules_in_net,
    set_module_by_name,
)
from quant_layers.linear import (
    AsymmetricallyChannelWiseBatchingQuantLinear,
    AsymmetricallyBatchingQuantLinear,
)
from run_mixed_precision import MODEL_ZOO

CKPT_ROOT = "./checkpoints/mixed_precision"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Run discovery: pair every .pth with the metadata in its output.log
# ----------------------------------------------------------------------

def parse_run(log_path):
    """Extract model name, config path, bit map and metrics from output.log."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    m_model = re.search(r"Args: Namespace\(model='(\w+)'", text)
    m_cfg = re.search(r"config='([^']+)'", text)
    if not (m_model and m_cfg):
        return None

    # Per-layer bits: the block after "Final bit assignments:"
    bits = {}
    block = re.search(r"Final bit assignments:\n(.*?)\n\s*Actual average",
                      text, re.DOTALL)
    if not block:
        return None
    for line in block.group(1).splitlines():
        m = re.match(r"\s*(\S+)\s+-> (\d+)-bit", line)
        if m:
            bits[m.group(1)] = int(m.group(2))

    m_avg = re.search(r"Actual average: ([\d.]+) bits", text)
    m_top1 = re.search(r"Top-1 accuracy\s*:\s*([\d.]+)", text)
    m_refine = re.search(r"no_refine=(\w+)", text)

    return {
        "model": m_model.group(1),
        "config": m_cfg.group(1),
        "bits": bits,
        "avg_bit": float(m_avg.group(1)) if m_avg else None,
        "top1": float(m_top1.group(1)) if m_top1 else None,
        "no_refine": (m_refine and m_refine.group(1) == "True"),
    }


def discover_runs():
    runs = {}
    for log_path in sorted(glob.glob(os.path.join(CKPT_ROOT, "*", "output.log"))):
        run_dir = os.path.dirname(log_path)
        pths = glob.glob(os.path.join(run_dir, "*.pth"))
        if not pths:
            continue
        info = parse_run(log_path)
        if info is None or not info["bits"]:
            continue
        info["pth"] = pths[0]
        info["run_id"] = os.path.basename(run_dir)
        top1 = f"{info['top1']:.2f}%" if info["top1"] is not None else "?"
        avg = f"{info['avg_bit']:.2f}" if info["avg_bit"] is not None else "?"
        tier = (f"{round(info['avg_bit'])}-bit"
                if info["avg_bit"] is not None else "?-bit")
        # run dirs are named NN_model_wB, so sorted() = presentation order
        num = info["run_id"].split("_")[0]
        prefix = f"{num}.  " if num.isdigit() else ""
        label = (f"{prefix}{info['model']}  |  {tier} (avg {avg})"
                 f"  |  top-1 {top1}")
        runs[label] = info
    return runs


# ----------------------------------------------------------------------
# Model reconstruction from a checkpoint
# ----------------------------------------------------------------------

def load_config(config_path):
    dir_path = os.path.dirname(os.path.abspath(config_path))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    Config = getattr(importlib.import_module(module_name), "Config")
    return Config()


def convert_reparamed_structure(model):
    """Swap ChannelWise linears for plain Asymmetric ones (structure only).

    Mirrors ``wrap_reparamed_modules_in_net`` but skips the state copy: the
    checkpoint (saved *after* the original conversion) supplies all tensors.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, AsymmetricallyChannelWiseBatchingQuantLinear):
            new_module = AsymmetricallyBatchingQuantLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                mode="raw",
                w_bit=module.w_quantizer.n_bits,
                a_bit=module.a_quantizer.n_bits,
                calib_batch_size=module.calib_batch_size,
                search_round=module.search_round,
                eq_n=module.eq_n,
                n_V=module.n_V,
                fpcs=module.fpcs,
                steps=module.steps,
            )
            set_module_by_name(model, name, new_module)
    return model


def load_state_flexible(model, state_dict):
    """Assign every saved tensor onto the model, replacing shape mismatches.

    Quantizer params can change shape during calibration/reparam, so a strict
    ``load_state_dict`` would fail; direct attribute assignment always works.
    """
    modules = dict(model.named_modules())
    missing = []
    for key, tensor in state_dict.items():
        mod_path, _, attr = key.rpartition(".")
        mod = modules.get(mod_path)
        if mod is None:
            missing.append(key)
            continue
        if attr in mod._buffers:
            mod._buffers[attr] = tensor.clone()
        elif isinstance(getattr(mod, attr, None), nn.Parameter):
            setattr(mod, attr, nn.Parameter(tensor.clone(), requires_grad=False))
        else:
            setattr(mod, attr, nn.Parameter(tensor.clone(), requires_grad=False))
    return missing


def enable_quant_inference(model):
    """Put every quant module in calibrated quant_forward mode."""
    for m in model.modules():
        if hasattr(m, "mode"):
            m.mode = "quant_forward"
        if hasattr(m, "calibrated"):
            m.calibrated = True
        if hasattr(m, "inited"):
            m.inited = True


def build_quantized_model(info):
    cfg = load_config(info["config"])
    fp_model = timm.create_model(MODEL_ZOO[info["model"]], pretrained=False)
    fp_model.eval()

    wrap_modules_in_net(fp_model, cfg, reparam=True, bit_assignment=info["bits"])
    convert_reparamed_structure(fp_model)

    state = torch.load(info["pth"], map_location="cpu", weights_only=True)
    load_state_flexible(fp_model, state)
    enable_quant_inference(fp_model)

    fp_model.to(DEVICE)
    fp_model.eval()

    data_cfg = resolve_data_config({}, model=fp_model)
    transform = create_transform(**data_cfg)
    return fp_model, transform


# ----------------------------------------------------------------------
# ImageNet class names
# ----------------------------------------------------------------------

def get_class_names():
    try:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        return [
            info.label_name_to_description(info.index_to_label_name(i))
            for i in range(1000)
        ]
    except Exception:
        return [f"class {i}" for i in range(1000)]


CLASS_NAMES = get_class_names()


# ----------------------------------------------------------------------
# Inference (with a small model cache so switching models is instant)
# ----------------------------------------------------------------------

RUNS = discover_runs()
_cache = {}
_fp_cache = {}


def _evict_oldest(cache):
    """Drop the oldest cached model and actually release its memory."""
    old = cache.pop(next(iter(cache)))
    del old
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model(label):
    if label not in _cache:
        if len(_cache) >= 2:          # keep at most 2 models in memory
            _evict_oldest(_cache)
        _cache[label] = build_quantized_model(RUNS[label])
    return _cache[label]


def get_fp_model(model_key):
    """Full-precision baseline (pretrained timm weights, no quantization)."""
    if model_key not in _fp_cache:
        if len(_fp_cache) >= 2:
            _evict_oldest(_fp_cache)
        model = timm.create_model(MODEL_ZOO[model_key], pretrained=True)
        model.to(DEVICE)
        model.eval()
        data_cfg = resolve_data_config({}, model=model)
        transform = create_transform(**data_cfg)
        _fp_cache[model_key] = (model, transform)
    return _fp_cache[model_key]


# ----------------------------------------------------------------------
# Model-size accounting (theoretical storage size of the weights)
# ----------------------------------------------------------------------

def fp32_size_mb(model):
    return sum(p.numel() for p in model.parameters()) * 32 / 8 / 2 ** 20


def quant_size_mb(fp_model, bits):
    """Weights of quantized layers stored at their assigned bit-width,
    everything else (biases, norms, unquantized layers) kept at FP32."""
    total_bits = 0
    for name, p in fp_model.named_parameters():
        mod_path, _, pname = name.rpartition(".")
        b = bits.get(mod_path, 32) if pname == "weight" else 32
        total_bits += p.numel() * b
    return total_bits / 8 / 2 ** 20


def describe(label):
    info = RUNS[label]
    n2 = sum(1 for b in info["bits"].values() if b == 2)
    n3 = sum(1 for b in info["bits"].values() if b == 3)
    n4 = sum(1 for b in info["bits"].values() if b >= 4)
    top1 = f"{info['top1']:.2f} %" if info["top1"] is not None else "n/a"
    avg = f"{info['avg_bit']:.4f}" if info["avg_bit"] is not None else "n/a"
    return (
        f"**Model:** {info['model']} ({MODEL_ZOO[info['model']]})  \n"
        f"**Average weight bit-width:** {avg} bits  \n"
        f"**ImageNet val top-1 (50k images):** {top1}  \n"
        # f"**Layer bit distribution:** {n2} layers @2-bit, {n3} @3-bit, {n4} @>=4-bit  \n"
        f"**Refinement:** {'skipped (--no-refine)' if info['no_refine'] else 'enabled'}  \n"
        f"**Checkpoint:** `{os.path.basename(info['pth'])}`"
    )


@torch.no_grad()
def run_model(model, transform, image):
    """Forward pass -> (top-5 dict, top-1 name, top-1 prob)."""
    x = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits[0].float(), dim=-1)
    top = torch.topk(probs, 5)
    top5 = {
        CLASS_NAMES[i]: float(p)
        for p, i in zip(top.values.cpu(), top.indices.cpu())
    }
    return top5, CLASS_NAMES[top.indices[0]], float(top.values[0])


def stats_md(arch, top1_name, conf, size_mb, avg_bit, extra=""):
    return (
        f"**Architecture:** {arch}  \n"
        f"**Top-1:** {top1_name}  \n"
        f"**Confidence:** {conf * 100:.1f} %  \n"
        f"**Model Size:** {size_mb:.1f} MB  \n"
        f"**Avg Bit:** {avg_bit}{extra}"
    )


def compare(image, label):
    """Generator: yields (status, fp_top5, fp_stats, q_top5, q_stats).

    Streaming status keeps the UI responsive during the slow one-time
    steps (HF weight download, checkpoint rebuild) so it never looks
    frozen, and any error is reported instead of hanging the queue.
    """
    import gradio as gr

    keep = (gr.skip(), gr.skip(), gr.skip(), gr.skip())
    if image is None:
        yield ("⚠️ Upload an image first, then click **Compare**.", *keep)
        return

    try:
        info = RUNS[label]
        arch = info["model"]

        if arch not in _fp_cache:
            yield (f"⏳ Loading FP32 **{arch}** baseline — the first time "
                   "this downloads the pretrained weights from HuggingFace "
                   "(can take a few minutes on a slow connection)…", *keep)
        fp_model, fp_transform = get_fp_model(arch)

        if label not in _cache:
            yield (f"⏳ Rebuilding the quantized **{arch}** model from the "
                   "checkpoint (about a minute on CPU)…", *keep)
        q_model, q_transform = get_model(label)

        yield ("⏳ Running inference on both models…", *keep)
        fp_top5, fp_name, fp_conf = run_model(fp_model, fp_transform, image)
        q_top5, q_name, q_conf = run_model(q_model, q_transform, image)

        fp_size = fp32_size_mb(fp_model)
        q_size = quant_size_mb(fp_model, info["bits"])
        ratio = fp_size / q_size if q_size else 0.0
        avg_bit = (f"{info['avg_bit']:.2f}"
                   if info["avg_bit"] is not None else "n/a")

        fp_stats = stats_md(arch, fp_name, fp_conf, fp_size, "32")
        q_stats = stats_md(
            arch, q_name, q_conf, q_size, avg_bit,
            extra=f"  \n**Compression:** {ratio:.1f}x smaller",
        )
        yield ("", fp_top5, fp_stats, q_top5, q_stats)
    except Exception as e:
        traceback.print_exc()
        yield (f"❌ **Error:** {e} — check the terminal for details, "
               "then try again.", *keep)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

def main():
    import gradio as gr

    if not RUNS:
        raise SystemExit(f"No runs with .pth + output.log found under {CKPT_ROOT}")

    default = next(iter(RUNS))
    hide_footer = "footer {display: none !important}"
    with gr.Blocks(title="Mixed-Precision AdaLog PTQ — Live Demo") as demo:
        gr.Markdown(
            "# Mixed-Precision Post-Training Quantization — Live Demo\n"
            "Fragility-aware bit allocation on AdaLog-quantized transformers. "
            "Pick a quantized checkpoint, upload any photo (or use your webcam), "
            "then click **Compare** to see the FP32 baseline against the "
            "proposed ~3-bit model. The FP32 side is the full-precision "
            "pretrained version of the *same* architecture you selected."
        )
        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=list(RUNS.keys()), value=default,
                    label="Quantized checkpoint",
                )
                model_info = gr.Markdown(describe(default))
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="pil", sources=["upload", "webcam"],
                    label="Input image (upload / webcam)",
                )
                run_btn = gr.Button("Compare", variant="primary")

        status = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## FP32 Model")
                fp_preds = gr.Label(num_top_classes=5,
                                    label="Top-5 predictions (FP32)")
                fp_stats = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("## Proposed Model")
                q_preds = gr.Label(num_top_classes=5,
                                   label="Top-5 predictions (mixed-precision)")
                q_stats = gr.Markdown()

        outputs = [status, fp_preds, fp_stats, q_preds, q_stats]
        model_dd.change(describe, inputs=model_dd, outputs=model_info)
        run_btn.click(compare, inputs=[image_in, model_dd], outputs=outputs)

    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, css=hide_footer,
        inbrowser=os.environ.get("DEMO_NO_BROWSER") != "1")


if __name__ == "__main__":
    main()
