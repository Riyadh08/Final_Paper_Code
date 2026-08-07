"""AdaLog calibration driver for an MMDetection backbone.

``utils.calibrator.QuantCalibrator`` assumes a classification loader that
yields ``(images, labels)`` and a model called as ``model(images)``.  For
detection the calibration inputs are pre-preprocessed mmdet batches and the
forward pass is the **backbone only** (the neck and heads stay full precision,
so nothing downstream needs calibrating).

Two detection-specific adjustments beyond the loader change:

* **Adaptive ``calib_batch_size``.**  The batching search loops slice
  ``raw_input`` along dim 0.  For classification that dim is the image count
  (32), but in a Swin backbone it is the *window* count for attention tensors
  (thousands at COCO resolution) and the image count for the FFNs.  A single
  fixed value therefore either explodes GPU memory or degenerates into
  thousands of one-row chunks.  Each module gets a chunk size derived from its
  own tensor shape instead.
* Calibration runs under ``torch.no_grad()`` throughout, as in the original.
"""

import logging

import torch
from tqdm import tqdm

from quant_layers import MinMaxQuantMatMul, MinMaxQuantConv2d, MinMaxQuantLinear
from utils.calibrator import QuantCalibrator

logger = logging.getLogger(__name__)


class DetQuantCalibrator(QuantCalibrator):
    """Calibrate the quant modules inside a detection backbone.

    Parameters
    ----------
    backbone : nn.Module
        The (partially) wrapped backbone.  Only modules with
        ``calibrated == False`` are visited, so this is equally usable for
        calibrating one freshly installed component or a whole model.
    calib_batches : list[dict]
        Output of :func:`utils.coco_data.build_calib_batches`.
    device : torch.device
    chunk_elems : int
        Target number of activation elements per search chunk.  Lower it if a
        stage-0 layer runs out of GPU memory; raise it to go faster.
    """

    def __init__(self, backbone, calib_batches, device, chunk_elems=4_000_000):
        super().__init__(backbone, calib_batches)
        self.device = device
        self.chunk_elems = chunk_elems

    def _forward_calib_set(self):
        with torch.no_grad():
            for batch in self.calib_loader:
                self.model(batch['inputs'].to(self.device))

    def _adapt_calib_batch_size(self, module):
        """Pick a dim-0 chunk size for this module from its cached tensors."""
        if isinstance(module, MinMaxQuantMatMul):
            rows = module.raw_input[0].shape[0]
            per_row = (module.raw_input[0][0].numel()
                       + module.raw_input[1][0].numel())
        else:
            rows = module.raw_input.shape[0]
            per_row = module.raw_input[0].numel()
        per_row = max(per_row + module.raw_out[0].numel(), 1)
        module.calib_batch_size = int(max(1, min(rows, self.chunk_elems // per_row)))

    def batching_quant_calib(self):
        pending = [(name, module) for name, module in self.model.named_modules()
                   if hasattr(module, 'calibrated') and not module.calibrated]
        with tqdm(total=len(pending), leave=False) as progress_bar:
            for name, module in pending:
                progress_bar.set_description(f"calibrating {name}")
                hooks = [module.register_forward_hook(self.outp_forward_hook)]
                if isinstance(module, (MinMaxQuantLinear, MinMaxQuantConv2d)):
                    hooks.append(module.register_forward_hook(
                        self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantMatMul):
                    hooks.append(module.register_forward_hook(
                        self.double_input_forward_hook))

                self._forward_calib_set()

                module.raw_out = torch.cat(module.tmp_out, dim=0)
                if isinstance(module, (MinMaxQuantLinear, MinMaxQuantConv2d)):
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(t, dim=0)
                                        for t in module.tmp_input]
                for hook in hooks:
                    hook.remove()
                module.tmp_input = module.tmp_out = None

                self._adapt_calib_batch_size(module)
                with torch.no_grad():
                    module.hyperparameter_searching()
                    if getattr(module, 'prev_layer', None) is not None:
                        progress_bar.set_description(f"reparaming {name}")
                        module.reparam()
                progress_bar.update()

        for _, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "quant_forward"
