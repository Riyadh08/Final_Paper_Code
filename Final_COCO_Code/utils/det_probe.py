"""Task-output drift probe for COCO detection / instance segmentation.

The ImageNet pipeline measures a component's quantization fragility as the
forward KL divergence between the full-precision and the quantized classifier
posterior.  A detector has no single logit vector, so this module defines the
same quantity over the detector's three task-level output distributions:

===========  ==============================================  ================
term         distribution                                    KL form
===========  ==============================================  ================
``rpn``      dense per-anchor objectness (all FPN levels)    Bernoulli
``roi``      per-proposal class posterior over 80 + bg       categorical
``mask``     per-pixel mask posterior of the 28x28 logits    Bernoulli
===========  ==============================================  ================

Alignment is the only subtlety.  Quantizing the backbone changes which
proposals the RPN emits, so ROI / mask outputs from two different models are
not comparable element-wise.  The probe therefore caches the **full-precision
proposals** once and feeds those same RoIs to every subsequent model, and it
scores the mask logits on the channel selected by the **full-precision**
predicted label.  Both distributions are then indexed identically for the FP
and quantized models, and the KL is a genuine divergence between aligned
distributions rather than a comparison of two different detections.

Only the encoder is quantized (neck + heads stay FP32), so this measures
exactly what the bit allocation should care about: how much backbone
quantization perturbs the detector's decisions.
"""

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from utils.coco_data import batch_to_device

logger = logging.getLogger(__name__)

_EPS = 1e-7


def _bernoulli_kl(fp_logits, q_logits, temperature=1.0):
    """Mean KL( sigmoid(fp/T) || sigmoid(q/T) ) over all elements."""
    p = torch.sigmoid(fp_logits.float() / temperature).clamp(_EPS, 1 - _EPS)
    q = torch.sigmoid(q_logits.float() / temperature).clamp(_EPS, 1 - _EPS)
    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    return kl.mean().item()


def _categorical_kl(fp_logits, q_logits, temperature=1.0):
    """Mean-per-row KL( softmax(fp/T) || softmax(q/T) )."""
    p = F.softmax(fp_logits.float() / temperature, dim=-1)
    log_q = F.log_softmax(q_logits.float() / temperature, dim=-1)
    return F.kl_div(log_q, p, reduction='batchmean').item()


class DetectionDriftProbe:
    """Cache full-precision detector outputs, then score any model against them.

    Usage::

        probe = DetectionDriftProbe(model, calib_batches, device)
        probe.cache_reference()          # once, on the FP model
        omega = probe.drift(model)       # after quantizing something
    """

    def __init__(
        self,
        model,
        calib_batches,
        device,
        n_proposals: int = 100,
        weights: Sequence[float] = (1.0, 1.0, 1.0),
        temperature: float = 1.0,
        use_rpn: bool = True,
        use_roi: bool = True,
        use_mask: bool = True,
    ):
        self.model = model
        self.calib_batches = calib_batches
        self.device = device
        self.n_proposals = n_proposals
        self.w_rpn, self.w_roi, self.w_mask = weights
        self.temperature = temperature
        self.use_rpn = use_rpn
        self.use_roi = use_roi
        self.use_mask = use_mask and hasattr(model.roi_head, 'mask_head')

        # cached full-precision state
        self._rois: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._reference: Optional[List[Dict[str, torch.Tensor]]] = None
        self.last_terms: Dict[str, float] = {}

        # Cascade R-CNN keeps its heads in ModuleLists; probe the first stage,
        # which is the one that actually consumes RPN proposals.
        self.n_stages = getattr(model.roi_head, 'num_stages', None)

    # ------------------------------------------------------------------
    # forward helpers
    # ------------------------------------------------------------------

    def _bbox_forward(self, roi_head, feats, rois):
        if self.n_stages is not None:
            return roi_head._bbox_forward(0, feats, rois)
        return roi_head._bbox_forward(feats, rois)

    def _mask_forward(self, roi_head, feats, rois):
        if self.n_stages is not None:
            return roi_head._mask_forward(0, feats, rois)
        return roi_head._mask_forward(feats, rois=rois)

    @torch.no_grad()
    def _capture(self, model, batch_index, batch, cache_alignment):
        """Return this batch's output logits; optionally cache RoIs / labels."""
        from mmdet.structures.bbox import bbox2roi

        data = batch_to_device(batch, self.device)
        feats = model.extract_feat(data['inputs'])
        out: Dict[str, torch.Tensor] = {}

        if self.use_rpn:
            cls_scores = model.rpn_head(feats)[0]
            out['rpn'] = torch.cat([s.flatten() for s in cls_scores]).cpu()

        if not (self.use_roi or self.use_mask):
            return out

        if cache_alignment:
            proposals = model.rpn_head.predict(
                feats, data['data_samples'], rescale=False)
            boxes = [p.bboxes[:self.n_proposals] for p in proposals]
            if sum(b.shape[0] for b in boxes) == 0:
                raise RuntimeError(
                    "The full-precision RPN produced no proposals on a "
                    "calibration image; pick different calibration images.")
            self._rois.append(bbox2roi(boxes).cpu())

        rois = self._rois[batch_index].to(self.device)
        bbox_results = self._bbox_forward(model.roi_head, feats, rois)
        cls_score = bbox_results['cls_score']
        if self.use_roi:
            out['roi'] = cls_score.cpu()

        if self.use_mask:
            if cache_alignment:
                # foreground label predicted by the FP model (last column is bg)
                self._labels.append(cls_score[:, :-1].argmax(dim=1).cpu())
            labels = self._labels[batch_index].to(self.device)
            mask_preds = self._mask_forward(model.roi_head, feats, rois)['mask_preds']
            if mask_preds.shape[1] > 1:      # class-specific masks
                idx = torch.arange(mask_preds.shape[0], device=mask_preds.device)
                mask_preds = mask_preds[idx, labels]
            out['mask'] = mask_preds.cpu()

        return out

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cache_reference(self):
        """Run the full-precision detector and cache outputs + alignment."""
        self._rois, self._labels = [], []
        self._reference = [
            self._capture(self.model, i, b, cache_alignment=True)
            for i, b in enumerate(self.calib_batches)
        ]
        shapes = {k: tuple(v.shape) for k, v in self._reference[0].items()}
        logger.info("  Cached FP detector outputs, per-batch shapes: %s", shapes)
        return self._reference

    @torch.no_grad()
    def drift(self, model):
        """Weighted forward-KL drift of ``model`` from the cached FP outputs."""
        if self._reference is None:
            raise RuntimeError("cache_reference() must be called first.")

        totals = {'rpn': 0.0, 'roi': 0.0, 'mask': 0.0}
        for i, batch in enumerate(self.calib_batches):
            cur = self._capture(model, i, batch, cache_alignment=False)
            ref = self._reference[i]
            if 'rpn' in cur:
                totals['rpn'] += _bernoulli_kl(ref['rpn'], cur['rpn'],
                                               self.temperature)
            if 'roi' in cur:
                totals['roi'] += _categorical_kl(ref['roi'], cur['roi'],
                                                 self.temperature)
            if 'mask' in cur:
                totals['mask'] += _bernoulli_kl(ref['mask'], cur['mask'],
                                                self.temperature)

        n = len(self.calib_batches)
        terms = {k: v / n for k, v in totals.items()}
        self.last_terms = terms
        return (self.w_rpn * terms['rpn']
                + self.w_roi * terms['roi']
                + self.w_mask * terms['mask'])
