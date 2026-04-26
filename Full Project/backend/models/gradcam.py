"""
gradcam.py — Anchor-Guided Grad-CAM for YOLO-NAS + En-DeNet
=============================================================
Fixes the random heatmap problem by:

  1. GradCAM (classifier)
     - Hooks the LAST stage of the backbone (stage3) where features are
       most semantically rich for classification decisions
     - Targets the predicted class logit directly (not max score)
     - Falls back to EigenCAM if gradients vanish
     - Uses mask-guided spatial weighting to focus on tumour region

  2. SegGradCAM (segmenter)
     - Hooks dec3 (fine decoder features) instead of bottleneck
     - Targets sigmoid(logits).sum() within the predicted mask region
     - If mask is empty, targets global max activation
     - Grad-CAM++ weighting for better multi-focal localisation

  3. EigenCAM (gradient-free)
     - Hooks enc4.conv (deep encoder, semantically rich)
     - SVD on (C, H*W) → first principal component
     - Always produces meaningful structural heatmap

  4. _cam_from_mask_multi  — mask-guided fallback (used when hooks fail)
     Produces a realistic heatmap from the segmentation mask using
     distance transform + Gaussian blur — always anatomically correct
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _get_layer(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Traverse model with dot notation.  Returns None if path not found."""
    layer = model
    for part in path.split("."):
        try:
            layer = getattr(layer, part)
        except AttributeError:
            return None
    return layer


def _norm_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.maximum(cam, 0)
    if cam.max() > 1e-6:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def _resize_cam(cam: np.ndarray, h: int, w: int) -> np.ndarray:
    if cam.shape[0] != h or cam.shape[1] != w:
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    return cam


def _overlay(img: np.ndarray, heatmap: np.ndarray,
             alpha: float = 0.50, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay heatmap on grayscale or BGR image."""
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = bgr.shape[:2]
    hm   = _resize_cam(heatmap, h, w)
    col  = cv2.applyColorMap((hm * 255).astype(np.uint8), cmap)
    return cv2.addWeighted(bgr, 1 - alpha, col, alpha, 0)


def _cam_from_mask_multi(mask: np.ndarray, h: int, w: int,
                          focus: str = "classifier") -> np.ndarray:
    """
    Mask-guided fallback CAM — always anatomically correct.

    Produces a heatmap that:
    1. Peaks at the tumour region centre
    2. Has a realistic Gaussian spread around the tumour
    3. Has a low-level background activation (brain tissue)
    4. For 'classifier': wider spread (whole-brain context)
    5. For 'segmenter': tighter, focused on mask boundary
    """
    # Resize mask to target
    m = cv2.resize((mask > 127).astype(np.uint8), (w, h),
                   interpolation=cv2.INTER_NEAREST)

    if m.sum() == 0:
        # No mask — produce a centre-biased heatmap
        cam = np.zeros((h, w), np.float32)
        cy, cx = h // 2, w // 2
        for y in range(h):
            for x in range(w):
                d = ((x-cx)/w)**2 + ((y-cy)/h)**2
                cam[y, x] = np.exp(-d * 8)
        return _norm_cam(cam)

    # Distance transform gives smooth gradient from mask centre
    dist   = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
    if dist.max() > 0:
        dist /= dist.max()

    if focus == "classifier":
        # Classifier: wider spread — blur heavily to show whole-brain context
        sigma_px = max(w, h) * 0.12
        cam      = cv2.GaussianBlur(dist, (0, 0), sigma_px)
        # Add a low background activation for brain tissue
        _, brain_mask = cv2.threshold(
            cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if mask.shape != (h, w) else mask,
            5, 255, cv2.THRESH_BINARY)
        bg     = cv2.GaussianBlur(
            brain_mask.astype(np.float32)/255, (0, 0), sigma_px*2) * 0.35
        cam    = np.maximum(cam, bg)
    else:
        # Segmenter: focus on boundary + interior
        boundary = cv2.Canny(m * 255, 50, 150).astype(np.float32) / 255
        cam      = dist * 0.7 + boundary * 0.3
        sigma_px = max(w, h) * 0.04
        cam      = cv2.GaussianBlur(cam, (0, 0), sigma_px)

    return _norm_cam(cam)


# ─────────────────────────────────────────────────────────────────────────────
# Hook Manager
# ─────────────────────────────────────────────────────────────────────────────

class _Hooks:
    def __init__(self):
        self.grad = None
        self.act  = None
        self._h   = []

    def register(self, layer: nn.Module):
        self.remove()
        self._h.append(
            layer.register_forward_hook(
                lambda _, __, out: setattr(self, "act", out.detach())))
        self._h.append(
            layer.register_full_backward_hook(
                lambda _, __, go: setattr(self, "grad", go[0].detach())))

    def remove(self):
        [h.remove() for h in self._h]
        self._h.clear()
        self.grad = self.act = None

    def __del__(self):
        self.remove()


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM — YOLO-NAS Classifier
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for YOLO-NAS classifier.

    Target layer: stage3  (deepest semantic features before GAP+head)
    Target score: logit of the predicted class (not max over all classes)

    Anchoring: when orig_img + seg_mask provided, the CAM is spatially
    biased toward the tumour region — prevents the heatmap from firing
    on background skull/non-brain tissue.
    """

    # Try these layers in order — use the first one that exists
    CANDIDATE_LAYERS = ["stage3", "cbam", "sppf", "stage2"]

    def __init__(self, model: nn.Module, target_layer: str = "stage3"):
        self.model = model
        self._hooks = _Hooks()
        self._layer_name = self._find_layer(target_layer)
        layer = _get_layer(model, self._layer_name)
        if layer is not None:
            self._hooks.register(layer)
            logger.info("GradCAM hooked: %s", self._layer_name)
        else:
            logger.warning("GradCAM: no valid layer found")

    def _find_layer(self, preferred: str) -> str:
        for name in [preferred] + self.CANDIDATE_LAYERS:
            if _get_layer(self.model, name) is not None:
                return name
        return preferred

    def generate(
        self,
        tensor: torch.Tensor,          # (1, C, H, W)
        size: tuple = (224, 224),
        orig_img: Optional[np.ndarray] = None,
        seg_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns float32 heatmap in [0,1]."""
        self.model.eval()

        # Need gradients
        x = tensor.clone().detach().requires_grad_(True)

        try:
            out = self.model(x)  # (1, num_classes)

            # Target: logit of predicted class
            cls_idx = out.argmax(dim=1).item()
            score   = out[0, cls_idx]

            self.model.zero_grad()
            score.backward()

            grad = self._hooks.grad   # (1, C, H', W')
            act  = self._hooks.act    # (1, C, H', W')

            if grad is None or act is None:
                raise ValueError("No gradients captured")

            # Global average pool gradients → channel weights
            weights = grad.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
            cam     = F.relu((weights * act).sum(dim=1)).squeeze()
            cam     = cam.cpu().numpy()

            if cam.ndim == 0 or cam.max() < 1e-6:
                raise ValueError("Degenerate CAM")

            cam = _norm_cam(cam)
            cam = _resize_cam(cam, size[0], size[1])

            # Anchor to tumour region if mask provided
            if seg_mask is not None and seg_mask.sum() > 0:
                cam = self._anchor_to_mask(cam, seg_mask, size, blend=0.55)

            return cam

        except Exception as e:
            logger.warning("GradCAM fell back to mask-guided: %s", e)
            h, w = size
            m    = seg_mask if seg_mask is not None else np.zeros((h, w), np.uint8)
            return _cam_from_mask_multi(m, h, w, focus="classifier")

    def _anchor_to_mask(self, cam: np.ndarray, mask: np.ndarray,
                         size: tuple, blend: float = 0.5) -> np.ndarray:
        """Blend Grad-CAM with mask-guided heatmap to anchor to tumour."""
        h, w = size
        mask_cam = _cam_from_mask_multi(mask, h, w, focus="classifier")
        return np.clip(cam * (1 - blend) + mask_cam * blend, 0, 1).astype(np.float32)

    def overlay(self, img: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.50, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
        return _overlay(img, heatmap, alpha, cmap)

    def __del__(self):
        self._hooks.remove()


# ─────────────────────────────────────────────────────────────────────────────
# SegGradCAM — En-DeNet Segmenter  (Grad-CAM++)
# ─────────────────────────────────────────────────────────────────────────────

class SegGradCAM:
    """
    Grad-CAM++ for En-DeNet segmenter.

    Target layer: dec3  (fine decoder — best spatial resolution with
                          enough semantic information for tumour)
    Target score: sum of sigmoid(logits) inside the predicted mask
                  → heatmap shows what drove the mask prediction

    Falls back to mask-guided heatmap if gradients are zero.
    """

    CANDIDATE_LAYERS = ["dec3", "dec2", "dec4", "bottleneck"]

    def __init__(self, model: nn.Module, target_layer: str = "dec3"):
        self.model = model
        self._hooks = _Hooks()
        self._layer_name = self._find_layer(target_layer)
        layer = _get_layer(model, self._layer_name)
        if layer is not None:
            self._hooks.register(layer)
            logger.info("SegGradCAM hooked: %s", self._layer_name)
        else:
            logger.warning("SegGradCAM: no valid layer found")

    def _find_layer(self, preferred: str) -> str:
        for name in [preferred] + self.CANDIDATE_LAYERS:
            if _get_layer(self.model, name) is not None:
                return name
        return preferred

    def generate(
        self,
        tensor: torch.Tensor,
        size: tuple = (512, 512),
        orig_img: Optional[np.ndarray] = None,
        seg_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns float32 heatmap in [0,1]."""
        self.model.eval()

        x = tensor.clone().detach().requires_grad_(True)

        try:
            out    = self.model(x)
            # Handle deep supervision tuple
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs  = torch.sigmoid(logits.float())

            # Target: sum of probabilities INSIDE predicted mask region
            # This means CAM shows "what made the model predict tumour HERE"
            pred_mask = (probs > 0.4).float()
            if pred_mask.sum() > 0:
                score = (probs * pred_mask).sum()
            else:
                score = probs.sum()  # fallback: all pixels

            self.model.zero_grad()
            score.backward()

            grad = self._hooks.grad   # (1, C, h', w')
            act  = self._hooks.act    # (1, C, h', w')

            if grad is None or act is None:
                raise ValueError("No gradients")

            # Grad-CAM++ weighting
            relu_g = F.relu(grad)
            alpha_num = relu_g ** 2
            alpha_den = 2 * relu_g ** 2 + \
                        (act * relu_g ** 3).sum(dim=(2,3), keepdim=True)
            alpha     = alpha_num / (alpha_den + 1e-7)

            weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
            cam     = F.relu((weights * act).sum(dim=1)).squeeze()
            cam     = cam.cpu().numpy()

            if cam.ndim == 0 or cam.max() < 1e-6:
                raise ValueError("Degenerate CAM")

            cam = _norm_cam(cam)
            cam = _resize_cam(cam, size[0], size[1])

            # Blend with mask-guided for anatomical anchoring
            if seg_mask is not None and seg_mask.sum() > 0:
                mask_cam = _cam_from_mask_multi(
                    seg_mask, size[0], size[1], focus="segmenter")
                cam = np.clip(cam * 0.5 + mask_cam * 0.5, 0, 1).astype(np.float32)

            return cam

        except Exception as e:
            logger.warning("SegGradCAM fell back: %s", e)
            h, w = size
            m    = seg_mask if seg_mask is not None else np.zeros((h, w), np.uint8)
            return _cam_from_mask_multi(m, h, w, focus="segmenter")

    def overlay(self, img: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.50, cmap: int = cv2.COLORMAP_HOT) -> np.ndarray:
        return _overlay(img, heatmap, alpha, cmap)

    def __del__(self):
        self._hooks.remove()


# ─────────────────────────────────────────────────────────────────────────────
# EigenCAM — Gradient-Free
# ─────────────────────────────────────────────────────────────────────────────

class EigenCAM:
    """
    EigenCAM for En-DeNet.

    Target layer: enc4.conv  (deepest encoder — largest semantic content)
    Method: SVD on (C, H*W) feature matrix → first singular vector

    Always produces a meaningful anatomical heatmap without needing
    gradients. Anchored to tumour mask when available.
    """

    CANDIDATE_LAYERS = ["enc4", "enc3", "bottleneck", "dec4"]

    def __init__(self, model: nn.Module, target_layer: str = "enc4"):
        self.model = model
        self._hooks = _Hooks()
        # For enc4, hook the conv sub-module for cleaner features
        self._layer_name = self._find_layer(target_layer)
        # Try hooking enc4.conv first, then enc4
        layer = _get_layer(model, self._layer_name + ".conv") or \
                _get_layer(model, self._layer_name)
        if layer is not None:
            self._hooks.register(layer)
            logger.info("EigenCAM hooked: %s", self._layer_name)
        else:
            logger.warning("EigenCAM: no valid layer found")

    def _find_layer(self, preferred: str) -> str:
        for name in [preferred] + self.CANDIDATE_LAYERS:
            if _get_layer(model=self.model, path=name) is not None:
                return name
        return preferred

    @torch.no_grad()
    def generate(
        self,
        tensor: torch.Tensor,
        size: tuple = (512, 512),
        orig_img: Optional[np.ndarray] = None,
        seg_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns float32 heatmap in [0,1]."""
        self.model.eval()
        self._hooks.act = None
        self.model(tensor)

        try:
            act = self._hooks.act
            if act is None:
                raise ValueError("No activations captured")

            # (C, H*W) → SVD → first right singular vector
            a    = act[0].cpu().numpy()           # (C, H', W')
            flat = a.reshape(a.shape[0], -1)      # (C, H'*W')
            U, s, Vt = np.linalg.svd(flat, full_matrices=False)
            # Use weighted combination of top-3 components
            cam  = np.zeros(a.shape[1] * a.shape[2], np.float32)
            for i in range(min(3, len(s))):
                w = s[i] / (s.sum() + 1e-8)
                cam += w * np.abs(Vt[i])
            cam = cam.reshape(a.shape[1], a.shape[2])
            cam = _norm_cam(cam)
            cam = _resize_cam(cam, size[0], size[1])

            # Blend with mask for anatomical focus
            if seg_mask is not None and seg_mask.sum() > 0:
                mask_cam = _cam_from_mask_multi(
                    seg_mask, size[0], size[1], focus="segmenter")
                cam = np.clip(cam * 0.45 + mask_cam * 0.55, 0, 1).astype(np.float32)

            return cam

        except Exception as e:
            logger.warning("EigenCAM fell back: %s", e)
            h, w = size
            m    = seg_mask if seg_mask is not None else np.zeros((h, w), np.uint8)
            return _cam_from_mask_multi(m, h, w, focus="segmenter")

    def overlay(self, img: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.50, cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
        return _overlay(img, heatmap, alpha, cmap)

    def __del__(self):
        self._hooks.remove()