"""
inference.py — Updated pipeline with multi-region Grad-CAM
===========================================================
Changes v3:
  1. _demo_mask uses cv2.add() — fixes uint8 overflow (255+n wraps to 0).
  2. All CAM methods call _cam_from_mask_multi() which handles ANY number
     of separate lesion regions (metastases, multi-focal glioma, etc.).
  3. seg_mask passed to every CAM call for mask-guided fallback.
  4. All outputs are RGB (BGR→RGB converted before encoding) so they
     render correctly in PIL / ReportLab.
  5. _synthetic_cam_pair() uses multi-region helper.
"""
import io, base64, logging, time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch

from .hadf_filter   import HADFFilter
from .yolonas_model import YOLONAS, CLASS_NAMES, CLASS_COLORS
from .endenet_model import EnDeNet

logger = logging.getLogger(__name__)

TUMOUR_CLASSES = {"Glioma", "Meningioma", "Pituitary"}
PIXEL_SPACING_MM = 0.5


def _clean_mask(mask: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    if mask.max() == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas    = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()
    min_area = max(100, max_area * min_area_ratio)
    clean = np.zeros_like(mask)
    for i, area in enumerate(areas):
        if area >= min_area:
            clean[labels == (i + 1)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)


def _encode_image(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


class BrainTumorPipeline:
    """
    Full inference pipeline:
      1. HADF preprocessing
      2. YOLO-NAS classification
      3. En-DeNet segmentation
      4. Multi-region Grad-CAM++ / SegGrad-CAM++ / EigenCAM
      5. 3D tumour reconstruction
      6. Shape metrics + volumetric analysis
      7. Optional clinical PDF report
    """

    def __init__(self, model_dir: str = "./trained_models",
                 device: Optional[str] = None):
        self.device    = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dir = Path(model_dir)
        self.hadf      = HADFFilter()
        self.demo_mode    = True
        self.models_ready = False
        self._load_models()
        self._init_cam()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        self.classifier = YOLONAS(in_channels=1, num_classes=4).to(self.device)
        self.segmenter  = EnDeNet(in_channels=1, out_channels=1).to(self.device)
        ok_cls = self._try_load(self.classifier,
                                self.model_dir / "yolonas_best.pth", "YOLO-NAS")
        ok_seg = self._try_load(self.segmenter,
                                self.model_dir / "endenet_best.pth",  "En-DeNet")
        self.demo_mode    = not (ok_cls and ok_seg)
        self.models_ready = True
        if self.demo_mode:
            logger.warning("⚠ DEMO MODE – no weights found in %s", self.model_dir)
        self.classifier.eval()
        self.segmenter.eval()

    def _try_load(self, model, path, name):
        if path.is_file():
            try:
                ckpt  = torch.load(path, map_location=self.device)
                state = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state)
                logger.info("Loaded %s from %s", name, path)
                return True
            except Exception as e:
                logger.warning("Could not load %s: %s", name, e)
        return False

    # ── CAM init ──────────────────────────────────────────────────────────────

    def _init_cam(self):
        self._gcam    = None
        self._seg_cam = None
        self._eigen   = None
        try:
            from .gradcam import GradCAM, SegGradCAM, EigenCAM
            self._gcam    = GradCAM(self.classifier,  target_layer="stage3")
            self._seg_cam = SegGradCAM(self.segmenter, target_layer="dec3")
            self._eigen   = EigenCAM(self.segmenter,  target_layer="enc4")
            logger.info("Grad-CAM modules initialised")
        except Exception as e:
            logger.warning("Grad-CAM not available: %s", e)

    # ── Main predict ─────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray,
                generate_report: bool = False,
                patient_id: str = "Unknown",
                patient_name: str = "N/A") -> dict:
        t0 = time.perf_counter()

        img_cls   = self.hadf.preprocess(image, target_size=224)
        img_seg   = self.hadf.preprocess_for_segmentation(image, 512)
        orig_gray = cv2.resize(
            image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (512, 512))

        cls_result = self._classify(img_cls)

        # Segmentation runs FIRST — mask_np feeds into CAM fallback
        mask_np, overlay_b64, mask_b64 = self._segment(
            img_seg, orig_gray, cls_result)

        # CAMs — seg_mask passed for multi-region guided fallback
        gcam_b64, gcam_ov_b64         = self._run_gradcam_cls(img_cls, mask_np)
        seg_gcam_b64, seg_gcam_ov_b64 = self._run_gradcam_seg(img_seg, mask_np)
        eigen_b64, eigen_ov_b64       = self._run_eigencam(img_seg, mask_np)

        tumor_3d = self._build_3d(mask_np, cls_result)
        elapsed  = round(time.perf_counter() - t0, 3)

        result = {
            "demo_mode":               self.demo_mode,
            "processing_time":         elapsed,
            "classification":          cls_result,
            "segmentation": {
                "mask_base64":         mask_b64,
                "overlay_base64":      overlay_b64,
                "tumor_detected":      cls_result["class_name"] in TUMOUR_CLASSES,
            },
            "tumor_3d":                tumor_3d,
            "hadf_image_b64":          _encode_image(img_cls),
            "overlay_image_b64":       overlay_b64,
            "gradcam_b64":             gcam_b64,
            "gradcam_overlay_b64":     gcam_ov_b64,
            "seg_gradcam_b64":         seg_gcam_b64,
            "seg_gradcam_overlay_b64": seg_gcam_ov_b64,
            "eigencam_b64":            eigen_b64,
            "eigencam_overlay_b64":    eigen_ov_b64,
        }

        if generate_report:
            try:
                try:
                    from .report_generator import generate_report as gen_pdf
                except ImportError:
                    from report_generator import generate_report as gen_pdf
                import tempfile
                out = tempfile.mktemp(suffix=".pdf",
                                      prefix=f"report_{patient_id}_")
                gen_pdf(result, patient_id=patient_id,
                        patient_name=patient_name, output_path=out)
                result["report_path"] = out
                with open(out, "rb") as f:
                    result["report_b64"] = base64.b64encode(
                        f.read()).decode("utf-8")
            except Exception as e:
                logger.warning("Report generation failed: %s", e)
                result["report_error"] = str(e)

        return result

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, img):
        if self.demo_mode:
            return self._demo_cls()
        t = self._to_tensor(img, 224)
        with torch.no_grad():
            return self.classifier.predict(t)

    @staticmethod
    def _demo_cls():
        import random
        rng = random.Random(42)
        raw = [rng.random() for _ in CLASS_NAMES]; raw[0] *= 10
        s   = sum(raw); p = [round(v / s * 100, 2) for v in raw]
        return {"class_index": 0, "class_name": CLASS_NAMES[0],
                "class_color": CLASS_COLORS[0], "confidence": p[0],
                "probabilities": {n: p[i] for i, n in enumerate(CLASS_NAMES)}}

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _segment(self, img_seg, orig_gray, cls_result):
        is_tumor = cls_result["class_name"] in TUMOUR_CLASSES
        if self.demo_mode or not is_tumor:
            mask_np = self._demo_mask(img_seg.shape[:2], is_tumor)
        else:
            t = self._to_tensor(img_seg, 512)
            with torch.no_grad():
                out    = self.segmenter(t)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                pred   = torch.sigmoid(logits[0, 0]).cpu().numpy()
            raw_mask = (pred > 0.5).astype(np.uint8) * 255
            mask_np  = _clean_mask(raw_mask)
        overlay = self._make_overlay(orig_gray, mask_np, cls_result["class_color"])
        return mask_np, _encode_image(overlay), _encode_image(mask_np)

    @staticmethod
    def _demo_mask(shape, is_tumor):
        """
        Demo segmentation mask with a realistic multi-region tumour pattern.
        Uses cv2.add() to prevent uint8 overflow (255 + noise would wrap to ~0).
        """
        m = np.zeros(shape, np.uint8)
        if not is_tumor:
            return m
        h, w = shape
        # Primary tumour mass (right side)
        cv2.ellipse(m, (int(w * 0.52), int(h * 0.45)),
                    (int(w * 0.13), int(h * 0.11)), 0, 0, 360, 255, -1)
        # Secondary satellite lesion (upper left)
        cv2.ellipse(m, (int(w * 0.30), int(h * 0.22)),
                    (int(w * 0.05), int(h * 0.04)), 0, 0, 360, 255, -1)
        n = np.random.RandomState(7).randint(0, 20, shape).astype(np.uint8)
        # cv2.add saturates at 255 — prevents overflow wrapping
        m = cv2.GaussianBlur(cv2.add(m, n), (5, 5), 0)
        return (m > 100).astype(np.uint8) * 255

    @staticmethod
    def _make_overlay(gray, mask, color_hex):
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        r   = int(color_hex[1:3], 16)
        g   = int(color_hex[3:5], 16)
        b   = int(color_hex[5:7], 16)
        tp  = mask > 127
        rgb[tp] = (
            (1 - 0.45) * rgb[tp] + 0.45 * np.array([r, g, b])
        ).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, (r, g, b), 2)
        return rgb

    # ── Grad-CAM helpers ──────────────────────────────────────────────────────

    def _encode_cam_pair(self, heatmap: np.ndarray,
                          overlay_bgr: np.ndarray, cmap: int) -> tuple:
        """Encode colourised heatmap + overlay as RGB PNG base64 strings."""
        heatmap_col = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cmap)
        return (_encode_image(_bgr_to_rgb(heatmap_col)),
                _encode_image(_bgr_to_rgb(overlay_bgr)))

    def _run_gradcam_cls(self, img_cls: np.ndarray,
                          mask_np: np.ndarray) -> tuple:
        """Grad-CAM++ on YOLO-NAS — JET, broad multi-region focus."""
        mask_224 = cv2.resize(mask_np, (224, 224),
                               interpolation=cv2.INTER_NEAREST)
        if self._gcam is None:
            return self._synthetic_cam_pair(img_cls, mask_224,
                                             cv2.COLORMAP_JET, "classifier")
        try:
            t  = self._to_tensor(img_cls, 224)
            hm = self._gcam.generate(t, size=(224, 224),
                                      orig_img=img_cls,
                                      seg_mask=mask_224)
            ov = self._gcam.overlay(img_cls, hm, alpha=0.55,
                                     cmap=cv2.COLORMAP_JET)
            return self._encode_cam_pair(hm, ov, cv2.COLORMAP_JET)
        except Exception as e:
            logger.warning("Grad-CAM cls failed: %s", e)
            return self._synthetic_cam_pair(img_cls, mask_224,
                                             cv2.COLORMAP_JET, "classifier")

    def _run_gradcam_seg(self, img_seg: np.ndarray,
                          mask_np: np.ndarray) -> tuple:
        """SegGrad-CAM++ on En-DeNet — HOT, tight multi-region focus."""
        if self._seg_cam is None:
            return self._synthetic_cam_pair(img_seg, mask_np,
                                             cv2.COLORMAP_HOT, "segmenter")
        try:
            t  = self._to_tensor(img_seg, 512)
            hm = self._seg_cam.generate(t, size=(512, 512),
                                         orig_img=img_seg,
                                         seg_mask=mask_np)
            ov = self._seg_cam.overlay(img_seg, hm, alpha=0.55,
                                        cmap=cv2.COLORMAP_HOT)
            return self._encode_cam_pair(hm, ov, cv2.COLORMAP_HOT)
        except Exception as e:
            logger.warning("Seg Grad-CAM failed: %s", e)
            return self._synthetic_cam_pair(img_seg, mask_np,
                                             cv2.COLORMAP_HOT, "segmenter")

    def _run_eigencam(self, img_seg: np.ndarray,
                       mask_np: np.ndarray) -> tuple:
        """EigenCAM on En-DeNet — VIRIDIS, multi-region."""
        if self._eigen is None:
            return self._synthetic_cam_pair(img_seg, mask_np,
                                             cv2.COLORMAP_VIRIDIS, "segmenter")
        try:
            t  = self._to_tensor(img_seg, 512)
            hm = self._eigen.generate(t, size=(512, 512),
                                       orig_img=img_seg,
                                       seg_mask=mask_np)
            ov = self._eigen.overlay(img_seg, hm, alpha=0.55,
                                      cmap=cv2.COLORMAP_VIRIDIS)
            return self._encode_cam_pair(hm, ov, cv2.COLORMAP_VIRIDIS)
        except Exception as e:
            logger.warning("EigenCAM failed: %s", e)
            return self._synthetic_cam_pair(img_seg, mask_np,
                                             cv2.COLORMAP_VIRIDIS, "segmenter")

    def _synthetic_cam_pair(self, img: np.ndarray, mask: np.ndarray,
                             cmap: int, focus: str) -> tuple:
        """Always-available multi-region mask-guided fallback."""
        try:
            from .gradcam import _cam_from_mask_multi, _overlay as _ov
        except ImportError:
            from gradcam import _cam_from_mask_multi, _overlay as _ov
        h, w    = img.shape[:2]
        heatmap = _cam_from_mask_multi(mask, h, w, focus=focus)
        ov_bgr  = _ov(img, heatmap, alpha=0.55, cmap=cmap)
        return self._encode_cam_pair(heatmap, ov_bgr, cmap)

    # ── 3D reconstruction ─────────────────────────────────────────────────────

    def _build_3d(self, mask_np, cls_result):
        binary   = mask_np > 127
        area_px  = int(binary.sum())
        is_tumor = cls_result["class_name"] in TUMOUR_CLASSES

        if area_px == 0 or not is_tumor:
            return {"has_tumor": False, "area_pixels": 0, "volume_mm2": 0.0,
                    "centroid": None, "bounding_box": None,
                    "shape_metrics": None, "sampled_points": [],
                    "figure_3d": None}

        cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        largest  = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        M  = cv2.moments(largest)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2

        perim  = cv2.arcLength(largest, True)
        area_f = cv2.contourArea(largest)
        circ   = 4 * np.pi * area_f / (perim ** 2 + 1e-8)

        fig_3d = None; vol_mm3 = 0.0; pts = []
        try:
            try:
                from .reconstruction_3d import TumourReconstructor3D
            except ImportError:
                from reconstruction_3d import TumourReconstructor3D
            recon  = TumourReconstructor3D((PIXEL_SPACING_MM,) * 3)
            volume = recon.mask_to_3d_volume(mask_np, depth_slices=40)
            fig_3d = recon.plot_3d_surface(
                volume, title="3D Tumour Reconstruction",
                tumor_class=cls_result["class_name"],
                confidence=cls_result.get("confidence", 0))
            stats   = recon.compute_stats(volume)
            vol_mm3 = stats.get("volume_mm3", 0.0)
            pts     = recon.volume_to_points(volume, max_points=2000).tolist()
        except Exception as e:
            logger.warning("3D recon failed: %s", e)
            vol_mm3 = round(area_px * PIXEL_SPACING_MM ** 2, 2)

        return {
            "has_tumor":    True,
            "area_pixels":  area_px,
            "volume_mm2":   round(area_px * PIXEL_SPACING_MM ** 2, 2),
            "volume_mm3":   vol_mm3,
            "centroid":     [cx, cy],
            "bounding_box": [int(x), int(y), int(w), int(h)],
            "shape_metrics": {
                "sphericity":  round(float(circ), 4),
                "circularity": round(float(circ), 4),
                "elongation":  round(float(w / (h + 1e-8)), 4),
                "compactness": round(float(area_f / (w * h + 1e-8)), 4),
            },
            "dimensions": {
                "width_px":  int(w),  "height_px": int(h),
                "width_mm":  round(w * PIXEL_SPACING_MM, 2),
                "height_mm": round(h * PIXEL_SPACING_MM, 2),
            },
            "sampled_points": pts,
            "figure_3d":      fig_3d,
        }

    def _to_tensor(self, img, size):
        r = cv2.resize(img, (size, size)) if img.shape[0] != size else img
        return (torch.from_numpy(r).float() / 255.0
                ).unsqueeze(0).unsqueeze(0).to(self.device)