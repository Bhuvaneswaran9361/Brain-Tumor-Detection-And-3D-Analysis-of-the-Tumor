"""
HADF – Hybrid Anisotropic Diffusion Filter
==========================================
Preprocesses MRI/CT scans by:
  1. Grayscale conversion + ROI extraction
  2. Anisotropic diffusion filtering (kurtosis-based stopping)
  3. Normalization to target size

Mathematical basis:
  I_{t+1} = I_t + γ · ∇·(c(∇I) · ∇I)
  c(∇I)   = exp(-(‖∇I‖/κ)²)   ← Perona-Malik coefficient
  Stops when |K_t − K_{t-1}| ≤ ε  (kurtosis convergence)
"""

import numpy as np
import cv2
import logging
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class HADFFilter:
    """Hybrid Anisotropic Diffusion Filter for MRI preprocessing."""

    def __init__(
        self,
        kappa: float = 50.0,
        gamma: float = 0.1,
        kurtosis_threshold: float = 0.001,
        max_iterations: int = 100,
    ):
        self.kappa = kappa
        self.gamma = gamma
        self.kurtosis_threshold = kurtosis_threshold
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray, target_size: int = 224) -> np.ndarray:
        """Full preprocessing pipeline: grayscale → ROI → HADF → resize."""
        gray = self._to_grayscale(image)
        roi  = self._extract_roi(gray)
        filtered = self._anisotropic_diffusion(roi)
        resized  = cv2.resize(filtered, (target_size, target_size))
        return self._normalize(resized)

    def preprocess_for_segmentation(self, image: np.ndarray, target_size: int = 512) -> np.ndarray:
        """Same pipeline but at 512×512 for the segmentation model."""
        return self.preprocess(image, target_size=target_size)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def _extract_roi(self, gray: np.ndarray) -> np.ndarray:
        """Remove dark borders / scanner annotations via Otsu thresholding."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)
        return gray[y1:y2, x1:x2]

    def _anisotropic_diffusion(self, img: np.ndarray) -> np.ndarray:
        """Perona-Malik anisotropic diffusion with kurtosis-based stopping."""
        img_f = img.astype(np.float64) / 255.0
        prev_kurtosis = self._kurtosis(img_f)

        for iteration in range(self.max_iterations):
            # Compute 4-directional gradients
            north = np.roll(img_f, -1, axis=0) - img_f
            south = np.roll(img_f,  1, axis=0) - img_f
            east  = np.roll(img_f, -1, axis=1) - img_f
            west  = np.roll(img_f,  1, axis=1) - img_f

            # Perona-Malik conduction coefficients
            cn = np.exp(-(north / self.kappa) ** 2)
            cs = np.exp(-(south / self.kappa) ** 2)
            ce = np.exp(-(east  / self.kappa) ** 2)
            cw = np.exp(-(west  / self.kappa) ** 2)

            # Diffusion update
            img_f += self.gamma * (cn * north + cs * south + ce * east + cw * west)
            img_f  = np.clip(img_f, 0, 1)

            # Kurtosis convergence check
            curr_kurtosis = self._kurtosis(img_f)
            if abs(curr_kurtosis - prev_kurtosis) <= self.kurtosis_threshold:
                logger.debug("HADF converged at iteration %d", iteration + 1)
                break
            prev_kurtosis = curr_kurtosis

        return (img_f * 255).astype(np.uint8)

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        flat = arr.flatten()
        mu   = flat.mean()
        std  = flat.std()
        if std < 1e-10:
            return 0.0
        return float(np.mean(((flat - mu) / std) ** 4)) - 3.0

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        mn, mx = img.min(), img.max()
        if mx == mn:
            return img
        return ((img - mn) / (mx - mn) * 255).astype(np.uint8)
