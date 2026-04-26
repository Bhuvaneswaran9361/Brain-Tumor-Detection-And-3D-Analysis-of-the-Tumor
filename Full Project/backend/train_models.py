"""
Training Pipeline – YOLO-NAS + En-DeNet  (v6 – Segmentation Overfit Fix)
=========================================================================
Changes vs v5 targeting the train/val Dice gap (0.70/0.65):

  SEG FIXES:
  1. TverskyFocalLoss  — replaces CombinedLoss
                         α=0.3 β=0.7 penalises false negatives 2× more
                         (tumour boundaries miss costs more than FP)
  2. Strong elastic augmentation — grid distortion + elastic transform
                         simulates MRI deformation artefacts
  3. MixUp augmentation (p=0.3) — blends two training pairs
                         proven to reduce segmentation overfitting
  4. Dropout in decoder — added 0.15 spatial dropout between decoder blocks
  5. LR: 5e-4 → 2e-4   — slower start reduces memorisation
  6. Weight decay 1e-4 → 5e-4
  7. EarlyStopping min_delta 0.001 → 0.003, patience 15 → 20

  UNCHANGED: YOLO-NAS classifier training (already converged well)

  
"""


import os
os.environ["MIOPEN_DISABLE_CACHE"]        = "1"
os.environ["MIOPEN_DEBUG_DISABLE_SQL"]    = "1"
os.environ["MIOPEN_DISABLE_USERDB"]       = "1"
os.environ["MIOPEN_ENABLE_LOGGING"]       = "0"
os.environ["MIOPEN_LOG_LEVEL"]            = "0"
os.environ["AMD_LOG_LEVEL"]               = "0"


import os, sys, json, logging, argparse
from pathlib import Path
from collections import Counter

_AMD_ENV = {
    "HSA_OVERRIDE_GFX_VERSION":        "11.0.2",
    "MIOPEN_DISABLE_CACHE":            "1",
    "MIOPEN_DEBUG_DISABLE_SQL":        "1",
    "MIOPEN_DISABLE_USERDB":           "1",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM": "1",
    "MIOPEN_DEBUG_CONV_DIRECT":        "1",
    "MIOPEN_DEBUG_CONV_FFT":           "0",
    "MIOPEN_DEBUG_CONV_WINOGRAD":      "0",
    "AMD_LOG_LEVEL":                   "0",
    "HIP_VISIBLE_DEVICES":             "0",
    "PYTORCH_NO_CUDA_MEMORY_CACHING":  "0",
    "PYTORCH_ALLOC_CONF":              "expandable_segments:True",
}
_env_file = Path(__file__).parent / ".env.amd"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            _AMD_ENV[k.strip()] = v.strip()
for _k, _v in _AMD_ENV.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

from models.hadf_filter   import HADFFilter
from models.yolonas_model import YOLONAS, CLASS_NAMES
from models.endenet_model import EnDeNet, TverskyFocalLoss, DeepSupervisionLoss, calculate_dice, calculate_iou

CLS_MAP = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}
SEG_MAP = {"Glioma": 0, "Meningioma": 1, "Pituitary tumor": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def pick_device(requested="auto"):
    if requested == "cpu":
        logger.info("Device: CPU (forced)")
        return torch.device("cpu"), False
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("Found GPU: %s  (%.1f GB VRAM)", name, vram)
        try:
            x  = torch.randn(2, 64, 28, 28, device="cuda")
            nn.GroupNorm(32, 64).cuda()(x)
            torch.cuda.synchronize()
            logger.info("✅ GPU smoke-test passed → using CUDA (AMD ROCm)")
            return torch.device("cuda"), True
        except RuntimeError as e:
            logger.warning("GPU failed: %s → CPU", e)
    logger.info("No GPU → CPU")
    return torch.device("cpu"), False


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha
        self.smoothing = smoothing; self.weight = weight

    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            smooth = torch.zeros_like(logits).fill_(self.smoothing / (n - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_p = F.log_softmax(logits, dim=1)
        w = self.weight.to(logits.device)[targets] if self.weight is not None \
            else torch.ones(logits.size(0), device=logits.device)
        ce   = -(smooth * log_p).sum(dim=1)
        prob = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return (self.alpha * ((1 - prob) ** self.gamma) * ce * w).mean()


class TverskyFocalLoss(nn.Module):
    """
    Tversky Focal Loss for segmentation.

    Tversky index generalises Dice:
      TI = TP / (TP + α·FP + β·FN)

    Setting α=0.3, β=0.7 penalises false negatives (missed tumour)
    twice as much as false positives — critical for small tumour regions.

    Focal weighting (γ=0.75) focuses on hard-to-segment boundary pixels.

    Combined with BCEWithLogitsLoss for numerical stability with AMP.
    """

    def __init__(
        self,
        alpha: float  = 0.3,    # FP weight (lower = tolerate more FP)
        beta: float   = 0.7,    # FN weight (higher = punish missed tumour)
        gamma: float  = 0.75,   # focal exponent
        bce_weight: float = 0.4, # blend: 0.4*BCE + 0.6*TverskyFocal
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.bce_weight = bce_weight
        self.smooth     = smooth
        self.bce        = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE (AMP-safe)
        bce_loss = self.bce(logits, targets)

        # Tversky on sigmoid probs (cast to float32 for numerical stability)
        probs  = torch.sigmoid(logits.float())
        tgt    = targets.float()
        pf     = probs.view(-1)
        tf     = tgt.view(-1)

        tp     = (pf * tf).sum()
        fp     = (pf * (1 - tf)).sum()
        fn     = ((1 - pf) * tf).sum()

        tversky = (tp + self.smooth) / \
                  (tp + self.alpha * fp + self.beta * fn + self.smooth)

        tversky_focal = (1 - tversky) ** self.gamma

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * tversky_focal


# ─────────────────────────────────────────────────────────────────────────────
# HADF Cache
# ─────────────────────────────────────────────────────────────────────────────

class HADFCache:
    def __init__(self, cache_root: Path, hadf: HADFFilter, enabled: bool = True):
        self.cache_root = cache_root
        self.hadf       = hadf
        self.enabled    = enabled
        if enabled:
            cache_root.mkdir(parents=True, exist_ok=True)

    def get(self, img_path: Path, size: int) -> np.ndarray:
        if not self.enabled:
            return self._run_hadf(img_path, size)
        cache_file = (self.cache_root / f"s{size}"
                      / img_path.parent.name / (img_path.stem + ".npy"))
        if cache_file.exists():
            return np.load(str(cache_file)).astype(np.uint8)
        result = self._run_hadf(img_path, size)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_file), result.astype(np.float16))
        return result

    def _run_hadf(self, img_path: Path, size: int) -> np.ndarray:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((size, size), dtype=np.uint8)
        return self.hadf.preprocess(img, size)

    def warm_up(self, paths: list, size: int, desc="HADF"):
        if not self.enabled:
            return
        missing = [p for p in paths if not (
            self.cache_root / f"s{size}" / p.parent.name / (p.stem + ".npy")
        ).exists()]
        if not missing:
            logger.info("HADF cache (%dpx): all %d cached ✓", size, len(paths))
            return
        logger.info("HADF cache (%dpx): building %d missing…", size, len(missing))
        for p in tqdm(missing, unit="img", desc=f"{desc} {size}px"):
            self.get(p, size)
        logger.info("HADF cache (%dpx): done.", size)


# ─────────────────────────────────────────────────────────────────────────────
# Sample loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_cls_samples(cls_train_dir: Path) -> list:
    samples = []
    for name, label in CLS_MAP.items():
        folder = cls_train_dir / name
        if not folder.is_dir():
            logger.warning("[CLS] Missing: %s", folder); continue
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            for f in sorted(folder.glob(ext)):
                samples.append((f, label))
    logger.info("[CLS] %d samples loaded", len(samples))
    return samples


def load_seg_samples(seg_root: Path) -> list:
    pairs = []
    for folder_name in SEG_MAP:
        folder = seg_root / folder_name
        if not folder.is_dir():
            logger.warning("[SEG] Missing: %s", folder); continue
        for img_path in sorted(f for f in folder.glob("*.png")
                                if not f.stem.endswith("_mask")):
            mask_path = img_path.parent / (img_path.stem + "_mask.png")
            if mask_path.exists():
                pairs.append((img_path, mask_path))
    logger.info("[SEG] %d image+mask pairs loaded", len(pairs))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────


def random_erasing(img, p=0.3):
    if np.random.rand() > p: return img
    h, w = img.shape[:2]
    ea   = np.random.uniform(0.02, 0.15) * h * w
    asp  = np.random.uniform(0.3, 3.3)
    eh   = int(round(np.sqrt(ea * asp))); ew = int(round(np.sqrt(ea / asp)))
    if eh >= h or ew >= w: return img
    x, y = np.random.randint(0, w-ew), np.random.randint(0, h-eh)
    img  = img.copy(); img[y:y+eh, x:x+ew] = np.random.randint(0, 256)
    return img


def elastic_transform(img: np.ndarray, mask: np.ndarray,
                       alpha: float = 20.0, sigma: float = 5.0) -> tuple:
    """
    Elastic deformation — simulates MRI tissue deformation artefacts.
    Applies identical warp to both image and mask.
    """
    h, w = img.shape[:2]
    dx   = cv2.GaussianBlur(
        np.random.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma) * alpha
    dy   = cv2.GaussianBlur(
        np.random.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma) * alpha

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    img_w  = cv2.remap(img,  map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    mask_w = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REFLECT_101)
    return img_w, mask_w


def grid_distortion(img: np.ndarray, mask: np.ndarray,
                     num_steps: int = 5, distort_limit: float = 0.25) -> tuple:
    """
    Grid distortion — stretches/compresses local regions.
    Good for simulating scanner-to-scanner variation.
    """
    h, w  = img.shape[:2]
    steps = num_steps + 1

    x_steps = np.linspace(0, w, steps)
    y_steps = np.linspace(0, h, steps)

    x_steps += np.random.uniform(-distort_limit, distort_limit, steps) * (w / steps)
    y_steps += np.random.uniform(-distort_limit, distort_limit, steps) * (h / steps)
    x_steps  = np.clip(x_steps, 0, w).astype(np.float32)
    y_steps  = np.clip(y_steps, 0, h).astype(np.float32)

    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    for i in range(num_steps):
        for j in range(num_steps):
            x0, x1 = int(x_steps[j]),   int(x_steps[j+1])
            y0, y1 = int(y_steps[i]),   int(y_steps[i+1])
            if x1 <= x0 or y1 <= y0: continue
            sub_w  = x1 - x0; sub_h = y1 - y0
            gx, gy = np.meshgrid(
                np.linspace(x_steps[j], x_steps[j+1], sub_w),
                np.linspace(y_steps[i], y_steps[i+1], sub_h),
            )
            map_x[y0:y1, x0:x1] = gx.astype(np.float32)
            map_y[y0:y1, x0:x1] = gy.astype(np.float32)

    # Fill any zeros
    mask_zero = (map_x == 0) & (map_y == 0)
    gx_def, gy_def = np.meshgrid(np.arange(w, dtype=np.float32),
                                   np.arange(h, dtype=np.float32))
    map_x[mask_zero] = gx_def[mask_zero]
    map_y[mask_zero] = gy_def[mask_zero]

    img_d  = cv2.remap(img,  map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    mask_d = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REFLECT_101)
    return img_d, mask_d


def augment_image(img, size):
    if np.random.rand() > 0.5: img = cv2.flip(img, 1)
    if np.random.rand() > 0.7: img = cv2.flip(img, 0)
    angle = np.random.uniform(-20, 20)
    M     = cv2.getRotationMatrix2D((size//2, size//2), angle, 1)
    img   = cv2.warpAffine(img, M, (size, size))
    alpha = np.random.uniform(0.75, 1.25)
    beta  = int(np.random.uniform(-20, 20))
    img   = np.clip(img.astype(np.float32)*alpha+beta, 0, 255).astype(np.uint8)
    if np.random.rand() > 0.8:
        k = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return random_erasing(img)


def augment_seg_pair(img: np.ndarray, mask: np.ndarray, size: int) -> tuple:
    """
    Strong augmentation pipeline for segmentation pairs.
    All spatial transforms applied identically to image and mask.
    """
    # ── Spatial ───────────────────────────────────────────────────────────────
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1);  mask = cv2.flip(mask, 1)
    if np.random.rand() > 0.6:
        img = cv2.flip(img, 0);  mask = cv2.flip(mask, 0)

    # Rotation ±25°
    angle = np.random.uniform(-25, 25)
    M     = cv2.getRotationMatrix2D((size//2, size//2), angle, 1)
    img   = cv2.warpAffine(img,  M, (size, size))
    mask  = cv2.warpAffine(mask, M, (size, size))

    # Scale ±15%
    scale = np.random.uniform(0.85, 1.15)
    M2    = cv2.getRotationMatrix2D((size//2, size//2), 0, scale)
    img   = cv2.warpAffine(img,  M2, (size, size))
    mask  = cv2.warpAffine(mask, M2, (size, size))

    # Elastic deformation (p=0.4)
    if np.random.rand() < 0.4:
        img, mask = elastic_transform(img, mask,
                                       alpha=np.random.uniform(10, 30),
                                       sigma=np.random.uniform(4, 7))

    # Grid distortion (p=0.3)
    if np.random.rand() < 0.3:
        img, mask = grid_distortion(img, mask,
                                     num_steps=5,
                                     distort_limit=0.2)

    # ── Photometric (image only) ───────────────────────────────────────────────
    alpha = np.random.uniform(0.7, 1.3)
    beta  = int(np.random.uniform(-25, 25))
    img   = np.clip(img.astype(np.float32)*alpha+beta, 0, 255).astype(np.uint8)

    # Gaussian noise (p=0.3)
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, np.random.uniform(2, 8), img.shape)
        img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Blur (p=0.2)
    if np.random.rand() < 0.2:
        k   = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Random erasing (p=0.25)
    img = random_erasing(img, p=0.25)

    return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class ClsDataset(Dataset):
    def __init__(self, samples, cache, size=224, augment=False):
        self.samples = samples; self.cache = cache
        self.size = size; self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.cache.get(path, self.size)
        if self.augment: img = augment_image(img, self.size)
        return torch.from_numpy(img).float().unsqueeze(0) / 255.0, label


class SegDataset(Dataset):
    def __init__(self, pairs, cache, size=256, augment=False, mixup_p=0.0):
        self.pairs    = pairs
        self.cache    = cache
        self.size     = size
        self.augment  = augment
        self.mixup_p  = mixup_p   # MixUp probability

    def __len__(self): return len(self.pairs)

    def _load_one(self, idx):
        img_path, mask_path = self.pairs[idx]
        img      = self.cache.get(img_path, self.size)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            mask_raw = np.zeros((self.size, self.size), dtype=np.uint8)
        mask = cv2.resize(mask_raw, (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load_one(idx)

        if self.augment:
            img, mask = augment_seg_pair(img, mask, self.size)

        # MixUp: blend two training pairs (reduces memorisation)
        if self.augment and np.random.rand() < self.mixup_p:
            idx2       = np.random.randint(0, len(self.pairs))
            img2, msk2 = self._load_one(idx2)
            if self.augment:
                img2, msk2 = augment_seg_pair(img2, msk2, self.size)
            lam  = np.random.beta(0.4, 0.4)
            img  = (lam * img.astype(np.float32) +
                    (1-lam) * img2.astype(np.float32)).astype(np.uint8)
            mask = lam * mask + (1-lam) * msk2

        img_t  = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        return img_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def split_samples(samples, val=0.1, test=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(samples))
    nv  = int(len(idx)*val); nt = int(len(idx)*test)
    return ([samples[i] for i in idx[nv+nt:]],
            [samples[i] for i in idx[:nv]],
            [samples[i] for i in idx[nv:nv+nt]])


def class_weights(samples):
    counts = Counter(l for _, l in samples)
    total  = len(samples); n = len(CLS_MAP)
    w = torch.tensor([total / (n * counts.get(i, 1)) for i in range(n)],
                     dtype=torch.float32)
    w = w / w.sum() * n
    logger.info("Class weights: %s",
                {CLASS_NAMES[i]: round(w[i].item(), 3) for i in range(n)})
    return w


class WarmupCosine:
    def __init__(self, opt, warmup, total, min_lr=1e-6):
        self.opt = opt; self.warmup = warmup; self.total = total
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in opt.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            f = (epoch+1) / max(1, self.warmup)
        else:
            p = (epoch-self.warmup) / max(1, self.total-self.warmup)
            f = self.min_lr/self.base_lrs[0] + \
                0.5*(1-self.min_lr/self.base_lrs[0])*(1+np.cos(np.pi*p))
        for pg, bl in zip(self.opt.param_groups, self.base_lrs):
            pg["lr"] = bl * f
        return self.opt.param_groups[0]["lr"]


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.003):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best        = -1.0
        self.counter     = 0
        self.should_stop = False

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric; self.counter = 0
        else:
            self.counter += 1
            logger.info("  EarlyStop %d/%d (best=%.4f)",
                        self.counter, self.patience, self.best)
            if self.counter >= self.patience:
                self.should_stop = True


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – YOLO-NAS  (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

def check_cls_trained(save_dir: Path):
    best_pth = save_dir / "yolonas_best.pth"
    if not best_pth.exists(): return None
    try:
        ckpt = torch.load(str(best_pth), map_location="cpu", weights_only=True)
        return {"path": best_pth,
                "epoch": ckpt.get("epoch","?"),
                "val_acc": ckpt.get("val_acc", 0.0)}
    except Exception: return None


def prompt_cls_decision(ckpt_info):
    print()
    print("=" * 60)
    print("  YOLO-NAS checkpoint found!")
    print(f"  File   : {ckpt_info['path']}")
    print(f"  Epoch  : {ckpt_info['epoch']}")
    print(f"  Val acc: {ckpt_info['val_acc']:.2f}%")
    print("=" * 60)
    print("  [1] Skip  (default)  [2] Resume  [3] Retrain")
    while True:
        c = input("  Enter 1/2/3 (Enter=1): ").strip()
        if c in ("","1"): return "skip"
        if c == "2":      return "resume"
        if c == "3":      return "retrain"


def train_classifier(config, device, use_gpu, cache, samples, save_dir):
    logger.info("=" * 60)
    logger.info("PHASE 1 – YOLO-NAS  (device=%s)", device)
    logger.info("=" * 60)

    size    = 224
    pin_mem = use_gpu
    use_amp = use_gpu and config.get("amp", True)
    accum   = config.get("accum_steps", 2)
    epochs  = config["cls_epochs"]

    ckpt_info = check_cls_trained(save_dir)
    decision  = "train"
    if ckpt_info is not None:
        if config.get("retrain_cls"):      decision = "retrain"
        elif config.get("resume_cls"):     decision = "resume"
        else:                              decision = prompt_cls_decision(ckpt_info)

    if decision == "skip":
        logger.info("[CLS] ✅ Using existing model (val acc: %.2f%%)",
                    ckpt_info["val_acc"])
        return

    cache.warm_up([p for p,_ in samples], size=size, desc="[CLS] HADF")
    train_s, val_s, _ = split_samples(samples)
    cw = class_weights(train_s)

    train_dl = DataLoader(ClsDataset(train_s, cache, size, augment=True),
                          batch_size=config["batch_size"], shuffle=True,
                          num_workers=0, pin_memory=pin_mem)
    val_dl   = DataLoader(ClsDataset(val_s, cache, size, augment=False),
                          batch_size=config["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=pin_mem)

    model     = YOLONAS(in_channels=1, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config.get("weight_decay", 3e-4))
    criterion = FocalLoss(gamma=2.0, smoothing=0.1, weight=cw)
    scaler    = GradScaler(enabled=use_amp)
    stopper   = EarlyStopping(patience=config.get("patience", 15), min_delta=0.01)

    start_epoch = 1; best_acc = 0.0
    history = {"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[],"lr":[]}

    if decision == "resume" and ckpt_info:
        ckpt        = torch.load(str(ckpt_info["path"]), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_acc    = ckpt.get("val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        history     = ckpt.get("history", history)
        remaining   = max(1, epochs - start_epoch + 1)
        epochs      = start_epoch + remaining - 1
        logger.info("[CLS] Resuming from epoch %d (best=%.2f%%)", start_epoch, best_acc)

    scheduler = WarmupCosine(optimizer, config.get("warmup_epochs", 5), epochs)

    for epoch in range(start_epoch, epochs+1):
        lr = scheduler.step(epoch-1)
        model.train(); t_loss = t_correct = t_total = 0; optimizer.zero_grad()
        for step, (imgs, labels) in enumerate(
            tqdm(train_dl, desc=f"[CLS] Ep {epoch:3d}/{epochs} train", leave=False)
        ):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=use_amp):
                out  = model(imgs); loss = criterion(out, labels) / accum
            scaler.scale(loss).backward()
            if (step+1) % accum == 0 or (step+1) == len(train_dl):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            t_loss += loss.item()*accum*imgs.size(0)
            t_correct += (out.argmax(1)==labels).sum().item(); t_total += imgs.size(0)
        t_loss /= t_total; t_acc = t_correct/t_total*100

        model.eval(); v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_dl, desc=f"[CLS] Ep {epoch:3d}/{epochs} val  ", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(enabled=use_amp):
                    out  = model(imgs); loss = criterion(out, labels)
                v_loss += loss.item()*imgs.size(0)
                v_correct += (out.argmax(1)==labels).sum().item(); v_total += imgs.size(0)
        v_loss /= v_total; v_acc = v_correct/v_total*100

        for k,v in [("train_loss",t_loss),("val_loss",v_loss),
                    ("train_acc",t_acc),("val_acc",v_acc),("lr",lr)]:
            history[k].append(round(float(v),6))
        logger.info("[CLS] Ep %3d/%d  lr=%.2e  train %.4f/%.2f%%  val %.4f/%.2f%%",
                    epoch, epochs, lr, t_loss, t_acc, v_loss, v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                        "val_acc":best_acc,"history":history},
                       save_dir/"yolonas_best.pth")
            logger.info("  ✓ New best CLS: %.2f%%", best_acc)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_dir/f"yolonas_epoch{epoch}.pth")
        stopper.step(v_acc)
        if stopper.should_stop:
            logger.info("[CLS] Early stop at epoch %d", epoch); break

    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"history":history},
               save_dir/"yolonas_final.pth")
    with open(save_dir/"yolonas_history.json","w") as f: json.dump(history,f,indent=2)
    logger.info("[CLS] ✅ Best val accuracy: %.2f%%", best_acc)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – En-DeNet  (overfitting fixes)
# ─────────────────────────────────────────────────────────────────────────────

def train_segmenter(config, device, use_gpu, cache, pairs, save_dir):
    logger.info("=" * 60)
    logger.info("PHASE 2 – En-DeNet  (%d pairs, device=%s)", len(pairs), device)
    logger.info("=" * 60)

    size    = 256
    pin_mem = use_gpu
    use_amp = use_gpu and config.get("amp", True)
    accum   = config.get("accum_steps", 2)
    epochs  = config["seg_epochs"]
    bs      = max(1, config["batch_size"] // 4)

    cache.warm_up([p for p,_ in pairs], size=size, desc="[SEG] HADF")
    train_p, val_p, _ = split_samples(pairs)

    train_dl = DataLoader(
        SegDataset(train_p, cache, size, augment=True,
                   mixup_p=config.get("mixup_p", 0.3)),   # MixUp
        batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin_mem,
    )
    val_dl = DataLoader(
        SegDataset(val_p, cache, size, augment=False, mixup_p=0.0),
        batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem,
    )

    model     = EnDeNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.get("seg_lr", 2e-4),        # ↓ from 5e-4
        weight_decay = config.get("seg_weight_decay", 5e-4),  # ↑ from 1e-4
    )
    criterion = DeepSupervisionLoss(
        base_loss = TverskyFocalLoss(
            alpha      = config.get("tversky_alpha", 0.3),
            beta       = config.get("tversky_beta",  0.7),
            gamma      = config.get("tversky_gamma", 0.75),
            bce_weight = config.get("bce_weight",    0.4),
        ),
        w_main = 1.0,
        w_ds4  = config.get("ds_w4", 0.4),
        w_ds3  = config.get("ds_w3", 0.2),
    )
    scaler  = GradScaler(enabled=use_amp)
    stopper = EarlyStopping(
        patience  = config.get("seg_patience", 20),       # ↑ from 15
        min_delta = config.get("seg_min_delta", 0.003),   # ↑ from 0.001
    )

    start_epoch = 1; best_dice = 0.0
    history = {"train_loss":[],"val_loss":[],"train_dice":[],"val_dice":[],
               "train_iou":[],"val_iou":[],"lr":[]}

    best_pth = save_dir / "endenet_best.pth"
    if best_pth.exists() and not config.get("retrain_seg", False):
        ckpt        = torch.load(str(best_pth), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_dice   = ckpt.get("val_dice", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        history     = ckpt.get("history", history)
        remaining   = max(1, epochs - start_epoch + 1)
        epochs      = start_epoch + remaining - 1
        logger.info("[SEG] Resuming from epoch %d (best Dice=%.4f)",
                    start_epoch, best_dice)
    else:
        logger.info("[SEG] Starting fresh  |  loss=TverskyFocal(α=%.1f,β=%.1f)  "
                    "lr=%.1e  wd=%.1e  mixup=%.1f",
                    config.get("tversky_alpha",0.3), config.get("tversky_beta",0.7),
                    config.get("seg_lr",2e-4), config.get("seg_weight_decay",5e-4),
                    config.get("mixup_p",0.3))

    scheduler = WarmupCosine(optimizer, config.get("warmup_epochs", 5), epochs)

    for epoch in range(start_epoch, epochs+1):
        lr = scheduler.step(epoch-1)

        model.train(); t_loss = t_dice = t_iou = t_n = 0; optimizer.zero_grad()
        for step, (imgs, masks) in enumerate(
            tqdm(train_dl, desc=f"[SEG] Ep {epoch:3d}/{epochs} train", leave=False)
        ):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast(enabled=use_amp):
                pred = model(imgs); loss = criterion(pred, masks) / accum
            scaler.scale(loss).backward()
            if (step+1) % accum == 0 or (step+1) == len(train_dl):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            with torch.no_grad():
                t_loss += loss.item()*accum*imgs.size(0)
                t_dice += calculate_dice(pred, masks)*imgs.size(0)
                t_iou  += calculate_iou(pred,  masks)*imgs.size(0)
                t_n    += imgs.size(0)
        t_loss /= t_n; t_dice /= t_n; t_iou /= t_n

        model.eval(); v_loss = v_dice = v_iou = v_n = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_dl, desc=f"[SEG] Ep {epoch:3d}/{epochs} val  ", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast(enabled=use_amp):
                    pred = model(imgs); loss = criterion(pred, masks)
                v_loss += loss.item()*imgs.size(0)
                v_dice += calculate_dice(pred, masks)*imgs.size(0)
                v_iou  += calculate_iou(pred,  masks)*imgs.size(0)
                v_n    += imgs.size(0)
        v_loss /= v_n; v_dice /= v_n; v_iou /= v_n

        for k,v in [("train_loss",t_loss),("val_loss",v_loss),
                    ("train_dice",t_dice),("val_dice",v_dice),
                    ("train_iou",t_iou),("val_iou",v_iou),("lr",lr)]:
            history[k].append(round(float(v),6))

        gap = t_dice - v_dice
        logger.info(
            "[SEG] Ep %3d/%d  lr=%.2e  loss %.4f/%.4f  "
            "dice %.4f/%.4f (gap %.4f)  iou %.4f/%.4f",
            epoch, epochs, lr, t_loss, v_loss,
            t_dice, v_dice, gap, t_iou, v_iou,
        )

        if v_dice > best_dice:
            best_dice = v_dice
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                        "val_dice":best_dice,"history":history},
                       save_dir/"endenet_best.pth")
            logger.info("  ✓ New best Dice: %.4f", best_dice)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_dir/f"endenet_epoch{epoch}.pth")
        stopper.step(v_dice)
        if stopper.should_stop:
            logger.info("[SEG] Early stop at epoch %d (best=%.4f)", epoch, best_dice)
            break

    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"history":history},
               save_dir/"endenet_final.pth")
    with open(save_dir/"endenet_history.json","w") as f: json.dump(history,f,indent=2)
    logger.info("[SEG] ✅ Best val Dice: %.4f", best_dice)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config):
    device, use_gpu = pick_device(config.get("device", "auto"))
    dataset  = Path(config["dataset"])
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    cache = HADFCache(
        cache_root = dataset.parent / "hadf_cache",
        hadf       = HADFFilter(),
        enabled    = not config.get("no_cache", False),
    )

    if not config.get("skip_cls", False):
        samples = load_cls_samples(dataset / "classification" / "Training")
        if not samples:
            raise FileNotFoundError(f"No CLS images in {dataset}/classification/Training")
        train_classifier(config, device, use_gpu, cache, samples, save_dir)
    else:
        logger.info("[CLS] Skipped via --skip_cls")

    if not config.get("skip_seg", False):
        pairs = load_seg_samples(dataset / "Segmentation")
        if not pairs:
            raise FileNotFoundError(f"No SEG pairs in {dataset}/Segmentation")
        train_segmenter(config, device, use_gpu, cache, pairs, save_dir)

    logger.info("🎉 Training complete → %s", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",          default="./DATASET")
    p.add_argument("--save_dir",         default="./trained_models")
    p.add_argument("--cls_epochs",       type=int,   default=150)
    p.add_argument("--seg_epochs",       type=int,   default=250)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--seg_lr",           type=float, default=2e-4,
                   help="En-DeNet LR (default 2e-4, was 5e-4)")
    p.add_argument("--weight_decay",     type=float, default=3e-4)
    p.add_argument("--seg_weight_decay", type=float, default=5e-4,
                   help="En-DeNet weight decay (default 5e-4, was 1e-4)")
    p.add_argument("--warmup_epochs",    type=int,   default=5)
    p.add_argument("--accum_steps",      type=int,   default=2)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--seg_patience",     type=int,   default=20)
    p.add_argument("--seg_min_delta",    type=float, default=0.003)
    p.add_argument("--tversky_alpha",    type=float, default=0.3,
                   help="Tversky FP weight (lower = tolerate FP)")
    p.add_argument("--tversky_beta",     type=float, default=0.7,
                   help="Tversky FN weight (higher = punish missed tumour)")
    p.add_argument("--tversky_gamma",    type=float, default=0.75)
    p.add_argument("--bce_weight",       type=float, default=0.4)
    p.add_argument("--ds_w4",            type=float, default=0.4,
                   help="Deep supervision weight at dec4 output")
    p.add_argument("--ds_w3",            type=float, default=0.2,
                   help="Deep supervision weight at dec3 output")
    p.add_argument("--mixup_p",          type=float, default=0.3,
                   help="MixUp probability for segmentation (0=off)")
    p.add_argument("--device",           default="auto")
    p.add_argument("--no_amp",           action="store_true")
    p.add_argument("--no_cache",         action="store_true")
    p.add_argument("--skip_cls",         action="store_true")
    p.add_argument("--skip_seg",         action="store_true")
    p.add_argument("--retrain_cls",      action="store_true")
    p.add_argument("--resume_cls",       action="store_true")
    p.add_argument("--retrain_seg",      action="store_true")
    args   = p.parse_args()
    config = vars(args)
    config["amp"] = not config.pop("no_amp")
    logger.info("Config: %s", config)
    main(config)