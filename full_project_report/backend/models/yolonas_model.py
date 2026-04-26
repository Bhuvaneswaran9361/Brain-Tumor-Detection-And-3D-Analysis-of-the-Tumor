"""
YOLO-NAS Brain Tumor Classifier  (AMD/ROCm-safe version)
=========================================================
All normalisation uses GroupNorm — works on AMD ROCm, NVIDIA CUDA,
Apple MPS, and CPU. No MIOpen JIT compilation required.

Classes: Glioma(0)  Meningioma(1)  Pituitary(2)  No Tumor(3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES  = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CLASS_COLORS = ["#FF4444", "#FF8C00", "#FFD700", "#44CC44"]


# ── Normalisation helper ──────────────────────────────────────────────────────
def _norm(channels: int) -> nn.Module:
    """GroupNorm with up to 32 groups. Divides channels evenly."""
    groups = min(32, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


# ── Blocks ───────────────────────────────────────────────────────────────────
class ConvNormAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            _norm(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class QARepVGGBlock(nn.Module):
    """Quantization-Aware Re-parameterizable VGG block with GroupNorm."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.use_identity = (in_c == out_c and stride == 1)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False), _norm(out_c))
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False), _norm(out_c))
        if self.use_identity:
            self.branch_id = _norm(in_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.use_identity:
            out = out + self.branch_id(x)
        return self.act(out)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling – Fast."""
    def __init__(self, in_c, out_c, pool_size=5):
        super().__init__()
        mid_c     = in_c // 2
        self.cv1  = ConvNormAct(in_c,      mid_c, k=1, p=0)
        self.cv2  = ConvNormAct(mid_c * 4, out_c, k=1, p=0)
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid      = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.sig(self.mlp(self.avg(x))) + self.sig(self.mlp(self.max(x)))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max( dim=1, keepdim=True).values
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


# ── Full YOLO-NAS ─────────────────────────────────────────────────────────────
class YOLONAS(nn.Module):
    """
    Input : (B, 1, 224, 224)
    Output: (B, 4) class logits
    """
    def __init__(self, in_channels=1, num_classes=4, dropout=0.3):
        super().__init__()
        self.stem   = nn.Sequential(
            ConvNormAct(in_channels, 32, k=3, s=2, p=1),
            ConvNormAct(32, 64, k=3, s=1, p=1),
        )
        self.stage1 = self._make_stage(64,  128, n=3, stride=2)
        self.stage2 = self._make_stage(128, 256, n=4, stride=2)
        self.stage3 = self._make_stage(256, 512, n=6, stride=2)
        self.sppf   = SPPF(512, 512)
        self.cbam   = CBAM(512)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _make_stage(in_c, out_c, n, stride):
        layers = [QARepVGGBlock(in_c, out_c, stride=stride)]
        for _ in range(n - 1):
            layers.append(QARepVGGBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.sppf(x)
        x = self.cbam(x)
        x = self.gap(x)
        return self.head(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        self.eval()
        probs = torch.softmax(self(x), dim=-1)[0]
        idx   = probs.argmax().item()
        return {
            "class_index":   idx,
            "class_name":    CLASS_NAMES[idx],
            "class_color":   CLASS_COLORS[idx],
            "confidence":    round(probs[idx].item() * 100, 2),
            "probabilities": {n: round(probs[i].item() * 100, 2)
                              for i, n in enumerate(CLASS_NAMES)},
        }


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce
        return loss.mean()