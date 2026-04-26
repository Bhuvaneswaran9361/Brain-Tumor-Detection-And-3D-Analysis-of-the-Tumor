"""
En-DeNet v3 – High-Accuracy Segmentation Network
==================================================
AMD/ROCm-safe: GroupNorm only, no BatchNorm2d.
AMP-safe:      no Sigmoid in forward — BCEWithLogitsLoss handles it.

Improvements over v2 targeting higher Dice/IoU:

  1. ASPP bottleneck  — Atrous Spatial Pyramid Pooling captures
                        multi-scale context (rates 1,3,6,12)
                        Replaces flat 512→512 DoubleConv

  2. Attention Gates  — Soft spatial attention on skip connections
                        suppresses irrelevant activations from healthy tissue
                        before they enter the decoder

  3. Residual encoder blocks  — skip connection within each encoder block
                                 stabilises deeper training

  4. Deep supervision         — auxiliary loss heads at dec3 + dec4
                                 backpropagate gradients to earlier layers
                                 (handled in CombinedLossDS)

  5. Dropout2d in decoder     — spatial dropout p=0.15 between decoder
                                 blocks reduces co-adaptation

  6. CBAM in bottleneck       — channel + spatial attention after ASPP

Memory budget (256×256, batch=4, 8 GB VRAM):
  Gradient checkpointing on encoder → ~3.8 GB peak
  Bottleneck channels: 512 (unchanged)

Input : (B, 1, 256, 256)
Output: (B, 1, 256, 256) raw logits  (apply sigmoid for inference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ── Norm factory ──────────────────────────────────────────────────────────────
def _norm(c: int) -> nn.Module:
    g = min(32, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ResDoubleConv(nn.Module):
    """
    Residual double convolution block.
    Adds a 1×1 projection skip if channels change — stabilises
    gradient flow in deeper encoder stages.
    """
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            _norm(out_c), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            _norm(out_c), nn.ReLU(inplace=True),
        )
        # Projection shortcut when channel count changes
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            _norm(out_c),
        ) if in_c != out_c else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.proj(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int,
                 use_checkpoint: bool = True, dropout: float = 0.0):
        super().__init__()
        self.conv           = ResDoubleConv(in_c, out_c, dropout)
        self.pool           = nn.MaxPool2d(2)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint and self.training:
            skip = checkpoint(self.conv, x, use_reentrant=False)
        else:
            skip = self.conv(x)
        return self.pool(skip), skip


# ─────────────────────────────────────────────────────────────────────────────
# ASPP Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class ASPPConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, rate: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=rate, dilation=rate, bias=False),
            _norm(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            _norm(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        size = x.shape[2:]
        return F.interpolate(self.block(x), size=size,
                             mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures tumour context at 4 different scales simultaneously:
      rate=1  : local fine detail
      rate=3  : small tumour core
      rate=6  : medium peritumoral region
      rate=12 : large whole-tumour context
    """
    def __init__(self, in_c: int = 512, out_c: int = 512,
                 rates: tuple = (1, 3, 6, 12)):
        super().__init__()
        mid = out_c // (len(rates) + 2)
        self.branches = nn.ModuleList([
            ASPPConv(in_c, mid, r) for r in rates
        ])
        self.pool    = ASPPPooling(in_c, mid)
        self.point   = nn.Sequential(
            nn.Conv2d(in_c, mid, 1, bias=False), _norm(mid), nn.ReLU(inplace=True)
        )
        total_mid = mid * (len(rates) + 2)
        self.proj = nn.Sequential(
            nn.Conv2d(total_mid, out_c, 1, bias=False),
            _norm(out_c), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.branches]
        parts.append(self.pool(x))
        parts.append(self.point(x))
        return self.proj(torch.cat(parts, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# CBAM — Channel + Spatial Attention
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        mid = max(1, c // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(c, mid), nn.ReLU(inplace=True), nn.Linear(mid, c),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        m = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        return x * self.sig(self.conv(m))


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.ch = ChannelAttention(c, r)
        self.sp = SpatialAttention()

    def forward(self, x): return self.sp(self.ch(x))


# ─────────────────────────────────────────────────────────────────────────────
# Attention Gate (for skip connections)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate on skip connections.
    Uses gating signal from decoder to highlight tumour-relevant
    regions in the encoder feature map before concatenation.

    g  : gating signal from decoder (coarser, deeper)
    x  : skip connection from encoder (finer, shallower)
    """
    def __init__(self, g_c: int, x_c: int, inter_c: int):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_c, inter_c, 1, bias=False), _norm(inter_c))
        self.Wx = nn.Sequential(
            nn.Conv2d(x_c, inter_c, 1, bias=False), _norm(inter_c))
        self.psi = nn.Sequential(
            nn.Conv2d(inter_c, 1, 1, bias=False),
            _norm(1), nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Upsample gate to match skip resolution
        g_up = F.interpolate(g, size=x.shape[2:],
                             mode="bilinear", align_corners=False)
        attn = self.psi(self.relu(self.Wg(g_up) + self.Wx(x)))
        return x * attn


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class DecBlock(nn.Module):
    """
    Upsampling decoder block with:
      - Attention gate on skip connection
      - Depthwise-separable convolutions (memory efficient)
      - SE channel attention on output
      - Optional spatial dropout
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int,
                 dropout: float = 0.15):
        super().__init__()
        self.up     = nn.ConvTranspose2d(in_c, in_c // 2, 2, stride=2)
        self.ag     = AttentionGate(in_c // 2, skip_c, inter_c=skip_c // 2)
        merged      = in_c // 2 + skip_c
        self.conv   = nn.Sequential(
            # Depthwise separable
            nn.Conv2d(merged, merged, 3, padding=1, groups=merged, bias=False),
            nn.Conv2d(merged, out_c,  1, bias=False),
            _norm(out_c), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            _norm(out_c), nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_c)

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                               mode="bilinear", align_corners=False)
        skip = self.ag(x, skip)              # attention-gated skip
        x    = torch.cat([x, skip], dim=1)
        return self.se(self.conv(x))


class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(1, c // r)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, c), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Deep Supervision Head
# ─────────────────────────────────────────────────────────────────────────────

class DSHead(nn.Module):
    """Auxiliary output head for deep supervision at intermediate scales."""
    def __init__(self, in_c: int, out_c: int = 1):
        super().__init__()
        self.head = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x: torch.Tensor,
                target_size: tuple) -> torch.Tensor:
        out = self.head(x)
        return F.interpolate(out, size=target_size,
                             mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Full En-DeNet v3
# ─────────────────────────────────────────────────────────────────────────────

class EnDeNet(nn.Module):
    """
    En-DeNet v3 — high-accuracy brain tumour segmentation.

    Input  : (B, 1, 256, 256)
    Output : (B, 1, 256, 256) raw logits

    During training with deep supervision, returns
    (main_logits, ds3_logits, ds4_logits) when self.training=True.
    During inference, returns only main_logits.
    """

    def __init__(
        self,
        in_channels:     int   = 1,
        out_channels:    int   = 1,
        use_checkpoint:  bool  = True,
        deep_supervision: bool = True,
        enc_dropout:     float = 0.0,
        dec_dropout:     float = 0.15,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ── Encoder (residual blocks) ──────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, 64,  use_checkpoint, enc_dropout)
        self.enc2 = EncoderBlock(64,  128, use_checkpoint, enc_dropout)
        self.enc3 = EncoderBlock(128, 256, use_checkpoint, enc_dropout)
        self.enc4 = EncoderBlock(256, 512, use_checkpoint, enc_dropout)

        # ── ASPP Bottleneck + CBAM ─────────────────────────────────────────────
        self.bottleneck = ASPP(512, 512, rates=(1, 3, 6, 12))
        self.cbam       = CBAM(512)

        # ── Decoder (attention gates + SE + dropout) ───────────────────────────
        self.dec4 = DecBlock(512, 512, 256, dec_dropout)
        self.dec3 = DecBlock(256, 256, 128, dec_dropout)
        self.dec2 = DecBlock(128, 128,  64, dec_dropout)
        self.dec1 = DecBlock( 64,  64,  32, dec_dropout)

        # ── Main output head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            _norm(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
            # No sigmoid — BCEWithLogitsLoss / TverskyFocalLoss handle it
        )

        # ── Deep supervision heads (auxiliary outputs at dec3 + dec4) ──────────
        if deep_supervision:
            self.ds_head4 = DSHead(256, out_channels)  # after dec4
            self.ds_head3 = DSHead(128, out_channels)  # after dec3

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Encoder
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.cbam(x)

        # Decoder
        d4 = self.dec4(x,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        main = self.head(d1)

        # Deep supervision (only during training)
        if self.deep_supervision and self.training:
            ds4 = self.ds_head4(d4, (H, W))
            ds3 = self.ds_head3(d3, (H, W))
            return main, ds4, ds3

        return main


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class TverskyFocalLoss(nn.Module):
    """
    Tversky Focal Loss — primary segmentation loss.

    TverskyIndex = TP / (TP + α·FP + β·FN)
    α=0.3, β=0.7 → penalises missed tumour pixels 2.3× more than FP.
    Focal exponent γ=0.75 focuses training on hard boundary pixels.
    Blended with BCEWithLogitsLoss (AMP-safe).
    """

    def __init__(
        self,
        alpha: float      = 0.3,
        beta: float       = 0.7,
        gamma: float      = 0.75,
        bce_weight: float = 0.4,
        smooth: float     = 1.0,
    ):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.bce_weight = bce_weight
        self.smooth     = smooth
        self.bce        = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce  = self.bce(logits, targets)
        prob = torch.sigmoid(logits.float())
        pf   = prob.view(-1);   tf = targets.float().view(-1)
        tp   = (pf * tf).sum()
        fp   = (pf * (1 - tf)).sum()
        fn   = ((1 - pf) * tf).sum()
        tversky = (tp + self.smooth) / \
                  (tp + self.alpha*fp + self.beta*fn + self.smooth)
        tfl     = (1 - tversky) ** self.gamma
        return self.bce_weight * bce + (1 - self.bce_weight) * tfl


class DeepSupervisionLoss(nn.Module):
    """
    Wraps TverskyFocalLoss with deep supervision.

    Loss = w_main·L(main) + w_ds4·L(ds4) + w_ds3·L(ds3)

    Auxiliary weights decay so the main head dominates:
      main=1.0, ds4=0.4, ds3=0.2
    """

    def __init__(
        self,
        base_loss:   nn.Module = None,
        w_main:      float     = 1.0,
        w_ds4:       float     = 0.4,
        w_ds3:       float     = 0.2,
    ):
        super().__init__()
        self.loss   = base_loss or TverskyFocalLoss()
        self.w_main = w_main
        self.w_ds4  = w_ds4
        self.w_ds3  = w_ds3

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            main, ds4, ds3 = outputs
            return (self.w_main * self.loss(main, targets) +
                    self.w_ds4  * self.loss(ds4,  targets) +
                    self.w_ds3  * self.loss(ds3,  targets))
        # Inference mode — single output
        return self.loss(outputs, targets)


# Keep CombinedLoss for backward compatibility with checkpoints
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.4, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight; self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce  = self.bce(logits, target)
        prob = torch.sigmoid(logits.float())
        pf   = prob.view(-1); tf = target.float().view(-1)
        dice = 1 - (2*(pf*tf).sum() + self.smooth)/(pf.sum()+tf.sum()+self.smooth)
        return bce + self.dice_weight * dice


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_dice(logits, target, threshold=0.5):
    # Handle deep supervision output tuple
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    prob = torch.sigmoid(logits.float())
    pb   = (prob > threshold).float()
    tgt  = target.float()
    return (2*(pb*tgt).sum() / (pb.sum() + tgt.sum() + 1e-8)).item()


def calculate_iou(logits, target, threshold=0.5):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    prob  = torch.sigmoid(logits.float())
    pb    = (prob > threshold).float()
    tgt   = target.float()
    inter = (pb * tgt).sum()
    union = pb.sum() + tgt.sum() - inter
    return (inter / (union + 1e-8)).item()