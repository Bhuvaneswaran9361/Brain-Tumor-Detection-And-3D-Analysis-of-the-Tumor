from .hadf_filter        import HADFFilter
from .yolonas_model      import YOLONAS, FocalLoss, CLASS_NAMES, CLASS_COLORS
from .endenet_model      import (EnDeNet, CombinedLoss, TverskyFocalLoss,
                                  DeepSupervisionLoss,
                                  calculate_dice, calculate_iou)
from .gradcam            import GradCAM, SegGradCAM, EigenCAM
from .reconstruction_3d  import TumourReconstructor3D
from .report_generator   import generate_report
from .inference          import BrainTumorPipeline

__all__ = [
    # Preprocessing
    "HADFFilter",

    # Classification model
    "YOLONAS", "FocalLoss", "CLASS_NAMES", "CLASS_COLORS",

    # Segmentation model
    "EnDeNet",
    "CombinedLoss",          # backward-compat alias
    "TverskyFocalLoss",      # primary loss (v3)
    "DeepSupervisionLoss",   # wraps TverskyFocalLoss with deep supervision
    "calculate_dice",
    "calculate_iou",

    # Explainability
    "GradCAM",               # Grad-CAM for YOLO-NAS classifier
    "SegGradCAM",            # Grad-CAM++ for En-DeNet segmenter
    "EigenCAM",              # Gradient-free EigenCAM

    # 3D reconstruction
    "TumourReconstructor3D", # mask → 3D volume → Plotly surface

    # Report
    "generate_report",       # clinical PDF report generator

    # Pipeline
    "BrainTumorPipeline",    # full inference orchestrator
]
