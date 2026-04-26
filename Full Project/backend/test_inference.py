"""
test_inference.py
==================
Place in:  backend/   (NOT inside backend/models/)

Run:
    cd backend
    python test_inference.py                          # synthetic image
    python test_inference.py --image path/to/mri.png  # real image
    python test_inference.py --demo                   # no weights needed
"""

import os
os.environ["MIOPEN_DISABLE_CACHE"]        = "1"
os.environ["MIOPEN_DEBUG_DISABLE_SQL"]    = "1"
os.environ["MIOPEN_DISABLE_USERDB"]       = "1"
os.environ["MIOPEN_ENABLE_LOGGING"]       = "0"
os.environ["MIOPEN_LOG_LEVEL"]            = "0"
os.environ["AMD_LOG_LEVEL"]               = "0"


import argparse, sys, os
from pathlib import Path

# ── Must run from backend/ ────────────────────────────────────────────────────
backend_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

import cv2, numpy as np, base64, shutil

p = argparse.ArgumentParser()
p.add_argument("--image",     default=None)
p.add_argument("--model_dir", default="./trained_models")
p.add_argument("--demo",      action="store_true")
args = p.parse_args()

# ── Image ─────────────────────────────────────────────────────────────────────
def synthetic_mri(size=224):
    img = np.zeros((size,size), np.uint8)
    cv2.ellipse(img,(size//2,size//2),(int(size*.42),int(size*.38)),0,0,360,180,-1)
    cv2.ellipse(img,(int(size*.55),int(size*.45)),(int(size*.10),int(size*.08)),0,0,360,240,-1)
    noise = np.random.RandomState(42).randint(0,30,(size,size)).astype(np.uint8)
    return cv2.GaussianBlur(cv2.add(img,noise),(5,5),0)

if args.image:
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Cannot read: {args.image}"); sys.exit(1)
    print(f"✅ Loaded: {args.image}  shape={image.shape}")
else:
    image = synthetic_mri()
    print(f"✅ Using synthetic test MRI  shape={image.shape}")

# ── Pipeline ──────────────────────────────────────────────────────────────────
print("\nLoading pipeline...")
from models.inference import BrainTumorPipeline

pipeline = BrainTumorPipeline(
    model_dir="./dummy_no_weights" if args.demo else args.model_dir)
print(f"  Demo mode : {pipeline.demo_mode}")
print(f"  Device    : {pipeline.device}")

# ── Infer ─────────────────────────────────────────────────────────────────────
print("\nRunning inference...")
result = pipeline.predict(image)

# ── Results ───────────────────────────────────────────────────────────────────
cls = result.get("classification", {})
seg = result.get("segmentation",   {})
t3d = result.get("tumor_3d",       {})

print()
print("=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"  Processing time  : {result.get('processing_time',0):.3f}s")
print(f"  Demo mode        : {result.get('demo_mode')}")
print()
print(f"  Tumour class     : {cls.get('class_name')}")
print(f"  Confidence       : {cls.get('confidence',0):.2f}%")
print(f"  Probabilities    :")
for name, prob in (cls.get("probabilities") or {}).items():
    bar = "█"*int(prob/5) + "░"*(20-int(prob/5))
    print(f"    {name:<14} {prob:5.1f}%  {bar}")

print()
print(f"  Tumour detected  : {seg.get('tumor_detected')}")
print(f"  Area (mm²)       : {t3d.get('volume_mm2',0):.2f}")
print(f"  Volume (mm³)     : {t3d.get('volume_mm3',0):.2f}")
if t3d.get("bounding_box"):
    x,y,w,h = t3d["bounding_box"]
    print(f"  Bounding box     : x={x} y={y} w={w} h={h}")
if t3d.get("centroid"):
    print(f"  Centroid         : {t3d['centroid']}")
if t3d.get("shape_metrics"):
    print(f"  Shape metrics    :")
    for k,v in t3d["shape_metrics"].items():
        print(f"    {k:<16} {v:.4f}")

print()
print("  Outputs          :")
checks = [
    ("hadf_image_b64",           "HADF preprocessed"),
    ("overlay_image_b64",        "Detection overlay"),
    ("gradcam_b64",              "Grad-CAM heatmap"),
    ("gradcam_overlay_b64",      "Grad-CAM overlay"),
    ("seg_gradcam_b64",          "SegGrad-CAM heatmap"),
    ("seg_gradcam_overlay_b64",  "SegGrad-CAM overlay"),
    ("eigencam_b64",             "EigenCAM heatmap"),
    ("eigencam_overlay_b64",     "EigenCAM overlay"),
]
for key, label in checks:
    v = result.get(key)
    print(f"    {'✅' if v else '⚠ '} {label:<30} "
          f"{'(' + str(len(v)//1024) + ' KB)' if v else 'not available'}")

fig_ok = t3d.get("figure_3d") not in (None, {})
print(f"    {'✅' if fig_ok else '⚠ '} 3D Plotly figure              "
      f"{'ready' if fig_ok else 'not available (pip install plotly scikit-image scipy)'}")
print(f"    {'✅' if t3d.get('sampled_points') else '⚠ '} "
      f"Sampled 3D points             {len(t3d.get('sampled_points',[]))}")

# ── Save images ───────────────────────────────────────────────────────────────
out_dir = backend_dir / "inference_test_output"
out_dir.mkdir(exist_ok=True)
saved = []
for key, fname in [
    ("hadf_image_b64",            "01_hadf.png"),
    ("overlay_image_b64",         "02_overlay.png"),
    ("gradcam_overlay_b64",       "03_gradcam_detection.png"),
    ("seg_gradcam_overlay_b64",   "04_gradcam_segmentation.png"),
    ("eigencam_overlay_b64",      "05_eigencam.png"),
]:
    b64 = result.get(key)
    if b64:
        (out_dir/fname).write_bytes(base64.b64decode(b64))
        saved.append(fname)

# segmentation outputs
for sub_key, img_key, fname in [
    ("segmentation","mask_base64",    "06_seg_mask.png"),
    ("segmentation","overlay_base64", "07_seg_overlay.png"),
]:
    b64 = (result.get(sub_key) or {}).get(img_key)
    if b64:
        (out_dir/fname).write_bytes(base64.b64decode(b64))
        saved.append(fname)

print()
print(f"  Saved {len(saved)} images → {out_dir}/")
for s in saved: print(f"    {s}")

# ── PDF report ────────────────────────────────────────────────────────────────
print()
print("  Generating PDF report...")
try:
    import reportlab
    result2 = pipeline.predict(image, generate_report=True,
                                patient_id="TEST-001",
                                patient_name="Test Patient")
    rp = result2.get("report_path")
    if rp and Path(rp).exists():
        dest = out_dir/"report.pdf"
        shutil.copy(rp, dest)
        size_kb = dest.stat().st_size // 1024
        print(f" ✅ PDF report saved: {dest}  ({size_kb} KB)")
    else:
        print(f"  ⚠  Report error: {result2.get('report_error','unknown')}")
except ImportError:
    print("  ⚠  reportlab not installed.")
    print("     Install: pip install reportlab")
except Exception as e:
    print(f"  ⚠  Report failed: {e}")

print()
print("=" * 55)
print("  ✅  Inference test complete")
print("=" * 55)