"""
GPU Test Script
===============
Checks whether CUDA GPU is available and prints detailed info.
Run: python gpu_test.py
"""

import sys

print("=" * 50)
print("  GPU / CUDA AVAILABILITY TEST")
print("=" * 50)

# ── Python version ──────────────────────────────────
print(f"\n Python : {sys.version.split()[0]}")

# ── PyTorch ─────────────────────────────────────────
try:
    import torch
    print(f" PyTorch: {torch.__version__}")
except ImportError:
    print(" PyTorch: NOT INSTALLED  →  pip install torch")
    sys.exit(1)

print()

# ── CUDA availability ────────────────────────────────
cuda_available = torch.cuda.is_available()
print(f" CUDA Available     : {'✅ YES' if cuda_available else '❌ NO'}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f" GPU Count          : {device_count}")

    for i in range(device_count):
        name       = torch.cuda.get_device_name(i)
        total_mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        capability = torch.cuda.get_device_capability(i)
        print(f"\n --- GPU {i} ---")
        print(f"   Name        : {name}")
        print(f"   VRAM        : {total_mem:.1f} GB")
        print(f"   Capability  : sm_{capability[0]}{capability[1]}")

    print(f"\n CUDA Version       : {torch.version.cuda}")
    print(f" cuDNN Version      : {torch.backends.cudnn.version()}")
    print(f" cuDNN Enabled      : {torch.backends.cudnn.enabled}")

    # ── Quick tensor computation test ────────────────
    print("\n Running GPU tensor test...")
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f" Matrix multiply (1000×1000) : ✅ PASSED")
    print(f" Result shape               : {list(c.shape)}")

    # Memory stats
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved()  / 1024**2
    print(f" Memory Allocated           : {allocated:.1f} MB")
    print(f" Memory Reserved            : {reserved:.1f} MB")

else:
    print("\n CUDA is NOT available. Possible reasons:")
    print("   1. No NVIDIA GPU in this machine")
    print("   2. CUDA toolkit not installed")
    print("   3. PyTorch installed without CUDA support")
    print("\n To install PyTorch with CUDA support:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

# ── MPS (Apple Silicon) ──────────────────────────────
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"\n MPS (Apple Silicon): {'✅ YES' if mps_available else '❌ NO'}")

# ── Active device summary ────────────────────────────
if cuda_available:
    active = "cuda"
elif mps_available:
    active = "mps"
else:
    active = "cpu"

print(f"\n{'=' * 50}")
print(f"  ACTIVE DEVICE  →  {active.upper()}")
print(f"{'=' * 50}\n")
