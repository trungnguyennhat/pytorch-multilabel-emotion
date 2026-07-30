"""
Smoke check: xác nhận PyTorch nhận GPU và in environment summary.
Chạy: python check_env.py
"""
import sys
import torch

print("=" * 50)
print("ENVIRONMENT SUMMARY")
print("=" * 50)
print(f"Python       : {sys.version}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version : {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU name     : {gpu_name}")
    print(f"GPU memory   : {gpu_mem:.1f} GB")
    
    # Quick tensor test trên GPU
    x = torch.randn(2, 3, device="cuda")
    print(f"\nTest tensor on GPU: shape={x.shape}, device={x.device}")
else:
    raise SystemExit("CUDA NOT AVAILABLE — check PyTorch installation")

print("=" * 50)

