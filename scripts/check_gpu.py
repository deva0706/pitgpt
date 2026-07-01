"""
check_gpu.py — PitGPT environment verification gate.

Run this AFTER installing dependencies:

    python scripts/check_gpu.py

It answers three questions, in order:
  1. Can PyTorch see CUDA and your GPU?
  2. Does a real tensor operation run on the GPU?
  3. Is bitsandbytes (4-bit / QLoRA) importable?

If all three pass, your environment is ready for training.
"""

import sys


def main() -> int:
    # ---- 1. Is PyTorch even installed? ----
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed. Install it before anything else.")
        return 1

    print(f"[info] PyTorch version : {torch.__version__}")

    # ---- 2. Does this build of PyTorch include CUDA support? ----
    cuda_ok = torch.cuda.is_available()
    print(f"[info] CUDA available  : {cuda_ok}")

    if not cuda_ok:
        # A CPU-only wheel, or a driver problem. Training will be painfully slow.
        print("[FAIL] PyTorch cannot see a CUDA GPU.")
        print("       Likely cause: you installed the CPU-only torch build.")
        print("       Fix: reinstall with the cu121 index-url (see README).")
        return 1

    # ---- 3. Report the CUDA runtime bundled inside the torch wheel ----
    print(f"[info] CUDA runtime    : {torch.version.cuda}")

    # ---- 4. Identify the GPU and its VRAM ----
    device_index = torch.cuda.current_device()          # which GPU (0 on a single-GPU laptop)
    name = torch.cuda.get_device_name(device_index)      # human-readable model name
    major, minor = torch.cuda.get_device_capability(device_index)  # compute capability
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    total_gb = total_bytes / (1024 ** 3)                 # bytes -> gigabytes

    print(f"[info] GPU             : {name}")
    print(f"[info] Compute capab.  : {major}.{minor}")
    print(f"[info] Total VRAM      : {total_gb:.2f} GB")

    # ---- 5. Prove we can actually compute on the GPU ----
    #     nvidia-smi proves the driver works; THIS proves PyTorch works.
    try:
        x = torch.rand(1000, 1000, device="cuda")        # allocate a matrix on the GPU
        y = x @ x                                         # matrix multiply on the GPU
        torch.cuda.synchronize()                          # wait for the GPU to finish
        print(f"[ OK ] GPU compute test passed (result sum = {y.sum().item():.1f})")
    except Exception as exc:                              # noqa: BLE001 - report any GPU error
        print(f"[FAIL] GPU compute test raised: {exc}")
        return 1

    # ---- 6. Confirm bitsandbytes (the 4-bit engine for QLoRA) imports ----
    try:
        import bitsandbytes as bnb
        print(f"[ OK ] bitsandbytes    : {bnb.__version__} (4-bit / QLoRA ready)")
    except Exception as exc:                              # noqa: BLE001 - common on Windows
        print(f"[WARN] bitsandbytes import failed: {exc}")
        print("       4-bit training may not work on this machine (see README Windows note).")
        # Not a hard failure: inference and plain LoRA can still run.

    print("\n[done] Environment verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
