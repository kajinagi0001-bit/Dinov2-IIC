from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def main() -> int:
    print("=== Python ===")
    print(f"executable: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print()

    print("=== nvidia-smi ===")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("nvidia-smi: not found on PATH")
    else:
        print(f"nvidia-smi: {nvidia_smi}")
        run(
            [
                nvidia_smi,
                "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    print()

    print("=== PyTorch CUDA ===")
    try:
        import torch
    except Exception as exc:
        print(f"torch import: failed ({type(exc).__name__}: {exc})")
        return 1

    print(f"torch version: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("result: GPU is not available to PyTorch in this environment")
        return 2

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        print(
            f"device {index}: {props.name}, capability={props.major}.{props.minor}, "
            f"memory={total_gb:.2f} GiB"
        )

    print()
    print("=== CUDA Tensor Smoke Test ===")
    device = torch.device("cuda:0")
    x = torch.randn((2048, 2048), device=device)
    y = x @ x.T
    torch.cuda.synchronize(device)
    print(f"matrix result shape: {tuple(y.shape)}")
    print(f"matrix result mean: {y.mean().item():.6f}")
    allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
    reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
    print(f"cuda memory allocated: {allocated_mb:.1f} MiB")
    print(f"cuda memory reserved: {reserved_mb:.1f} MiB")
    print("result: GPU is available to PyTorch")
    return 0


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())
    if completed.returncode != 0:
        print(f"command exited with code {completed.returncode}")


if __name__ == "__main__":
    raise SystemExit(main())
