import sys
from typing import Any, Dict, List, Optional
import torch


def get_device() -> torch.device:
    """
    Selects the best available hardware accelerator in priority order:
    1. CUDA (NVIDIA GPU or AMD ROCm)
    2. MPS (Apple Silicon GPU)
    3. XPU (Intel Arc / Data Center GPU)
    4. DirectML (Windows cross-vendor GPU support)
    5. XLA (Google TPU / Accelerator)
    6. CPU (Fallback)
    """
    # 1. NVIDIA CUDA / AMD ROCm
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 2. Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 3. Intel XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    # 4. Windows DirectML (via torch_directml)
    try:
        import torch_directml  # type: ignore

        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass

    # 5. Google TPU / XLA
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        return xm.xla_device()
    except ImportError:
        pass

    # Fallback
    return torch.device("cpu")


def get_detailed_device_info() -> Dict[str, Any]:
    """
    Gathers detailed system hardware specs, backend versions, 
    device counts, and memory details.
    """
    info: Dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "primary_device": str(get_device()),
        "devices_found": [],
    }

    # --- CUDA / ROCm ---
    if torch.cuda.is_available():
        backend_type = "ROCm (AMD)" if torch.version.hip else "CUDA (NVIDIA)"
        count = torch.cuda.device_count()
        
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            info["devices_found"].append({
                "type": "cuda",
                "backend": backend_type,
                "index": i,
                "name": props.name,
                "total_vram_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "driver_version": torch.version.cuda or torch.version.hip,
            })

    # --- Apple Silicon MPS ---
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["devices_found"].append({
            "type": "mps",
            "backend": "Metal Performance Shaders (Apple)",
            "index": 0,
            "name": "Apple Silicon Unified Memory GPU",
            "is_built": torch.backends.mps.is_built(),
        })

    # --- Intel XPU ---
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        count = torch.xpu.device_count()
        for i in range(count):
            props = torch.xpu.get_device_properties(i)
            info["devices_found"].append({
                "type": "xpu",
                "backend": "Intel SYCL / OneAPI",
                "index": i,
                "name": props.name,
                "total_vram_gb": round(props.total_memory / (1024**3), 2),
            })

    # --- Fallback CPU Info ---
    info["devices_found"].append({
        "type": "cpu",
        "backend": "Host CPU",
        "threads_available": torch.get_num_threads(),
    })

    return info


def inspect_tensor_device(tensor: torch.Tensor) -> None:
    """Demonstrates how to inspect device attributes directly on a Tensor."""
    print(f"Tensor Device String : {tensor.device}")
    print(f"Tensor Device Type   : {tensor.device.type}")
    print(f"Tensor Device Index  : {tensor.device.index if tensor.device.index is not None else 'N/A'}")


# --- Execution Example ---
if __name__ == "__main__":
    # 1. Get primary target device
    device = get_device()
    print(f"🚀 Optimal Target Device: {device}\n")

    # 2. Print system hardware summary report
    print("--- 📊 Hardware Diagnostic Report ---")
    report = get_detailed_device_info()
    print(f"PyTorch Version : {report['pytorch_version']}")
    print(f"Python Version  : {report['python_version']}")
    print(f"Primary Device  : {report['primary_device']}")
    print("\nDetected Hardware Devices:")
    for dev in report["devices_found"]:
        print(f"  • {dev}")

    # 3. Create a Tensor directly on the detected device
    print("\n--- 🧪 Tensor Allocation Test ---")
    x = torch.randn((3, 3), device=device)
    inspect_tensor_device(x)