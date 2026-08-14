"""Print the local AI-detector software and hardware environment as JSON."""

import importlib.metadata
import json
import os
import platform
import sys
from datetime import UTC, datetime

try:
    import torch
except ImportError:
    torch = None


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def mps_available():
    return (
        torch is not None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def select_device():
    if torch is None:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def main():
    packages = [
        "accelerate",
        "datasets",
        "evaluate",
        "numpy",
        "PyYAML",
        "scikit-learn",
        "torch",
        "transformers",
    ]
    cuda_available = torch is not None and torch.cuda.is_available()
    environment = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hardware": {
            "logical_cpu_count": os.cpu_count(),
            "selected_device": select_device(),
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if torch else 0,
            "cuda_version": torch.version.cuda if torch else None,
            "mps_available": mps_available(),
        },
        "packages": {name: package_version(name) for name in packages},
    }
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
