import torch

def check_cuda():
    """Print and return active device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {name}")
        return device
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device("cpu")
