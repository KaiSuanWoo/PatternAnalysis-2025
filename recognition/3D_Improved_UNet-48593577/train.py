import torch
from utils import check_cuda

def main():
    print("=== COMP3710 Project: 3D Improved U-Net ===")
    device = check_cuda()
    print(f"Running on: {device}")

if __name__ == "__main__":
    main()
