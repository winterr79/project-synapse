import torch

# This script checks for the availability of a CUDA-enabled GPU.

if __name__ == "__main__":
    # torch.cuda.is_available() returns True if a CUDA GPU is found
    if torch.cuda.is_available():
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"✅ Success! Found {gpu_count} CUDA-enabled GPU(s).")
        
        # Get the name of the primary GPU (device 0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   - Primary GPU: {gpu_name}")
        print("\nTraining will run on the GPU.")
    else:
        print("❌ Warning: No CUDA-enabled GPU found.")
        print("   - PyTorch will use the CPU for training.")
        print("   - The training process will be EXTREMELY SLOW.")