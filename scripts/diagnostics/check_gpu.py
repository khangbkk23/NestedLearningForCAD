import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys

def check_gpu():
    print("="*30)
    print("GPU & CUDA CHECK SCRIPT")
    print("="*30)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check Torch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        # Get CUDA Version
        print(f"CUDA version (runtime): {torch.version.cuda}")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Total Memory: {props.total_memory / (1024**3):.2f} GB")
            
        # Try a small tensor operation on GPU
        try:
            x = torch.randn(1, device='cuda')
            print("\n[SUCCESS] Test tensor created on GPU successfully.")
        except Exception as e:
            print(f"\n[FAILURE] Failed to create tensor on GPU: {e}")
    else:
        print("\n[!] CUDA is NOT available.")
        print("    - If you have an NVIDIA GPU, check if drivers and CUDA Toolkit are installed.")
        print("    - Check if you installed the 'cuXXX' version of PyTorch.")

    print("="*30)

if __name__ == "__main__":
    check_gpu()
