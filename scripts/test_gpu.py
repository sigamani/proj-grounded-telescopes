#!/usr/bin/env python3

import subprocess
import sys

def check_gpu():
    """Check GPU availability and CUDA setup"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("PyTorch not installed")

def check_ray():
    """Check Ray setup"""
    try:
        import ray
        print(f"Ray version: {ray.__version__}")
        
        if not ray.is_initialized():
            ray.init()
        
        print("Ray cluster resources:")
        resources = ray.cluster_resources()
        for key, value in resources.items():
            print(f"  {key}: {value}")
            
    except ImportError:
        print("Ray not installed")
    except Exception as e:
        print(f"Ray error: {e}")

def check_vllm():
    """Check vLLM setup"""
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        print("vLLM not installed")

if __name__ == "__main__":
    print("=== GPU Test ===")
    check_gpu()
    print("\n=== Ray Test ===")
    check_ray()
    print("\n=== vLLM Test ===")
    check_vllm()