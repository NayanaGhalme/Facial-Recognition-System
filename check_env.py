import torch
import numpy as np
import onnxruntime as ort

# Fixed the underscores here
print(f"NumPy version: {np.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
print(f"Available ONNX Providers: {ort.get_available_providers()}")

# Extra check: Test if PyTorch can actually talk to the GPU
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")