import torch

# print the PyTorch version
print("PyTorch version:", torch.__version__)

# Print if cuda is available
print("CUDA available:", torch.cuda.is_available())

# Print the number of GPUs
print("Number of GPUs:", torch.cuda.device_count())

# Print the name of the GPU
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

