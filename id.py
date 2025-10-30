import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices visible:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")