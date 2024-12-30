# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# print(torch.backends.mps.is_available())

import torch

# Metal API 사용 가능 여부 확인
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())

# 사용 가능한 경우 'mps' 장치를 선택
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)