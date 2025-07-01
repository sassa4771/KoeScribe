import torch

cuda_is_available = torch.cuda.is_available()

print(f'torch_version = {torch.__version__}')
print(f'cuda_available = {cuda_is_available}')
if cuda_is_available:
    print(f'cuda_device_count = {torch.cuda.device_count()}')
    print(f'cuda_GPU_Num = {torch.cuda.current_device()}')
    print(f'GPU_Name = {torch.cuda.get_device_name()}')
    print(f'GPU_Compute_Capability = {torch.cuda.get_device_capability()}')

# %%
