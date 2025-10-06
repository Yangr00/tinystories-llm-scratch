import torch

# Check if a CUDA-enabled GPU is available
if torch.cuda.is_available():
    print("GPU is available! 🚀")
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print details for each GPU
    for i in range(num_gpus):
        print(f"--- GPU {i} ---")
        print(f"Device Name: {torch.cuda.get_device_name(i)}")

    # Create a tensor and move it to the GPU to confirm it works
    tensor = torch.randn(3, 3)
    print("\nCreating a tensor on the CPU:")
    print(tensor)

    tensor_gpu = tensor.to("cuda")
    print("\nMoving the tensor to the GPU:")
    print(tensor_gpu)

else:
    print("GPU is not available. Running on CPU. 😔")
