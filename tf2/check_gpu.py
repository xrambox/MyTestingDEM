import tensorflow as tf

# List all available devices
devices = tf.config.list_physical_devices()
print("All Available Devices:")
for device in devices:
    print(device)

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("\nAvailable GPUs:")
for gpu in gpus:
    print(gpu)

# Check if TensorFlow is built with GPU support
print("\nIs TensorFlow built with CUDA support?")
print(tf.test.is_built_with_cuda())

# Check if GPUs are available and visible to TensorFlow
print("\nIs GPU available?")
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

# Get the number of GPUs available
print("\nNum GPUs Available: ", len(gpus))
