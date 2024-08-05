import tensorflow as tf

# Check if GPUs are available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs are available.")
    for gpu in gpus:
        print(f"GPU device: {gpu.name}")
else:
    print("No GPUs available.")

# Print details about the GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Perform a simple TensorFlow operation to confirm usage
with tf.device('/GPU:0'):  # Specify GPU:0 for this operation
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    c = tf.matmul(a, b)
    print("TensorFlow operation result on GPU:")
    print(c)
