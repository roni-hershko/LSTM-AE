import numpy as np
import pandas as pd

# Step 1: Generate random sequences
num_sequences = 10000  # Total number of sequences
sequence_length = 50   # Length of each sequence

# Generate random sequences in the range [0, 1]
synthetic_data = np.random.rand(num_sequences, sequence_length)

# Step 2: Post-process sequences
for j in range(num_sequences):
    # Sample a random integer i in the range [20, 30]
    i = np.random.randint(20, 31)
    # Modify the sequence in the range [i-5, i+5]
    synthetic_data[j, max(0, i-5):min(sequence_length, i+6)] *= 0.1

# Step 3: Split the data
train_size = int(0.6 * num_sequences)
val_size = int(0.2 * num_sequences)
test_size = num_sequences - train_size - val_size

train_data = synthetic_data[:train_size]
val_data = synthetic_data[train_size:train_size + val_size]
test_data = synthetic_data[train_size + val_size:]

# Save the data for later use
np.save("train_data.npy", train_data)
np.save("val_data.npy", val_data)
np.save("test_data.npy", test_data)

print("Synthetic data created and saved!")
