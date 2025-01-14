import os
import numpy as np


filepath = '../../SC-Quantathon-2024/renamed-datasets/classical_bitstring_20k.txt'

# Function to calculate Shannon entropy
def shannon_entropy(bits):
    
    # Calculate the frequency of 0's and 1's
    counts = np.bincount(bits)
    probabilities = counts / len(bits)
    
    # Filter out zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

# Read the file and convert to a numpy array of integers
with open(filepath, 'r') as f:
    data = f.read().strip()  # Read the file and remove any trailing spaces/newlines
    array = np.array([int(char) for char in data if char in '01'], dtype=np.int8)


    for L in (1, 2, 4, 8, 16, 32):
        truncate_length = (len(array) // L) * L
        truncated_array = array[:truncate_length]

        # Reshape the array into chunks and calculate parity
        chunks = truncated_array.reshape(-1, L)
        parity_array = np.sum(chunks, axis=1) % 2  # 0 for even parity, 1 for odd parity
        entropy = shannon_entropy(parity_array)

        print(f"For L: {L}, the shannon entropy is {entropy}")

