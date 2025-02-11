import numpy as np
from scipy.fft import fft
from scipy.stats import entropy
from qiskit import *
from qiskit_aer import *
from qiskit_ibm_runtime import *
from qiskit.visualization import *

def generate_low_entropy_array(size: int) -> np.ndarray:
    prob_zero = 0.9
    return np.random.choice([0, 1], size=size, p=[prob_zero, 1 - prob_zero])

def generate_high_entropy_array(size: int) -> np.ndarray:
    prob_zero = 0.5
    return np.random.choice([0, 1], size=size, p=[prob_zero, 1 - prob_zero])

# Step 1: Generate QRNG data using Qiskit
def generate_qrng_data(num_bits):
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)  # Apply Hadamard gate to create superposition
    circuit.measure(0, 0)  # Measure the qubit

    backend_sim = AerSimulator()  # Initialize the AerSimulator
    compiled_sim = transpile(circuit, backend_sim)  # Transpile the circuit
    result_sim = backend_sim.run(compiled_sim, shots=num_bits).result()  # Run the simulation
    counts = result_sim.get_counts()

    # Convert measurement results to binary bitstring
    bits = []
    for bit, count in counts.items():
        bits.extend([int(bit)] * count)
    return np.array(bits[:num_bits])

# Step 2: Create a Toeplitz matrix
def create_toeplitz(first_row, first_column):
    n = len(first_row)
    m = len(first_column)
    toeplitz_matrix = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            if j >= i:
                toeplitz_matrix[i, j] = first_row[j - i]
            else:
                toeplitz_matrix[i, j] = first_column[i - j]
    return toeplitz_matrix

# Step 3: Calculate Shannon entropy
def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return entropy(probabilities, base=2)

# Step 4: Apply Toeplitz transformation dynamically
def apply_toeplitz_dynamically(qrng_data, toeplitz_matrix, block_size):
    transformed_data = []
    for i in range(0, len(qrng_data) - block_size + 1, block_size):
        block = qrng_data[i:i + block_size]
        transformed_block = np.dot(toeplitz_matrix, block) % 2
        transformed_data.extend(transformed_block)
    return np.array(transformed_data)

# Step 5: Apply FFT-based Toeplitz transformation
def apply_fft_toeplitz(qrng_data, toeplitz_matrix, block_size):
    transformed_data = []
    for i in range(0, len(qrng_data) - block_size + 1, block_size):
        block = qrng_data[i:i + block_size]
        fft_block = fft(block)
        fft_toeplitz = fft(toeplitz_matrix, axis=1)
        transformed_block = np.real(np.fft.ifft(fft_block * fft_toeplitz)).astype(int) % 2
        transformed_data.extend(transformed_block)
    return np.array(transformed_data)

def apply_parity_extractor(qrng_data, block_size)
    truncate_length = (len(array) // block_size) * block_size
    truncated_array = array[:truncate_length]

    # Reshape the array into chunks and calculate parity
    chunks = truncated_array.reshape(-1, block_size)
    parity_array = np.sum(chunks, axis=1) % 2  # 0 for even parity, 1 for odd parity
    return np.array(parity_array)