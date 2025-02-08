from qiskit.visualization import *
from qiskit_ibm_runtime import *
import numpy as np
from qiskit import *
from qiskit_aer import *
from scipy.fft import fft
from scipy.stats import entropy

def generate_low_entropy_array(size: int) -> np.ndarray:
    prob_zero = 0.9
    return np.random.choice([0, 1], size=size, p=[prob_zero, 1 - prob_zero])

def generate_high_entropy_array(size: int) -> np.ndarray:
    prob_zero = 0.5
    return np.random.choice([0, 1], size=size, p=[prob_zero, 1 - prob_zero])

def generate_qrng_data(num_bits):
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)  # Apply Hadamard gate to create superposition
    circuit.measure(0, 0)  # Measure the qubit

    backend_sim = AerSimulator()  # Initialize the AerSimulator
    compiled_sim = transpile(circuit, backend_sim)  # Transpile the circuit
    result = backend_sim.run(compiled_sim, shots=num_bits, memory=True).result()      
    raw_data = result.get_memory()
    return np.array(raw_data)

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


def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return entropy(probabilities, base=2)

def apply_von_neumann_extractor(qrng_data):
    extracted_bits = []
    
    # Iterate over the bitstring in pairs
    for i in range(0, len(qrng_data) - 1, 2):
        a = qrng_data[i]
        b = qrng_data[i+1]
        
        # Only keep the result when the pair has different bits
        if a != b:
            extracted_bits.append(a)
    
    # Return the extracted bitstring
    return np.array(extracted_bits)

def apply_toeplitz_transformation(qrng_data, blocksize=128):
    transformed_data = []
    first_row = np.random.randint(0, 2, blocksize)
    first_column = np.random.randint(0, 2, blocksize)
    toeplitz_matrix = create_toeplitz(first_row, first_column)

    # Step 3: Apply Toeplitz transformation
    for i in range(0, len(qrng_data), blocksize):
        data = qrng_data[i:i+blocksize].astype(int)
        if len(data) == blocksize: 
            transformed_data += list((toeplitz_matrix @ data) % 2)
    return np.array(transformed_data)

def apply_fft_toeplitz(qrng_data):
    fft_transformed_data = np.real(fft(apply_toeplitz_transformation(qrng_data)) > 0.5)
    return np.array(fft_transformed_data.astype(int))

def apply_parity_extractor(qrng_data, block_size=4):
    truncate_length = (len(array) // block_size) * block_size
    truncated_array = array[:truncate_length]

    # Reshape the array into chunks and calculate parity
    chunks = truncated_array.reshape(-1, block_size)
    parity_array = np.sum(chunks, axis=1) % 2  # 0 for even parity, 1 for odd parity
    return np.array(parity_array)