from qiskit.visualization import *
from qiskit_ibm_runtime import *
import numpy as np
from qiskit import *
from qiskit_aer import *
from scipy.fft import fft
from scipy.stats import entropy


def generate_low_entropy_array(size: int) -> np.array:
    """
    Classically generate a low-entropy binary array where 0s appear with 90% probability.

    Args:
        size (int): The size of the array.

    Returns:
        np.array: A NumPy array of 0s and 1s with a bias towards 0s.
    """
    prob_zero = 0.9
    return np.array(np.random.choice([0, 1], size=size, p=[prob_zero, 1 - prob_zero]), dtype=np.uint8)

def generate_high_entropy_array(size: int) -> np.array:
    """
    Classically generate a high-entropy binary array where 0s and 1s appear with equal probability.

    Args:
        size (int): The size of the array.

    Returns:
        np.array: A NumPy array of 0s and 1s with equal distribution.
    """
    prob_zero = 0.5
    return np.array(np.random.choice([0, 1], size=size, p=[prob_zero, 1 - prob_zero]), dtype=np.uint8)

def generate_qrng_data(num_bits: int) -> np.array:
    """
    Generate simulated quantum random numbers using a single-qubit quantum circuit.

    Args:
        num_bits (int): Number of random bits to generate.

    Returns:
        np.array: A NumPy array of quantum-generated random bits.
    """
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)  # Apply Hadamard gate to create superposition
    circuit.measure(0, 0)  # Measure the qubit

    backend_sim = AerSimulator()  # Initialize the AerSimulator
    compiled_sim = transpile(circuit, backend_sim)  # Transpile the circuit
    result = backend_sim.run(compiled_sim, shots=num_bits, memory=True).result()      
    raw_data = result.get_memory()
    return np.array(raw_data, dtype=np.uint8)

def create_toeplitz(first_row: np.array, first_column: np.array) -> np.array:
    """
    Create a Toeplitz matrix given its first row and first column.

    Args:
        first_row (np.array): The first row of the Toeplitz matrix.
        first_column (np.array): The first column of the Toeplitz matrix.

    Returns:
        np.array: The generated Toeplitz matrix.
    """
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

def shannon_entropy(data: np.array) -> float:
    """
    Compute the Shannon entropy of a binary dataset.

    Args:
        data (np.array): Input binary data.

    Returns:
        float: Shannon entropy value.
    """
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return entropy(probabilities, base=2)

def apply_von_neumann_extractor(qrng_data: np.array) -> np.array:
    """
    Apply the Von Neumann extractor to remove bias from quantum random bits.

    Args:
        qrng_data (np.array): Input quantum-generated binary data.

    Returns:
        np.array: Extracted unbiased binary sequence.
    """
    extracted_bits = []
    
    # Iterate over the bitstring in pairs
    for i in range(0, len(qrng_data) - 1, 2):
        a = qrng_data[i]
        b = qrng_data[i+1]
        
        # Only keep the result when the pair has different bits
        if a != b:
            extracted_bits.append(a)
    
    return np.array(extracted_bits)

def apply_toeplitz_transformation(qrng_data: np.array, blocksize: int = 128) -> np.array:
    """
    Apply a Toeplitz hashing transformation to the input binary data.

    Args:
        qrng_data (np.array): Input quantum-generated binary data.
        blocksize (int): Size of the Toeplitz transformation block.

    Returns:
        np.array: Transformed data using Toeplitz hashing.
    """
    transformed_data = []
    first_row = np.random.randint(0, 2, blocksize) ^ np.array(qrng_data[:blocksize])
    first_column = np.random.randint(0, 2, blocksize) ^ np.array(qrng_data[:blocksize])
    toeplitz_matrix = create_toeplitz(first_row, first_column)

    for i in range(0, len(qrng_data), blocksize):
        data = qrng_data[i:i+blocksize].astype(int)
        if len(data) == blocksize: 
            transformed_data += list((toeplitz_matrix @ data) % 2)
    return np.array(transformed_data)

def apply_fft_toeplitz(qrng_data: np.array) -> np.array:
    """
    Apply an FFT-based transformation to Toeplitz-hashed data.

    Args:
        qrng_data (np.array): Input quantum-generated binary data.

    Returns:
        np.array: Transformed data after FFT and thresholding.
    """
    fft_transformed_data = np.real(fft(apply_toeplitz_transformation(qrng_data)) > 0.5)
    return np.array(fft_transformed_data.astype(int))

def apply_parity_extractor(qrng_data: np.array, blocksize: int = 4) -> np.array:
    """
    Apply a parity extractor to reduce bias in the binary sequence.

    Args:
        qrng_data (np.array): Input quantum-generated binary data.
        blocksize (int): Size of the blocks to compute parity.

    Returns:
        np.array: Extracted bits based on parity.
    """
    truncate_length = (len(qrng_data) // blocksize) * blocksize
    truncated_array = np.array(qrng_data[:truncate_length], dtype=int)

    # Reshape the array into chunks and calculate parity
    chunks = truncated_array.reshape(-1, blocksize)
    parity_array = np.sum(chunks, axis=1) % 2  # 0 for even parity, 1 for odd parity
    return np.array(parity_array)