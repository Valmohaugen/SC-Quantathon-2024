# SC Quantathon 2024: Quantum Random Number Generation

This repository is the home for the 2024 DoraHacks SC Quantathon challenge, focused on implementing, characterizing, and verifying a Quantum Random Number Generator (QRNG) on IBM quantum processors using Hadamard-gate circuits with multiple extraction and post-processing methods.

## Features

* **QRNG Implementation (Stage 1):**

  * Four bit-string generation algorithms: Mod2 XOR, Iteration with chunking, Concatenation, and a combined method applying all three.
  * Runs on IBM QPUs (Brisbane, Sherbrooke) and the Qiskit Aer simulator.
  * Configurable qubit count, shot number, chunk size, and method selection.

* **ML Classification of Quantum vs. Classical Randomness (Stages 2 & 5):**

  * SVM classifier achieving 75%+ accuracy distinguishing QRNG output from pseudo-random bitstrings.
  * Evaluated XGBoost, Gradient Boosting, and SVM -- SVM selected as best performer.
  * Stage 5 re-evaluates classification accuracy after entropy post-processing.

* **Noise and Fidelity Characterization (Stage 3):**

  * Analysis of hardware noise sources: decoherence times (T1, T2) and gate error probabilities.
  * Readout assignment error analysis across qubits and machines.
  * Comparison of noise-free simulator, noisy simulator, and pseudo-random baselines.
  * Hadamard-to-native gate decomposition analysis.

* **Entropy Extraction and Post-Processing (Stage 4):**

  * Toeplitz matrix hashing for randomness extraction (using BYUCamachoLab's `ottoeplitz`).
  * Von Neumann extractor for bias removal.
  * Parity extractor for entropy concentration.
  * FFT-based Toeplitz transformation.
  * HPCG benchmark using QRNG data via a C shared library shim that intercepts `rand()` calls.

* **Data Generation Pipeline:**

  * Large-scale QRNG data generation across multiple methods, qubit counts (10--100,000), and backends.
  * Post-processing analysis notebooks comparing entropy across extraction methods.

## Project Structure

```
SC-Quantathon-v1-2024/
  README.md
  requirements.txt
  .gitignore
  MiniGrant/                            # Extended data generation pipeline
    DataGeneration/
      DataGeneration.py                 # Main QRNG script (4 methods, multi-backend)
      increased_data_generation.ipynb   # Scaled-up generation notebook
      binary_string_to_binary.py        # Text-to-binary file converter
    PostProcessing/
      postprocessing.py                 # Entropy extraction functions library
      parity-extractor.py              # Standalone parity analysis
      real_entropy_comparison.ipynb     # Entropy analysis on real QPU data
      simple_entropy_comparison.ipynb   # Entropy analysis on simulated data
  SC-Quantathon-2024/                   # 5-stage challenge submissions
    common/
      ottoeplitz.py                     # Toeplitz hashing (BYUCamachoLab)
    renamed-datasets/                   # Processed datasets for classification
    Stage1/                             # QRNG implementation
    Stage2/                             # SVM quantum vs. classical classifier
    Stage3/                             # Noise and fidelity characterization
    Stage4/                             # Entropy extraction + HPCG benchmark
    Stage5/                             # Post-extraction QRNG verification
```

## How to Use This Repository

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Valmohaugen/SC-Quantathon-v1-2024.git
   cd SC-Quantathon-v1-2024
   ```

2. **Set up your environment:**

   ```bash
   conda create -n quantathon python=3.10
   conda activate quantathon
   pip install -r requirements.txt
   ```

3. **Set your IBM Quantum token** (required for QPU execution):

   ```bash
   export IBM_QUANTUM_TOKEN="your-token-here"
   ```

4. **Generate QRNG data:**

   ```bash
   cd MiniGrant/DataGeneration
   python DataGeneration.py
   ```

   Adjust `method`, `machine`, `num_qubits`, and other parameters at the top of the script.

5. **Explore the challenge stages:**

   Open the Stage notebooks in Jupyter:
   - `SC-Quantathon-2024/Stage1/Stage1.ipynb` -- QRNG circuits
   - `SC-Quantathon-2024/Stage2/SVMClassifier.ipynb` -- ML classification
   - `SC-Quantathon-2024/Stage3/` -- Noise analysis (5 notebooks)
   - `SC-Quantathon-2024/Stage4/entropize.ipynb` -- Entropy extraction
   - `SC-Quantathon-2024/Stage5/SVMOurQvsPEntropied.ipynb` -- Final verification

## Results

* Generated QRNG bitstrings at scales from 10 to 100,000 qubits across 4 methods and 2 IBM backends.
* Built an SVM classifier that distinguishes quantum from classical random bitstrings with 75%+ accuracy after entropy post-processing.
* Characterized the impact of T1/T2 decoherence and gate errors on QRNG output quality.
* Demonstrated Toeplitz hashing and Von Neumann extraction improve entropy of raw QRNG output.
* Applied QRNG as the entropy source for an HPCG benchmark via a shared library `rand()` shim.

**Dependencies:**

* Python 3.10+, Jupyter, qiskit, qiskit-aer, qiskit-ibm-runtime, numpy, scipy, matplotlib, scikit-learn, pandas.
