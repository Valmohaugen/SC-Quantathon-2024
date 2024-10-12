# SC-Quantathon-2024
For stage 1, we were tasked with implementing a QRNG on IBM QPUs. To do this, we created multiple algotrithms to generate a bit string the size of the number of qubits greater than 1. Our algorithms are:
- Concatenation
- Iteration with chunking
- Reduce XOR
Check the "Stage1" folder.

For stage 2, we built a classifier to achieve high accuracy with models. We tested:
- XGB
- Gradient boosting
- SVM (winner winner chicken dinner)
Check the "Stage2" folder.

For stage 3, we characterized noise and fidelity by analyzing our QRNG with the following methods: 
- Looking at the hardware specifications that lead to noise 
- Seeing how these problems readout assignment error
- Plotting the three simulators: a no noise quantum simulator, a noisy quantum simulator, and a pseudo-random number simulator
From this, we wanted to determine which values of the decoherence times (T1 and T2) and gate error probability would lead to the most noise, thus causing the qubit to fall into one of the states with a greater probability than the other which reduces randomness.
Check the "Stage3" folder.

For stage 4, we implemented pre-processing and post-processing for high entropy by cleaning our QRNG data to extract the highest entropy randomness. We used two extraction methods:
- Toeplitz matrix hashing randomness extraction
- Von Neumann randomness extraction
Check the "Stage4" folder.

For stage 5, we were tasked with building a high-accuracy classification model for QRNG verification. Because of the quality boost to our QRNG data, our classifier reached a minimum accuracy of 75%.
Check the "Stage5" folder.
