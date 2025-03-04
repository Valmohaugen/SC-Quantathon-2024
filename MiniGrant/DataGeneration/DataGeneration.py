'''
Method 1: Mod2
Method 2: Iteration chunker
Method 3: Concatenation
Method 4: All previous methods combined
'''

# Imports
import time
from datetime import datetime, timezone
import numpy as np
from qiskit import *
from qiskit_aer import *
from qiskit_ibm_runtime import *
from qiskit.visualization import *

# Variables
method = 1                       # Chosen method to run
machine = 'ibm_brisbane'            # Chosen machine to submit jobs to
num_qubits = 100                 # Number of qubits to run on
num_shots = 1024                 # Number of shots to take
chunk_size = 30                  # Size of the chunking for the mod2 and iteration methods
mod2_mods = 3                    # Number of times to apply mod2. The value inputted results in 2 runs per 1 value. mod2_mods=3 --> 6 jobs QPU submitted

# Lets us use IBM machines
QiskitRuntimeService.save_account(channel = 'ibm_quantum', token = '4b3a2ecb92b9446de636396b89dd76f0d61c2715122c6a353720e5b58c124a7d3d20528012332d57add9a37a61aaf508b54c97698ddc770f922d68fd213b1b25',
                                  overwrite = True, set_as_default = True)
service = QiskitRuntimeService(instance = "ibm-q/open/main")

# QRNG for method1, method2, and method3
def number_generator_simulator_123(num_qubits, num_shots):
    t0 = time.time()                                                         # Initial time 
    circ = QuantumCircuit(num_qubits, num_qubits)                            # Creates circuit with num_qubits
    circ.h(range(num_qubits))                                                # Applies a Hadamard gate to all qubits
    circ.measure(range(num_qubits), range(num_qubits))                       # Measures all qubits

    simulator = AerSimulator()                                               # Lets us use the Aer Simulator 
    compiled_circuit = transpile(circ, simulator)                            # Compiled circuit using Aer 
    result = simulator.run(compiled_circuit, shots = num_shots).result()     # Result with of circuit
    counts = result.get_counts()                                             # Creates a dictionary mapping states to frequency obtained 
    t1 = time.time()                                                         # Final time
    
    return counts, (t1 - t0)

def number_generator_brisbane_123(num_qubits, num_shots):
    circ = QuantumCircuit(num_qubits, num_qubits)                 # Creates circuit with num_qubits
    circ.h(range(num_qubits))                                     # Applies a hadamard gate to all qubits
    circ.measure(range(num_qubits), range(num_qubits))            # Measures all qubits

    brisbane_backend = service.backend('ibm_brisbane')            # Creates a backend with ibm_brisbane
    pm = generate_preset_pass_manager(backend=brisbane_backend, optimization_level=3)   # Creates a pass manager with backend and optimization level
    isa_circuit = pm.run(circ)                                    # Creates the transpiled/optimized circuit
    with Session(backend=brisbane_backend) as session:    
        sampler = Sampler()                                       # Initializes the sampler to run the circuit
        job = sampler.run([isa_circuit], shots=num_shots)         # Runs the circuit as a job
        print(job.status())                                       # Prints the status of the job    
        print(job.result())                                       # Prints the result of the job
        counts = job.result()[0].data.c.get_counts()              # Creates a dictionary mapping the states to frquency obtained
        t0 = job.creation_date                              # Initial time
        execution_spans = job.result().metadata["execution"]["execution_spans"]
        t1 = execution_spans[0]["__value__"]["stop"].replace(tzinfo=timezone.utc)

    return counts, (t1 - t0)

def number_generator_sherbrooke_123(num_qubits, num_shots):
    circ = QuantumCircuit(num_qubits, num_qubits)                 # Creates circuit with num_qubits
    circ.h(range(num_qubits))                                     # Applies a hadamard gate to all qubits
    circ.measure(range(num_qubits), range(num_qubits))            # Measures all qubits

    sherbrooke_backend = service.backend('ibm_sherbrooke')        # Creates a backend with ibm_sherbrooke
    pm = generate_preset_pass_manager(backend=sherbrooke_backend, optimization_level=3)   # Creates a pass manager with backend and optimization level
    isa_circuit = pm.run(circ)                                    # Creates the transpiled/optimized circuit
    with Session(backend=sherbrooke_backend) as session:
        sampler = Sampler()                                       # Initializes the sampler to run the circuit
        job = sampler.run([isa_circuit], shots=num_shots)         # Runs the circuit as a job
        print(job.status())                                       # Prints the status of the job    
        print(job.result())                                       # Prints the result of the job
        counts = job.result()[0].data.c.get_counts()              # Creates a dictionary mapping the states to frquency obtained
        t0 = job.creation_date                              # Initial time
        execution_spans = job.result().metadata["execution"]["execution_spans"]
        t1 = execution_spans[0]["__value__"]["stop"].replace(tzinfo=timezone.utc)
    
    return counts, (t1 - t0)

# QRNG for method4
def number_generator_simulator_4(num_qubits, num_shots):
    t0 = time.time()
    circ = QuantumCircuit(num_qubits, num_qubits)                 # Creates circuit with number of qubits obtained
    circ.h(range(num_qubits))                                     # Applies a hadamard gate to all qubits
    circ.measure(range(num_qubits), range(num_qubits))            # Measures all qubits and assigns them to classical bits

    simulator = AerSimulator()                                    # Lets us use the Aer Simulator 
    compiled_circuit = transpile(circ, simulator)                 # Compiled circuit using Aer 
    result = simulator.run(compiled_circuit, shots=num_shots, memory=True).result()      # Result with the 10000 shots as to not run forever on IBM machines
    raw_data = result.get_memory()
    data = ''.join(raw_data)
    t1 = time.time()

    return data, (t1 - t0)     # Returns a concatenated string of all binary digits in the order that they were measured

def number_generator_brisbane_4(num_qubits, num_shots):
    circ = QuantumCircuit(num_qubits, num_qubits)                 # Creates circuit with number of qubits obtained
    circ.h(range(num_qubits))                                     # Applies a hadamard gate to all qubits
    circ.measure(range(num_qubits), range(num_qubits))            # Measures all qubits and assigns them to classical bits

    # Runs the QRNG on ibm_brisbane with num_shots
    brisbane_backend = service.backend('ibm_brisbane')            # Creates a backend with ibm_brisbane
    pm = generate_preset_pass_manager(backend=brisbane_backend, optimization_level=3)   # Can change optimization_level to whatever's desired
    isa_circuit = pm.run(circ)
    with Session(backend=brisbane_backend) as session:
        sampler = Sampler(mode=session)
        job = sampler.run([isa_circuit], shots=num_shots)
        counts = job.result()[0].data.c.get_counts()
        print(f'Job ID: {job.job_id()}\nJob status: {job.status()}')
        raw_data = list(counts.keys())
        data = ''.join(raw_data)
        t0 = job.creation_date                              # Initial time
        execution_spans = job.result().metadata["execution"]["execution_spans"]
        t1 = execution_spans[0]["__value__"]["stop"].replace(tzinfo=timezone.utc)
        
    return data, (t1 - t0)     # Returns a concatenated string of all binary digits in the order that they were measured

def number_generator_sherbrooke_4(num_qubits, num_shots):
    circ = QuantumCircuit(num_qubits, num_qubits)                 # Creates circuit with number of qubits obtained
    circ.h(range(num_qubits))                                     # Applies a hadamard gate to all qubits
    circ.measure(range(num_qubits), range(num_qubits))            # Measures all qubits and assigns them to classical bits

    # Runs the QRNG on ibm_sherbooke with num_shots
    sherbooke_backend = service.backend('ibm_sherbooke')          # Creates a backend with ibm_sherbooke
    pm = generate_preset_pass_manager(backend=sherbooke_backend, optimization_level=3)   # Can change optimization_level to whatever's desired
    isa_circuit = pm.run(circ)
    with Session(backend=sherbooke_backend) as session:
        sampler = Sampler(mode=session)
        job = sampler.run([isa_circuit], shots=num_shots)
        counts = job.result()[0].data.c.get_counts()
        print(f'Job ID: {job.job_id()}\nJob status: {job.status()}')
        raw_data = list(counts.keys())
        data = ''.join(raw_data)
        t0 = job.creation_date                              # Initial time
        execution_spans = job.result().metadata["execution"]["execution_spans"]
        t1 = execution_spans[0]["__value__"]["stop"].replace(tzinfo=timezone.utc)

    return data, (t1 - t0)     # Returns a concatenated string of all binary digits in the order that they were measured

# Necessary functions for the following methods
def obtain_ties(counts, machine):  
    max_value = max(counts.values())
    max_keys = [key for key, value in counts.items() if value == max_value]
    max_dict = {key: counts[key] for key in max_keys}

    while (len(max_keys) > 1):
        max_keys = tie_breaker(max_keys, machine)

    return max_keys[0]

def tie_breaker(ties, machine):          # Method 1 - Breaks a tie if there is one
    new_num_qubits = int(np.ceil(np.log2(len(ties))))

    if machine == 'ibm_brisbane':
        counts, _ = number_generator_brisbane_123(new_num_qubits, num_shots)
    elif machine == 'ibm_sherbrooke':
        counts, _ = number_generator_sherbrooke_123(new_num_qubits, num_shots)
    elif machine == 'simulator':
        counts, _ = number_generator_simulator_123(new_num_qubits, num_shots)

    # Discard out of range keys
    counts = {key: value for key, value in counts.items() if int(key, 2) < len(ties)}
    max_value = max(counts.values())  # Max among the ties
    max_keys = [key for key, value in counts.items() if value == max_value]  # Keys (indexes!) that tied
    
    new_ties = []
    for key in max_keys:
        index = int(key, 2)  # Convert binary string key to integer
        new_ties.append(ties[index])  # Append the corresponding original tie

    return new_ties

def obtain_rand(counts):          # Method 1 - Obtains a random value
    # Get all values
    max_value = max(counts.values())
    max_keys = [key for key, value in counts.items() if value == max_value]
    max_dict = {key: counts[key] for key in max_keys}
    # Sum all Q-bit entries for tied keys
    # Get the first key
    new_key = [int(i) for i in max_keys[0]]
    for i in range(1, len(max_keys)):
        for j in range(0, len(new_key)):
            new_key[j] += int(max_keys[i][j])

    super_new_key = ''
    # Bitwise XOR (mod % 2) on each qubit
    for i in range(0, len(new_key)):
        # super_new_key[i] = str(int(new_key[i]) % 2)
        super_new_key += str(int(new_key[i]) % 2)

    return super_new_key

def check_tie(counts):          # Method 3 - Checks if there is a tie
    max_value = max(counts.values())
    max_keys = [key for key, value in counts.items() if value == max_value]

    if len(max_keys) > 1:
        return None
    else:
        return max_keys[0]

def iterationChunker(machine, chunk_size):          # Method 4 - Iterates simliar to method 2
    chunks = int(np.floor(num_qubits / chunk_size))
    remainder = int(num_qubits - chunks * chunk_size)
    rand_num = ''
    elapsedTime = 0

    for i in range(0, chunks):
        if machine == 'simulator':
            num_val, time_val = number_generator_simulator_4(chunk_size, num_shots)
            rand_num += num_val
            elapsedTime += time_val
        elif machine == 'ibm_brisbane':
            num_val, time_val = number_generator_brisbane_4(chunk_size, num_shots)
            rand_num += num_val
            elapsedTime += time_val
        elif machine == 'ibm_sherbrooke':
            num_val, time_val = number_generator_sherbrooke_4(chunk_size, num_shots)
            rand_num += num_val
            elapsedTime += time_val


    if remainder != 0:
        for i in range(0, chunks):
            if machine == 'simulator':
                num_val, time_val = number_generator_simulator_4(chunk_size, num_shots)
                rand_num += num_val
                elapsedTime += time_val
            elif machine == 'ibm_brisbane':
                num_val, time_val = number_generator_brisbane_4(chunk_size, num_shots)
                rand_num += num_val
                elapsedTime += time_val
            elif machine == 'ibm_sherbrooke':
                num_val, time_val = number_generator_sherbrooke_4(chunk_size, num_shots)
                rand_num += num_val
                elapsedTime += time_val

    return rand_num, elapsedTime



# Actual methods for QRNG
def method1(num_qubits, chunk_size, machine):
    chunks = int(np.floor(num_qubits / chunk_size))
    remainder = int(num_qubits - chunks * chunk_size)
    rand_num = ''

    if machine == 'ibm_brisbane':
        for i in range(0, chunks):
            counts, elapsedTime = number_generator_brisbane_123(chunk_size, num_shots)
            rand_value = obtain_rand(counts)
            rand_num += rand_value
    elif machine == 'ibm_sherbrooke':
        for i in range(0, chunks):
            counts, elapsedTime = number_generator_sherbrooke_123(chunk_size, num_shots)
            rand_value = obtain_rand(counts)
            rand_num += rand_value
    elif machine == 'simulator': 
        for i in range(0, chunks):
            counts, elapsedTime = number_generator_simulator_123(chunk_size, num_shots)
            rand_value = obtain_rand(counts)
            rand_num += rand_value

    if remainder != 0:
        if machine == 'ibm_brisbane':
            counts, elapsedTime = number_generator_brisbane_123(remainder, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value
        elif machine == 'ibm_sherbrooke':
            counts, elapsedTime = number_generator_sherbrooke_123(remainder, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value
        elif machine == 'simulator':
            counts, elapsedTime = number_generator_simulator_123(remainder, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value

    return rand_num, elapsedTime

def method2(num_qubits, chunk_size, machine):
    chunks = int(np.floor(num_qubits / chunk_size))
    remainder = int(num_qubits - chunks * chunk_size)
    rand_num = ''

    if machine == 'ibm_brisbane':
        for i in range(0, chunks):
            counts, elapsedTime = number_generator_brisbane_123(chunk_size, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value
    elif machine == 'ibm_sherbrooke':
        for i in range(0, chunks):
            counts, elapsedTime = number_generator_sherbrooke_123(chunk_size, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value
    elif machine == 'simulator':
        for i in range(0, chunks):
            counts, elapsedTime = number_generator_simulator_123(chunk_size, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value

    if remainder != 0:
        if machine == 'ibm_brisbane':
            counts, elapsedTime = number_generator_brisbane_123(remainder, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value
        elif machine == 'ibm_sherbrooke':
            counts, elapsedTime = number_generator_sherbrooke_123(remainder, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value
        elif machine == 'simulator':
            counts, elapsedTime = number_generator_simulator_123(remainder, num_shots)
            rand_value = obtain_ties(counts, machine)
            rand_num += rand_value

    return rand_num, elapsedTime

def method3(num_qubits, division_size, machine):
    extra = num_qubits % division_size
    iterations = int((num_qubits - extra) / division_size)
    final_output = []

    if machine == 'ibm_brisbane':
        for iteration in range(iterations):
            division_chunk = None

            while division_chunk is None:
                counts, elapsedTime = number_generator_brisbane_123(division_size, num_shots)
                division_chunk = check_tie(counts)
            final_output.append(division_chunk)

        if extra != 0:
            division_chunk = None

            while division_chunk is None:
                counts, elapsedTime = number_generator_brisbane_123(extra, num_shots)
                division_chunk = check_tie(counts)
            final_output.append(division_chunk)

        return ''.join(final_output), elapsedTime
    elif machine == 'ibm_sherbrooke':
        for iteration in range(iterations):
            division_chunk = None

            while division_chunk is None:
                counts, elapsedTime = number_generator_sherbrooke_123(division_size, num_shots)
                division_chunk = check_tie(counts)
            final_output.append(division_chunk)

        if extra != 0:
            division_chunk = None

            while division_chunk is None:
                counts, elapsedTime = number_generator_sherbrooke_123(extra, num_shots)
                division_chunk = check_tie(counts)
            final_output.append(division_chunk)

        return ''.join(final_output), elapsedTime
    elif machine == 'simulator':
        for iteration in range(iterations):
            division_chunk = None

            while division_chunk is None:
                counts, elapsedTime = number_generator_simulator_123(division_size, num_shots)
                division_chunk = check_tie(counts)
            final_output.append(division_chunk)

        if extra != 0:
            division_chunk = None

            while division_chunk is None:
                counts, elapsedTime = number_generator_simulator_123(extra, num_shots)
                division_chunk = check_tie(counts)
            final_output.append(division_chunk)

        return ''.join(final_output), elapsedTime

def method4(machine, chunk_size, mod2_mods):
    iterations = mod2_mods
    outputs = []
    final_rand_num = ''
    final_time = 0

    for i in range(iterations):
        outputs.append(iterationChunker(machine, chunk_size)[0])
        final_time += iterationChunker(machine, chunk_size)[1]

    for i in range(0, iterations, mod2_mods):
        pair_xor = ''
        
        for j in range (0, len(outputs[i])):
            cur_char = 0
            for k in range (0, mod2_mods):
                cur_char += int(outputs[i + k][j])
            pair_xor += str(cur_char % 2)

        final_rand_num += pair_xor
    
    return final_rand_num, final_time

# Obtains data with throughput and date
match method:
    case 1:
        data, elapsed_time = method1(num_qubits, chunk_size, machine)
    case 2:
        data, elapsed_time = method2(num_qubits, chunk_size, machine)
    case 3:
        data, elapsed_time = method3(num_qubits, chunk_size, machine)
    case 4:
        data, elapsed_time = method4(machine, chunk_size, mod2_mods)

if machine == 'simulator':
    elapsedTime = int(elapsed_time)
else:
    elapsedTime = int(elapsed_time.total_seconds())

# Calculate throughput as bits per second (then convert to megabits per second)
throughput = len(data) / elapsedTime
date = f"{datetime.now().strftime('%b')}{datetime.now().day}"

# Writes data to a file
if machine == 'simulator':
    with open(f'../Data/{date}_{machine}_method{method}_{num_qubits}qubits_{num_shots}shots_{chunk_size}chunkSize_{mod2_mods}mods.txt', 'w') as f:
        f.write(f'Number of bits: {len(data)}')
        f.write(f'\nThroughput: {throughput/1e6} Mb/s')
        f.write(f'\nTime taken: {elapsedTime} s\n')
        for shot in data:
            f.write(shot)
elif machine == 'ibm_brisbane':
    with open(f'../Data/{date}_{machine.split('_')[-1]}_method{method}_{num_qubits}qubits_{num_shots}shots_{chunk_size}chunkSize_{mod2_mods}mods.txt', 'w') as f:
        f.write(f'Number of bits: {len(data)}')
        f.write(f'\nThroughput: {throughput/1e6} Mb/s')
        f.write(f'\nTime taken: {elapsedTime} s\n')
        for shot in data:
            f.write(shot)
elif machine == 'ibm_sherbrooke':
    with open(f'../Data/{date}_{machine.split('_')[-1]}_method{method}_{num_qubits}qubits_{num_shots}shots_{chunk_size}chunkSize_{mod2_mods}mods.txt', 'w') as f:
        f.write(f'Number of bits: {len(data)}')
        f.write(f'\nThroughput: {throughput/1e6} Mb/s')
        f.write(f'\nTime taken: {elapsedTime} s\n')
        for shot in data:
            f.write(shot)