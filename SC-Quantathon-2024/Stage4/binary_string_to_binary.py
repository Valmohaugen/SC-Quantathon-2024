def binary_string_to_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        binary_data = infile.read().strip()

    # Convert binary string to bytes
    byte_data = int(binary_data, 2).to_bytes((len(binary_data) + 7) // 8, byteorder='big')

    # Write the byte data to the output file
    with open(output_file, 'wb') as outfile:
        outfile.write(byte_data)

if __name__ == "__main__":
    input_file = 'mod2_simulator_10k.txt'   # Replace with your binary text file
    output_file = 'quantum_bits.txt' # Replace with your desired binary output file

    binary_string_to_file(input_file, output_file)
    print(f"Binary data written to {output_file}")

