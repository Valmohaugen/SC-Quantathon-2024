# Use this file to create a file of equivalent raw binary bits from a file of
#   '0' and '1' text file character bits.
# Usage: run as command line: "python3 binary_string_to_binary.py"

def binary_string_to_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        binary_data = infile.read().strip()

    # Convert binary string to bytes
    byte_data = int(binary_data, 2).to_bytes((len(binary_data) + 7) // 8, byteorder='big')

    # Write the byte data to the output file
    with open(output_file, 'wb') as outfile:
        outfile.write(byte_data)

if __name__ == "__main__":
    input_file = 'simulator_30_1024_30_2.txt'   # Replace with your binary text file
    output_file = 'quantum_bits.txt' # Replace with your desired binary output file

    binary_string_to_file(input_file, output_file)
    print(f"Binary data written to {output_file}")