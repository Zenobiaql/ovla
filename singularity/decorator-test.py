import cmath
import os

def DFT(exp_signal):
    def inner(mag, tail):
        signal = exp_signal(mag, tail)
        N = len(signal)
        for i in range(N):
            dft = 0
            for j in range(N):
                dft += signal[j] * cmath.exp(-2j * cmath.pi * i * j / N)
            yield dft
    return inner

@DFT
def exp_signal(mag, tail):
    sequence = [mag**i for i in range(100) if mag**i >= tail]
    return sequence

# Write DFT results to file
file_path = '/mnt/data-qilin/0113-Tensor/dft.txt'

# Check if the file exists, if not create a new file
if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        pass  # Create an empty file

# Write DFT results to the file
with open(file_path, 'w') as f:
    for i in exp_signal(2, 0.01):
        f.write(f'{i}\n')

# Verify if the file is written properly
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        if content:
            print("DFT results have been written to 'dft.txt' successfully.")
        else:
            print("The file 'dft.txt' is empty.")
else:
    print("The file 'dft.txt' does not exist.")

print(os.getcwd())