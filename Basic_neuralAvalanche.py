import numpy as np
import matplotlib.pyplot as plt

N = 100
connection_prob = 0.05
weights = (0.5,2.0)
synaptic_connections = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if(i != j and np.random.rand() < connection_prob):
            synaptic_connections[i,j] = np.random.uniform(*weights)
        
Threshold_value = 1.2
T = 100
dt = 0.1
num_iterations = 1000

avalanche_sizes = []
for _ in range(num_iterations):
    I = np.zeros(N)
    s = np.zeros(N)
    initial_neuron = np.random.randint(0,N)
    s[initial_neuron] = 1
    for _ in range(1,N):
        for i in range(N-1):
            I = np.sum(synaptic_connections[i,:]*s)
            if(I > Threshold_value):
                s[i] = 1
            else:
                s[i] = 0          
    avalanche_size = np.sum(s)
    if(avalanche_size > 0):
        avalanche_sizes.append(avalanche_size)
        
avalanche_sizes = np.array(avalanche_sizes)

min_size = np.min(avalanche_sizes)
max_size = np.max(avalanche_sizes)
num_bins = 50

plt.figure(figsize=(7,5))
plt.hist(avalanche_sizes, bins=num_bins, log=True, density=True, alpha=0.5, label='Avalanche size distribution', color=(0.9,0.1,0.1))
plt.xlabel('Avalanche sizes')
plt.ylabel('Frequency')
plt.title('Avalanche size distribution')
plt.legend()
plt.show()
