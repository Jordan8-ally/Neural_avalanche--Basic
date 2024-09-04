import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

N = 100
connection_prob = 0.1
weights = (0.5,2.0)
synaptic_connections = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if(i != j and np.random.rand() < connection_prob):
            synaptic_connections[i,j] = np.random.uniform(*weights)
        
Threshold_value = 1.0
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

bins = np.logspace(np.log10(min_size), np.log10(max_size), num_bins)
hist, _ = np.histogram(avalanche_sizes, bins=bins, density=True)
bins_center = 0.5*(bins[1:] + bins[:-1])

def power_law(x, a, alpha, lamb):
    return a*x**(-alpha)*np.exp(-lamb*x)

p0 = (np.max(hist), 1.5, 0.01)
if(np.sum(hist)>0):
    try:
        popt, pcov = curve_fit(power_law, bins_center, hist, p0=p0, maxfev=5000)
     
        sns.set_theme(style='darkgrid')
        plt.figure(figsize=(7,5))
        plt.loglog(bins_center, hist, 'o', color='red', label='Data')
        plt.loglog(bins_center, power_law(bins_center, *popt), 'b--', label=f'Power law fit\nalpha={popt[1]:.2f}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Avalanche size')
        plt.ylabel('Frequency')
        plt.title('Avalanche size distribution')    
        plt.show()
        print(f"Fitted power law parameters: a = {popt[0]}, alpha = {popt[1]}")
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
else:
    print("Not enough data to fit a power law.")
