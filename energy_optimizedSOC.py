### Energy optimized SOC in neural networks based on Ising model

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import make_interp_spline

### Parameters
n = 500
k = 10
num_iteration = 1000
T = 100
dt = 0.1

### This is an identical graph structure generally used in Abelian Sandpile model
def initialize_graph(n,k):
    G = nx.erdos_renyi_graph(n, 0.3)
    a = [f'a{i+1}' for i in range(k)]   ### The set {a1, a2, ..., ak}
    s = {i: np.random.choice(a) for i in G.nodes()}
    return G, s

def energy(G,s):
    E = 0
    for i in G.nodes():
        for j in G.neighbors(i):
               if(s[i] != s[j]):   ### Reducing the misalignments in the spins 
                   E += 1
    return E

def simulate_avalanche(s,G,k):
    size = np.random.randint(1, len(G)//2)
    nodes = list(G.nodes())
    effected_nodes = np.random.choice(nodes, size, replace=False)
    updated_s = s.copy()
    
    for node in effected_nodes:
        if(np.random.rand() < 0.5):
            updated_s[node] = np.random.choice([f'a{i+1}' for i in range(k)])
        
    return updated_s, effected_nodes

def run_SOC(G,s,num_iteration,k):
    energy_over_time = []
    E = energy(G,s)
    for i in range(num_iteration):
        updated_s, _ = simulate_avalanche(s,G,k)
        new_E = energy(G,updated_s)
        if(new_E < E):
            s = updated_s
            E = new_E
            energy_over_time.append((i,E))
    return energy_over_time, s

G, s = initialize_graph(n,k)

energy_over_time, s = run_SOC(G,s,num_iteration,k)

### Plots
plt.figure(figsize=(7,7))
num_iteration, energies = zip(*energy_over_time)
smooth_iteration = np.linspace(min(num_iteration), max(num_iteration), 1000)
spline = make_interp_spline(num_iteration, energies)
smooth_energies = spline(smooth_iteration)
plt.plot(smooth_iteration, smooth_energies, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('Energy over time in SOC avalanche mechanism')
plt.show()

plt.figure(figsize=(7,7))
plt.hist([s[i] for i in s.keys()], bins=k, edgecolor='black', color='lightcoral', alpha=0.9)
plt.title('Histogram of Spin Values After Optimization')
plt.show()