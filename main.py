"""
This model contains calculations for data derivations of functions defined in methods.py based on assigned parameters.
Data sets are saved into text files for efficient data logging and to develop graphs and figures. 

See also 
--------
graph.py 
figures.py
"""

import os
import numpy as np
from methods import variance_with_t, variance_with_h

# Create a folder to save data
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Calculation 1: Temperature-dependent analysis
N = 50  # Lattice size
start_temp, stop_temp, num_temp = 0.5, 3.5, 50 # Temperature range
N_iter, equil = 1001, 100 
M0, E0, X0, C0 = variance_with_t(start_temp, stop_temp, num_temp, N_iter, equil, N)
np.savetxt(os.path.join(save_dir, "M0.txt"), M0, fmt="%.6f") # save values to txt file(s)
np.savetxt(os.path.join(save_dir, "E0.txt"), E0, fmt="%.6f")
np.savetxt(os.path.join(save_dir, "X0.txt"), X0, fmt="%.6f")
np.savetxt(os.path.join(save_dir, "C0.txt"), C0, fmt="%.6f")

# Calculation 2: Magnetic field-dependent analysis
start_h, stop_h, num_h = -2.0, 0.0, 50  # Magnetic field range
M1 = variance_with_h(start_h, stop_h, num_h, N, T=1.5)
np.savetxt(os.path.join(save_dir, "M0_1.txt"), M1, fmt="%.6f") # save values to txt file

# Calculation 3: Fork-like graph data
M0_seeds = []
for seed in range(5):
    np.random.seed(seed)
    M0_seed, _, _, _ = variance_with_t(start_temp, stop_temp, num_temp, N_iter, equil, N)
    M0_seeds.append(M0_seed)
M0_seeds = np.array(M0_seeds)
np.savetxt(os.path.join(save_dir, "M0_seeds.txt"), M0_seeds.reshape(seed, -1)) # save values to txt file fmt="%.6f"
