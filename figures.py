"""
This module develops figures to demonstrate coarse-graining effects by comparing lattice states and coarse grained states.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from methods import coarse_graining

# Create folder to save figures
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures (Course-graining)")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Parameters
N = 729  # Lattice size
N_iter = 1000  # No. of Monte Carlo steps
No_coarse_grainings = 4  # No. of coarse-graining steps
block_size = 3  # Size of coarse-grained blocks
temperatures = [2.269 - 0.3, 2.269, 2.269 + 0.3] # Temperatures for the three states

fig, ax = plt.subplots(3, 2, figsize=(10, 15))
for i, T in enumerate(temperatures):
    states = coarse_graining(T, N, N_iter, No_coarse_grainings, block_size)
    
    # Initial lattice state(s) (left)
    ax[i, 0].imshow(states[0], cmap="gray")
    ax[i, 0].set_title(f"T = {T:.3f} - Original")
    ax[i, 0].axis("off")

    # Coarse-grained state(s) (right)
    ax[i, 1].imshow(states[-1], cmap="gray")
    ax[i, 1].set_title(f"T = {T:.3f} - Coarse-grained")
    ax[i, 1].axis("off")
    
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "coarse_graining_combined.png"))
