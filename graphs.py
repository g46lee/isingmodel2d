"""
This module contains functions for generarating graphical data and allows visual analysis of physical phenomena. 

Functions:
    reduced_M_vs_T()
    avg_energy_per_spin_vs_T()
    magnetic_susceptibility_vs_T()
    heat_capacity_vs_T()
    reduced_M_vs_h()
    fork_like_graph()
    reduced_M_with_onsager()
    log_reduced_M()
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from methods import onsager_formula
from main import start_temp, stop_temp, num_temp, start_h, stop_h, num_h

# Create a folder to save graphs
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Graphs")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load values from main.py
M0 = np.loadtxt("data/M0.txt")
E0 = np.loadtxt("data/E0.txt")
X0 = np.loadtxt("data/X0.txt")
C0 = np.loadtxt("data/C0.txt")
M1 = np.loadtxt("data/M0_1.txt")
M0_seeds = np.loadtxt("data/M0_seeds.txt").reshape(5, -1)  # Reshape back to the original structure


# Graph 1: Reduced Magnetization vs. Temperature
def reduced_M_vs_T():
    temp = np.linspace(start_temp, stop_temp, num_temp)
    plt.figure()
    plt.scatter(temp, np.mean(M0, axis=0), marker='o', c='black', label="Magnetization")
    plt.xlabel("Temperature ($T/k_B$)")
    plt.ylabel("Magnetization (M)")
    plt.title("Reduced Magnetization vs. Temperature")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "reduced_M_vs_T.png"))


# Graph 2: Energy vs. Temperature
def avg_energy_per_spin_vs_T():
    temp = np.linspace(start_temp, stop_temp, num_temp)
    plt.figure()
    plt.scatter(temp, np.mean(E0, axis=0), marker='o', c='black', label="Energy")
    plt.xlabel("Temperature ($T/k_B$)")
    plt.ylabel("Average Energy per Spin ($E/J$)")
    plt.title("Average Energy per Spin vs. Temperature")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "avg_energy_per_spin_vs_T.png"))

# Graph 3: Magnetic Susceptibility vs. Temperature
def magnetic_susceptibility_vs_T():
    temp = np.linspace(start_temp, stop_temp, num_temp)
    plt.figure()
    plt.scatter(temp, X0, marker='o', c='black', label="Susceptibility (χ)")
    plt.xlabel("Temperature ($T/k_B$)")
    plt.ylabel("Magnetic Susceptibility")
    plt.title("Magnetic Susceptibility vs. Temperature")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "magnetic_susceptibility_vs_T.png"))

# Graph 4: Heat Capacity vs. Temperature
def heat_capacity_vs_T():
    temp = np.linspace(start_temp, stop_temp, num_temp)
    plt.figure()
    plt.scatter(temp, C0, marker='o', c='black', label="Heat Capacity (C)")
    plt.xlabel("Temperature ($T/k_B$)")
    plt.ylabel("Heat Capacity (C)")
    plt.title("Heat Capacity vs. Temperature")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "heat_capacity_vs_T.png"))

# Graph 5: Reduced Magnetization vs. External Magnetic Field
def reduced_M_vs_h():
    fields = np.linspace(start_h, stop_h, num_h)
    plt.figure()
    plt.scatter(fields, M1, marker='o', c='black', label="Magnetization")
    plt.xlabel("Magnetic Field (h in units of $J / \mu_B$)")
    plt.ylabel("Magnetization (M)")
    plt.title("Reduced Magnetization vs Magnetic Field")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "reduced_M_vs_h.png"))

# Fork-like Graph: Magnetization vs. Temperature with Different Seeds
def fork_like_graph():
    temp = np.linspace(start_temp, stop_temp, len(M0_seeds[0]))
    plt.figure()
    for i, M0_seed in enumerate(M0_seeds):  # Iterate over each seed
        plt.scatter(temp, M0_seed, s=10, label=f"Seed {i + 1}")
    plt.xlabel("Temperature ($T/k_B$)")
    plt.ylabel("Magnetization (M)")
    plt.title("Fork-like Graph: Magnetization vs. Temperature")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "fork_like_graph.png"))

# Graph 6: Reduced Magnetization vs. Temperature with Onsager Formula
def reduced_M_with_onsager():
    temp = np.linspace(start_temp, stop_temp, len(M0[0]))
    reduced_M = np.mean(M0, axis=0) / np.max(np.mean(M0, axis=0))
    plt.figure()
    plt.scatter(temp, reduced_M, color='black', s=10, label="Simulation Data")

    # Plot Onsager formula
    onsager_values = [onsager_formula(T) for T in temp]
    plt.plot(temp, onsager_values, color='green', label="Onsager Formula")

    # Plot critical temperature line
    plt.axvline(x=2.269, color='red', linestyle='--', label="Predicted $T_{crit}$")
    plt.text(2.3, 0.5, "$T_{crit}$", color='red')  # Label for T_crit

    plt.xlabel("Temperature ($T / k_B$)")
    plt.ylabel("$M/M_0$")
    plt.title("Reduced Magnetization vs Temperature (Onsager Formula)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "Onsager_formula_comparison.png"))

# Graph 7: Log-Log Plot of Reduced Magnetization vs. Reduced Temperature
def log_reduced_M():
    temp = np.linspace(start_temp, stop_temp, len(M0[0]))
    reduced_temp = (2.269 - temp)/ 2.269
    reduced_M = np.mean(M0, axis=0) / np.max(np.mean(M0, axis=0))
    mask = reduced_temp > 0 # Mask out unphysical values
    reduced_temp = reduced_temp[mask]
    reduced_M = reduced_M[mask]

    # Log-log graph
    plt.figure()
    plt.loglog(reduced_temp, reduced_M, 'o', color='black', label="Simulation Data")
    
    # Onsager line for comparison
    plt.plot(reduced_temp, reduced_temp ** 0.125, color='green', label="Onsager Formula")

    plt.xlabel("Reduced Temperature ($T_{crit} - T/T_{crit}$)")
    plt.ylabel("Reduced Magnetization (m)")
    plt.title("Log-Log Plot: Reduced Magnetization vs. Reduced Temperature")
    plt.legend()
    plt.grid(which="both")
    plt.savefig(os.path.join(save_dir, "log_plot_Magn_Temp.png"))


if __name__ == "__main__":
    reduced_M_vs_T()
    avg_energy_per_spin_vs_T()
    magnetic_susceptibility_vs_T()
    heat_capacity_vs_T()
    reduced_M_vs_h()
    fork_like_graph()
    reduced_M_with_onsager()
    log_reduced_M()
