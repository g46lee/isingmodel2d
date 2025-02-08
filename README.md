# 2D Ising Model Simulation

This repository implements the **2D Ising Model** using the **Metropolis Monte Carlo Algorithm** in Python. 
It explores the behaviors of ferromagnetic systems undergoing phase transitions, with a focus on critical phenomena, magnetization, and thermodynamic properties.

This project was originally developed for an international science challenge (ISEC) and was optimized in 2024 to improve computational efficiency, visualization, and performance. 
For a detailed explanation of the methodology, reuslt, and the project as a whole, kindly refer to the project report "Ising_model.pdf".

---

## Features
- **Lattice Simulation**: Models a 2D lattice of spins (+1, -1) with periodic boundary conditions.
- **Phase Transition Analysis**: Simulates phase transitions and computes magnetization, energy, heat capacity, and magnetic susceptibility.
- **Critical Temperature Estimation**: Determines the Curie temperature (Tc) where phase transitions occur.
- **Visualization**: Generates plots such as bifurcation diagrams and magnetization vs. temperature graphs.
- **Analytical Validation**: Compares simulation results against Onsager’s exact solution.

---

## Repository Structure
```plaintext
.
├── main.py            # Main script to run the simulation
├── figures.py         # Functions for generating plots
├── graphs.py          # Graphical and analytical data processing
├── methods.py         # Core simulation logic, including Monte Carlo sweeps
├── models.py          # Lattice setup and energy calculations
├── Ising_Model.pdf    # Detailed project report
