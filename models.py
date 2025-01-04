"""
This module contains core classes and functions of the project.

Classes:
    Ising2D

Functions:
    boltzmann(float, float) - float
    cogr(np.ndarray, int) - np.ndarray

Constants:
    k - Boltzmann constant
    J - coupling constant
"""

import numpy as np
from scipy.stats import mode
k, J = 1, 1

def boltzmann(energy, T):
    """
    Performs Boltzmann distribution function.

    Parameters
    ----------
    energy: float
    T: float

    Returns
    -------
    boltzmann: float
    """
    return np.exp(-energy / (T*k))


def cogr(state: np.ndarray, size: int) -> np.ndarray:
    """
    Perform coarse-graining on the input lattice state.

    Parameters
    ----------
    state: np.ndarray
        2D array representing the lattice.
    size: int 
        size of the coarse-grained blocks.

    Returns
    ----------
    coarse_grained: np.ndarray
        a coarse-grained lattice as a smaller 2D array.
    """
    if state.shape[0] % size != 0 or state.shape[1] % size != 0:
        raise ValueError("Lattice size must be divisible by coarse-graining size.") # debugging
    
    reshaped = state.reshape(state.shape[0] // size, size, state.shape[1] // size, size)
    reshaped = reshaped.transpose(0, 2, 1, 3).reshape(-1, size*size)
    block_modes = mode(reshaped, axis=1, nan_policy="omit").mode
    coarse_grained = block_modes.reshape(state.shape[0]//size, state.shape[1] // size)
    return coarse_grained


class Ising2D:
    """
    Implementation of 2D Ising model using Monte Carlo Metropolis method. Represents
    N*N square lattice, where each point takes either +1 or -1 (up or down magnetic spin).
    """
    def __init__(self, N: int, T: float, init_state=None, h=0):
        self.N = N
        self.T = T
        self.h = h
        self.state = (np.random.choice([-1, 1], (N, N)) if init_state is None else np.copy(init_state))

    def get_total_energy(self):
        """Calculate the total energy of the lattice."""
        energy = 0
        for i in range(self.N):
            for j in range(self.N):
                energy += self.spin_energy((i, j))
        return energy / 2

    def spin_energy(self, pos):
        """Calculate the energy of a single spin at a given position."""
        i, j = pos
        neighbors = (self.state[(i + 1) % self.N, j]
                    + self.state[(i - 1) % self.N, j]
                    + self.state[i, (j + 1) % self.N]
                    + self.state[i, (j - 1) % self.N])
        return -J * self.state[i, j] * neighbors - self.h * self.state[i, j]

    def step(self):
        """Perform one Metropolis Monte Carlo step and update it"""
        for _ in range(self.N * self.N):
            i, j = np.random.randint(0, self.N, 2)  #choose random spin
            delta_e = 2 * self.spin_energy((i, j))  #calculate energy difference
            if delta_e < 0 or boltzmann(delta_e, self.T) > np.random.rand(): #update matrix
                self.state[i, j] *= -1