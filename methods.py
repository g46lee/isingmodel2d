"""
This module contains tools for examining properties of ferromagnetic systems
using utilities from main.py module.

Functions:
    variance_with_t
    variance_with_h
    coarse_graining
    onsager_formula
"""

import numpy as np
from models import Ising2D, cogr

def variance_with_t(start, stop, num, N_iter=1001, equil=100, N=50, init_state=None, h=0):
    """
    Classic Metropolis Implementation, describes variance of parameters of Ising Model with
    temperature.

    Parameters
    ----------
    start, stop, num: float
        range of temperature points to use
    N_iter, equil: int (optional)
        number of iterations for Metropolis algorithm and for equilibration respectively
    N: int, init_state: np.ndarray, h: float (optional)
        parameters for Ising2D class initialisation
    
    Returns
    -------
    M0, E0, X0, C0: tuple
        average magnetisation, average energy,
        susceptibility and heat capacity of the Ising Model with respect to T

    See also
    --------
    variance with h
    """
    E0, M0 = np.ones(num), np.ones(num)
    for i in range(5):
        E, M = [],[]

        for t in np.linspace(start, stop, num):
            np.random.seed(i)
            lat = Ising2D(N, round(t, 2), init_state=init_state, h=h)
            M1, E1 = 0,0

            for _ in range(equil):
                lat.step()
            for j in range(1, N_iter):
                lat.step()
                if j % 100 == 0:
                    M1 += np.mean(lat.state)
                    E1 += lat.get_total_energy()
            E.append(E1 / (N * N * (N_iter // 100)))
            M.append(M1 / (N_iter // 100))

        E0 = np.vstack((E0, E))
        M0 = np.vstack((M0, M))

    M0 = M0[1:]
    E0 = E0[1:]
    X0 = np.var(M0, axis=0)
    C0 = np.var(E0, axis=0)
    return M0, E0, X0, C0


def variance_with_h(start, stop, num, N, T, init_state=None, equil=100, N_iter=1001):
    """
    Function for studying variance of properties of Ising Model with the strength
    of external magnetic field with constant temperature.
    Returns average magnetisation with respect to h.

    Parameters
    ----------
    start, stop, num: float
        range of magnetic field points to use
    N_iter, equil: int (optional)
        number of iterations for Metropolis algorithm and for equilibration respectively
    N: int, init_state: np.ndarray (optional)
        parameters for Ising2D class initialisation
    T: float
        temperature of the system
    
    Returns
    -------
    M0: np.ndarray
        average magnetisation with respect to h

    See also
    --------
    variance_with_t
    """
    M0 = []
    for h in np.linspace(start, stop, num):
        M = []
        lat = Ising2D(N=N, T=T, init_state=init_state, h=h)
        for _ in range(equil):
            lat.step()
        for j in range(N_iter):
            lat.step()
            if j % 100 == 0:
                M.append(np.mean(lat.state))

        M0.append(np.mean(M))
    return np.array(M0)


def coarse_graining(t, N, N_iter, num, size):
    """
    Coarse graining algorithm performed on Ising2D class instance for
    a particular temperature value.

    Parameters
    ----------
        t: float
            temperature of the lattice.
        N: int
            size of the lattice.
        N_iter: int
            no. of Monte Carlo steps.
        num: int 
            no. of coarse-graining iterations.
        size: int 
            size of coarse-grained blocks.

    Returns:
        states: list 
            list of lattice states (original and coarse-grained).
    """
    lat = Ising2D(N, T=t)

    # Simulate the lattice for N_iter steps
    for _ in range(N_iter):
        lat.step()
    
    # Save the initial state
    a = np.copy(lat.state)
    print(f"Initial lattice shape: {a.shape}")  # debugging
    states = [a]

    # Perform coarse-graining iteratively
    for i in range(num):
        print(f"Coarse-graining iteration {i + 1}: Lattice shape before = {a.shape}")  # debugging
        if a.shape[0] % size != 0 or a.shape[1] % size != 0:
            raise ValueError(f"Lattice shape {a.shape} is not divisible by block size {size}.") # debugging
        a = cogr(a, size)
        print(f"Coarse-graining iteration {i + 1}: Lattice shape after = {a.shape}")  # debugging
        states.append(a)
    return states


def onsager_formula(T):
    """
    Onsager solution for the 2D Ising model.

    Parameters
    ----------
    - T: float
        temperature in units of J/k_B
    
    Returns:
    - Reduced magnetization according to Onsager's formula.
    """
    T_crit = 2.269  # Critical temperature
    if T < T_crit:
        return (1 - (np.sinh(2/T)) ** (-4)) ** 0.125
    else:
        return 0
    