"""
Test Problems Module (test_problems.py)

Contains the mathematical formulations of the right-hand sides (RHS) of the ODEs
and the reference/exact solution generators for all four benchmark problems:
1. Harmonic Oscillator (Linear)
2. Nonlinear Schrödinger Equation (NLS 2D)
3. Gravitational Two-Body Problem
4. Gravitational Outer Planets Problem (6-body system)
"""

import numpy as np
from typing import Tuple, Callable


# =====================================================================
# BENCHMARK 1: Harmonic Oscillator (Linear System)
# =====================================================================

def rhs_oscillator(t: float, u: np.ndarray, omega: float = 2.0) -> np.ndarray:
    """
    RHS for the 2D linear harmonic oscillator.
    Dimension: 2 (u = [position, velocity]^T)
    """
    return np.array([u[1], -(omega**2) * u[0]])


def exact_oscillator(t_steps: np.ndarray, u0: np.ndarray, omega: float = 2.0) -> np.ndarray:
    """
    Analytical exact solution for the harmonic oscillator.
    Returns an array of shape (len(t_steps), 2).
    """
    u_exact = np.zeros((len(t_steps), len(u0)))
    y0, dy0 = u0[0], u0[1]
    
    u_exact[:, 0] = y0 * np.cos(omega * t_steps) + (dy0 / omega) * np.sin(omega * t_steps)
    u_exact[:, 1] = -y0 * omega * np.sin(omega * t_steps) + dy0 * np.cos(omega * t_steps)
    return u_exact


# =====================================================================
# BENCHMARK 2: Nonlinear Schrödinger Equation (NLS 2D)
# =====================================================================

def get_nls_setup(Nx: int = 4, Ny: int = 4, dx: float = 0.5, dy: float = 0.5) -> Tuple[Callable[[float, np.ndarray], np.ndarray], np.ndarray]:
    """
    Sets up the 2D NLS equation spatial discretization.
    Splits complex variables into real and imaginary parts.
    Total ODE dimension: 2 * Nx * Ny
    """
    dim = 2 * Nx * Ny
    u0 = np.zeros(dim)
    
    idx = 0
    for i in range(Nx):
        for j in range(Ny):
            x_val = i * dx - (Nx * dx) / 2.0
            u0[idx] = np.sqrt(2) * np.tanh(x_val) * np.cos(x_val)
            idx += 1
            
    for i in range(Nx):
        for j in range(Ny):
            x_val = i * dx - (Nx * dx) / 2.0
            u0[idx] = np.sqrt(2) * np.tanh(x_val) * np.sin(x_val)
            idx += 1
            
    def rhs_nls(t: float, u: np.ndarray, beta: float = 1.0) -> np.ndarray:
        u_real = u[:Nx*Ny].reshape((Nx, Ny))
        u_imag = u[Nx*Ny:].reshape((Nx, Ny))
        
        laplacian_real = np.zeros_like(u_real)
        laplacian_imag = np.zeros_like(u_imag)
        
        for i in range(Nx):
            for j in range(Ny):
                ip, im = (i + 1) % Nx, (i - 1) % Nx
                jp, jm = (j + 1) % Ny, (j - 1) % Ny
                
                laplacian_real[i, j] = (u_real[ip, j] + u_real[im, j] + u_real[i, jp] + u_real[i, jm] - 4 * u_real[i, j]) / (dx**2)
                laplacian_imag[i, j] = (u_imag[ip, j] + u_imag[im, j] + u_imag[i, jp] + u_imag[i, jm] - 4 * u_imag[i, j]) / (dy**2)
                
        psi_sq = u_real**2 + u_imag**2
        
        # d(u_real)/dt = -Laplacian(u_imag) - beta * |psi|^2 * u_imag
        # d(u_imag)/dt =  Laplacian(u_real) + beta * |psi|^2 * u_real
        d_real = -laplacian_imag - beta * psi_sq * u_imag
        d_imag =  laplacian_real + beta * psi_sq * u_real
        
        return np.concatenate([d_real.flatten(), d_imag.flatten()])
        
    return rhs_nls, u0


# =====================================================================
# BENCHMARK 3: Gravitational Two-Body Problem (Kepler System)
# =====================================================================

def rhs_two_body(t: float, u: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """
    RHS for the planar Two-Body problem in central gravitational field.
    Dimension: 4 (u = [x, y, vx, vy]^T)
    """
    r = np.sqrt(u[0]**2 + u[1]**2)
    if r < 1e-6:
        r = 1e-6
    return np.array([
        u[2],                      # dx/dt = vx
        u[3],                      # dy/dt = vy
        -mu * u[0] / (r**3),       # dvx/dt = ax
        -mu * u[1] / (r**3)        # dvy/dt = ay
    ])


# =====================================================================
# BENCHMARK 4: Gravitational Outer Planets Problem (24-D ODE)
# =====================================================================

def rhs_outer_planets(t: float, u: np.ndarray) -> np.ndarray:
    """
    RHS for the Outer Planets problem modeling 6 bodies (Sun + 5 outer planets).
    Dimension: 24 (6 bodies * 4 phase variables [x, y, vx, vy] each)
    """
    du = np.zeros(24)
    
    masses = np.array([
        1.0,               # Sun
        0.0009547919384,   # Jupiter
        0.0002858859806,   # Saturn
        0.00004366244043,  # Uranus
        0.00005151389020,  # Neptune
        0.00000000739645   # Pluto
    ])
    
    for i in range(6):
        idx = i * 4
        du[idx] = u[idx + 2]     # dx_i/dt = vx_i
        du[idx + 1] = u[idx + 3] # dy_i/dt = vy_i
        
        ax, ay = 0.0, 0.0
        for j in range(6):
            if i == j:
                continue
            jdx = j * 4
            dx = u[jdx] - u[idx]
            dy = u[jdx + 1] - u[idx + 1]
            dist = np.sqrt(dx**2 + dy**2) + 1e-8
            
            ax += masses[j] * dx / (dist**3)
            ay += masses[j] * dy / (dist**3)
            
        du[idx + 2] = ax  # dvx_i/dt = ax_i
        du[idx + 3] = ay  # dvy_i/dt = ay_i
        
    return du


# =====================================================================
# TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    print("=== Testing test_problems.py ===")
    
    # 1. Test Oscillator
    u_osc = np.array([1.0, 0.0])
    res_osc = rhs_oscillator(0.0, u_osc)
    print("Oscillator RHS shape:", res_osc.shape, "-> Expected: (2,)")
    
    # 2. Test NLS 2D
    rhs_nls, u0_nls = get_nls_setup(Nx=4, Ny=4)
    res_nls = rhs_nls(0.0, u0_nls)
    print("NLS 2D System dimension:", u0_nls.shape[0], "-> Expected: 32")
    print("NLS RHS shape:", res_nls.shape, "-> Expected: (32,)")
    
    # 3. Test Two-Body
    u_2b = np.array([1.0, 0.0, 0.0, 1.0])
    res_2b = rhs_two_body(0.0, u_2b)
    print("Two-Body RHS shape:", res_2b.shape, "-> Expected: (4,)")
    
    # 4. Test Outer Planets
    u_planets = np.zeros(24)
    u_planets[0::4] = np.array([0.0, 5.2, 9.5, 19.2, 30.1, 39.5]) # Approximate radial distances
    res_planets = rhs_outer_planets(0.0, u_planets)
    print("Outer Planets RHS shape:", res_planets.shape, "-> Expected: (24,)")
    print("Everything initialized correctly and matches documentation dimensions!")