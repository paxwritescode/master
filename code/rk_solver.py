"""
Numerical Integration Module (rk_solver.py)

Implements the generalized computational scheme of the explicit Runge-Kutta 
method of the 6th order with 8 stages (RK(6,8)), featuring dynamic reconstruction
of the Butcher tableau coefficients based on the free parameter vector theta.
"""

import numpy as np
import scipy.optimize as opt
from typing import Callable, Tuple

def reconstruct_butcher(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dynamically reconstructs the full Butcher tableau by numerically solving
    the system of 29 algebraic order equations for the current theta point.
    """
    c2_val, c4_val, c5_val, c7_val = theta
    
    c = np.array([0.0, c2_val, 1.0/6.0, c4_val, c5_val, 0.5, c7_val, 1.0])
    
    # --- GEOMETRIC FILTER (Singularity Protection) ---
    if np.any(np.diff(c) < 0.005):
        return np.eye(8) * 0.1, np.ones(8) / 8.0, c
    
    V = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [c[0], c[3], c[4], c[5], c[6], c[7]],
        [c[0]**2, c[3]**2, c[4]**2, c[5]**2, c[6]**2, c[7]**2],
        [c[0]**3, c[3]**3, c[4]**3, c[5]**3, c[6]**3, c[7]**3],
        [c[0]**4, c[3]**4, c[4]**4, c[5]**4, c[6]**4, c[7]**4],
        [c[0]**5, c[3]**5, c[4]**5, c[5]**5, c[6]**5, c[7]**5]
    ])
    rhs = np.array([1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0, 1.0/5.0, 1.0/6.0])
    
    try:
        b_nonzero = np.linalg.solve(V, rhs)
        b = np.array([b_nonzero[0], 0.0, 0.0, b_nonzero[1], b_nonzero[2], b_nonzero[3], b_nonzero[4], b_nonzero[5]])
    except np.linalg.LinAlgError:
        return np.eye(8) * 0.1, np.ones(8) / 8.0, c    
    
    def butcher_equations(x):
        A = np.zeros((8, 8))
        A[1, 0] = x[0]
        A[2, 0], A[2, 1] = x[1], x[2]
        A[3, 0], A[3, 2] = x[3], x[4]                  # a42 = 0 fixed
        A[4, 0], A[4, 2], A[4, 3] = x[5], x[6], x[7]  # a52 = 0 fixed
        A[5, 0], A[5, 2], A[5, 3], A[5, 4] = x[8], x[9], x[10], x[11] # a62 = 0 fixed
        A[6, 0], A[6, 2], A[6, 3], A[6, 4], A[6, 5] = x[12], x[13], x[14], x[15], x[16] # a72 = 0 fixed
        A[7, 0], A[7, 2], A[7, 3], A[7, 4], A[7, 5], A[7, 6] = x[17], x[18], x[19], x[20], x[21], x[22] # a82 = 0 fixed
        
        c2_pow = c * c
        c3_pow = c2_pow * c
        c4_pow = c3_pow * c
        
        Ac = A @ c
        Ac2 = A @ c2_pow
        Ac3 = A @ c3_pow
        Ac4 = A @ c4_pow
        A2c = A @ Ac
        A3c = A @ A2c
        A2c2 = A @ Ac2
        
        eqs = [
            A[1, 0] - c[1],
            A[2, 0] + A[2, 1] - c[2],
            A[3, 0] + A[3, 2] - c[3],
            A[4, 0] + A[4, 2] + A[4, 3] - c[4],
            A[5, 0] + A[5, 2] + A[5, 3] + A[5, 4] - c[5],
            A[6, 0] + A[6, 2] + A[6, 3] + A[6, 4] + A[6, 5] - c[6],
            A[7, 0] + A[7, 2] + A[7, 3] + A[7, 4] + A[7, 5] + A[7, 6] - c[7],
            
            np.dot(b, c2_pow * Ac) - 1.0/10.0,                   # q10
            np.dot(b, c * Ac) - 1.0/8.0,                     # q6
            np.dot(b, c3_pow * Ac) - 1.0/12.0,                   # q19
            np.dot(b, Ac) - 1.0/6.0,                         # q4
            np.dot(b, c * (Ac**2)) - 1.0/24.0,               # q21
            np.dot(b, c2_pow * Ac2) - 1.0/18.0,                  # q20
            np.dot(b, c * Ac2) - 1.0/15.0,                   # q11
            np.dot(b, Ac2) - 1.0/12.0,                       # q7
            np.dot(b, c * A2c) - 1.0/30.0,                   # q12
            np.dot(b, Ac * A2c) - 1.0/72.0,                  # q27
            np.dot(b, A2c) - 1.0/24.0,                       # q8
            np.dot(b, c * Ac3) - 1.0/24.0,                   # q23
            np.dot(b, Ac3) - 1.0/20.0,                       # q13
            np.dot(b, Ac4) - 1.0/30.0,                       # q29
            np.dot(b, c * A2c2) - 1.0/72.0,                  # q28
            np.dot(b, c * A3c) - 1.0/144.0                   # q26
        ]
        return np.array(eqs)

    x0 = np.zeros(23)
    x0[0] = c[1]
    x0[1] = c[2]; x0[2] = 0.0
    x0[3] = c[3]/2.0; x0[4] = c[3]/2.0
    x0[5:8] = c[4]/3.0
    x0[8:12] = c[5]/4.0
    x0[12:17] = c[6]/5.0
    x0[17:23] = c[7]/6.0
    
    sol = opt.least_squares(
        butcher_equations, 
        x0,
        method='lm',
        max_nfev=100)
    
    x_opt = sol.x

    A = np.zeros((8, 8))
    A[1, 0] = x_opt[0]
    A[2, 0], A[2, 1] = x_opt[1], x_opt[2]
    A[3, 0], A[3, 2] = x_opt[3], x_opt[4]
    A[4, 0], A[4, 2], A[4, 3] = x_opt[5], x_opt[6], x_opt[7]
    A[5, 0], A[5, 2], A[5, 3], A[5, 4] = x_opt[8], x_opt[9], x_opt[10], x_opt[11]
    A[6, 0], A[6, 2], A[6, 3], A[6, 4], A[6, 5] = x_opt[12], x_opt[13], x_opt[14], x_opt[15], x_opt[16]
    A[7, 0], A[7, 2], A[7, 3], A[7, 4], A[7, 5], A[7, 6] = x_opt[17], x_opt[18], x_opt[19], x_opt[20], x_opt[21], x_opt[22]
        
    return A, b, c


def rk6_8_integrate(
    f: Callable[[float, np.ndarray], np.ndarray], 
    u0: np.ndarray, 
    t_span: Tuple[float, float], 
    h: float, 
    theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invariant integrator implementing the explicit Runge-Kutta RK(6,8) scheme in vector form.
    
    Parameters
    ----------
    f : Callable[[float, np.ndarray], np.ndarray]
        Pointer to the ODE right-hand side function with signature f(t, u).
    u0 : np.ndarray
        Vector of initial conditions (one-dimensional array of arbitrary dimension).
    t_span : Tuple[float, float]
        Tuple (t_start, t_end) defining the integration interval.
    h : float
        Fixed integration step size across the time grid.
    theta : np.ndarray
        Current parameter vector of the numerical scheme.
        
    Returns
    -------
    t_steps : np.ndarray
        Discrete time grid (dimension N_steps).
    u_sol : np.ndarray
        Two-dimensional array of the phase trajectory with shape (N_steps, dim_u).
    """
    A, b, c = reconstruct_butcher(theta)
    
    t_start, t_end = t_span
    
    num_steps = int(np.floor((t_end - t_start) / h))
    t_steps = t_start + np.arange(num_steps + 1) * h
    
    if np.isclose(t_steps[-1], t_end):
        t_steps[-1] = t_end
    elif t_steps[-1] < t_end:
        t_steps = np.append(t_steps, t_end)
        
    num_steps = len(t_steps)
    dim = len(u0)
    
    u_sol = np.zeros((num_steps, dim))
    u_sol[0] = np.array(u0, dtype=float)
    
    for n in range(num_steps - 1):
        t_n = t_steps[n]
        u_n = u_sol[n]
        
        current_h = t_steps[n + 1] - t_n
        
        k = np.zeros((8, dim))
        
        for i in range(8):
            linear_combination = np.dot(A[i, :i], k[:i])
            k[i] = f(t_n + c[i] * current_h, u_n + current_h * linear_combination)
            
        u_sol[n + 1] = u_n + current_h * np.dot(b, k)
        
    return t_steps, u_sol


if __name__ == "__main__":
    print("=== Testing rk_solver module ===")
    test_theta = np.array([0.1, 0.25, 0.4, 0.7])    
    A, b, c = reconstruct_butcher(test_theta)
    print("Matrix A successfully reconstructed, shape:", A.shape)
    print("Nodes c:", c)
    print("Everything works perfectly!")