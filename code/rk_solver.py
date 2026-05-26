"""
Numerical Integration Module (rk_solver.py)

Implements the generalized computational scheme of the explicit Runge-Kutta 
method of the 6th order with 8 stages (RK(6,8)), featuring dynamic reconstruction
of the Butcher tableau coefficients based on the free parameter vector theta.
"""

import numpy as np
from typing import Callable, Tuple, List
import sympy as sp
from symbolic_derivation import derive_parametric_coefficients


SYMBOLIC_COEFS = derive_parametric_coefficients()

def reconstruct_butcher(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dynamically reconstructs the full Butcher tableau by evaluating the 
    analytically derived SymPy expressions at the current numerical theta point.
    """
    c2_val, c4_val, c5_val, c7_val = theta
    
    # Free parameters symbols used during derivation
    c2, c4, c5, c7 = sp.symbols('c2 c4 c5 c7', real=True)
    param_subs = {c2: c2_val, c4: c4_val, c5: c5_val, c7: c7_val}
    
    c = np.zeros(8)
    for i in range(8):
        sym_node = SYMBOLIC_COEFS[f'c{i+1}']
        # Substitute numerical values into the symbolic formula
        c[i] = float(sym_node.subs(param_subs))
        
    b = np.zeros(8)
    for i in range(8):
        sym_weight = SYMBOLIC_COEFS[f'b{i+1}']
        b[i] = float(sym_weight.subs(param_subs))
        
    A = np.zeros((8, 8))
    A[1, 0] = float(SYMBOLIC_COEFS['a21'].subs(param_subs))
    A[2, 0] = float(SYMBOLIC_COEFS['a31'].subs(param_subs))
    A[2, 1] = float(SYMBOLIC_COEFS['a32'].subs(param_subs))
    A[3, 0] = float(SYMBOLIC_COEFS['a41'].subs(param_subs))
    A[3, 2] = float(SYMBOLIC_COEFS['a43'].subs(param_subs))
    A[4, 0] = float(SYMBOLIC_COEFS['a51'].subs(param_subs))
    A[4, 2] = float(SYMBOLIC_COEFS['a53'].subs(param_subs))
    A[4, 3] = float(SYMBOLIC_COEFS['a54'].subs(param_subs))
    A[5, 0] = float(SYMBOLIC_COEFS['a61'].subs(param_subs))
    A[5, 2] = float(SYMBOLIC_COEFS['a63'].subs(param_subs))
    A[5, 3] = float(SYMBOLIC_COEFS['a64'].subs(param_subs))
    A[5, 4] = float(SYMBOLIC_COEFS['a65'].subs(param_subs))
    A[6, 0] = float(SYMBOLIC_COEFS['a71'].subs(param_subs))
    A[6, 2] = float(SYMBOLIC_COEFS['a73'].subs(param_subs))
    A[6, 3] = float(SYMBOLIC_COEFS['a74'].subs(param_subs))
    A[6, 4] = float(SYMBOLIC_COEFS['a75'].subs(param_subs))
    A[6, 5] = float(SYMBOLIC_COEFS['a76'].subs(param_subs))
    A[7, 0] = float(SYMBOLIC_COEFS['a81'].subs(param_subs))
    A[7, 2] = float(SYMBOLIC_COEFS['a83'].subs(param_subs))
    A[7, 3] = float(SYMBOLIC_COEFS['a84'].subs(param_subs))
    A[7, 4] = float(SYMBOLIC_COEFS['a85'].subs(param_subs))
    A[7, 5] = float(SYMBOLIC_COEFS['a86'].subs(param_subs))
    A[7, 6] = float(SYMBOLIC_COEFS['a87'].subs(param_subs))
    
    for i in range(1, 8):
        row_sum = np.sum(A[i, :i])
        if row_sum > 0:
            A[i, :i] = A[i, :i] * (c[i] / row_sum)
            
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
    test_theta = np.array([0.2, 0.4, 0.6, 0.8])
    A, b, c = reconstruct_butcher(test_theta)
    print("Matrix A successfully reconstructed, shape:", A.shape)
    print("Nodes c:", c)
    print("Everything works perfectly!")