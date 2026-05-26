"""
Quality Functional / Loss Function Module (loss_function.py)

Computes the scalar value of the generalized global error functional C(theta)
by executing the parametric RK(6,8) solver across all four benchmark problems
over a set of integration steps, normalizing results against the classical scheme.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List

# Import components from our previous modules
from rk_solver import rk6_8_integrate
from test_problems import rhs_oscillator, exact_oscillator, get_nls_setup, rhs_two_body, rhs_outer_planets

np.seterr(all='ignore')

THETA_CLASSICAL = np.array([0.2, 0.4, 0.6, 0.8])


def compute_problem_error(
    rhs_func, u0: np.ndarray, t_span: tuple, h: float, theta: np.ndarray, ref_sol: np.ndarray = None
) -> float:
    """
    Calculates the maximum Chebyshev norm of the global error for a specific step h.
    Gracefully handles mathematical degeneracies (e.g., division by zero in SymPy).
    """
    try:
        t_steps, u_num = rk6_8_integrate(rhs_func, u0, t_span, h, theta)
    except (TypeError, ZeroDivisionError, ValueError):
        # Return a massive penalty loss if the Butcher tableau degenerates
        return 1e5
    
    if ref_sol is None:
        sol_ref = solve_ivp(rhs_func, t_span, u0, method='DOP853', t_eval=t_steps, atol=1e-13, rtol=1e-13)
        if not sol_ref.success:
            return 1e5
        u_ref = sol_ref.y.T
    else:
        u_ref = ref_sol

    global_error = np.max(np.linalg.norm(u_num - u_ref, axis=1))
    
    if not np.isfinite(global_error):
        return 1e5
        
    return global_error


def compute_loss(theta: np.ndarray) -> float:
    """
    Evaluates the generalized objective functional C(theta) across all benchmarks.
    
    Parameters
    ----------
    theta : np.ndarray
        Current parameter vector [c_2, c_4, c_5, c_7]^T.
        
    Returns
    -------
    float
        Scalar value of the normalized aggregate error.
    """
    h_set = [0.05, 0.1]
    
    normalized_errors = []
    
    # -------------------------------------------------------------------------
    # BENCHMARK 1: Linear Harmonic Oscillator
    # -------------------------------------------------------------------------
    u0_osc = np.array([1.0, 0.0])
    t_span_osc = (0.0, 2.0)
    for h in h_set:
        t_steps_c, u_num_c = rk6_8_integrate(rhs_oscillator, u0_osc, t_span_osc, h, THETA_CLASSICAL)
        u_ref_osc = exact_oscillator(t_steps_c, u0_osc)
        err_classical = np.max(np.linalg.norm(u_num_c - u_ref_osc, axis=1))
        
        err_parametric = compute_problem_error(rhs_oscillator, u0_osc, t_span_osc, h, theta, ref_sol=u_ref_osc)
        
        denom = err_classical if err_classical > 1e-15 else 1e-15
        normalized_errors.append(err_parametric / denom)

    # -------------------------------------------------------------------------
    # BENCHMARK 2: Nonlinear Two-Body Kepler System
    # -------------------------------------------------------------------------
    u0_2body = np.array([1.0, 0.0, 0.0, 0.5]) # Eccentric orbit
    t_span_2body = (0.0, 3.0)
    for h in h_set:
        err_classical = compute_problem_error(rhs_two_body, u0_2body, t_span_2body, h, THETA_CLASSICAL)
        err_parametric = compute_problem_error(rhs_two_body, u0_2body, t_span_2body, h, theta)
        
        denom = err_classical if err_classical > 1e-15 else 1e-15
        normalized_errors.append(err_parametric / denom)

    # -------------------------------------------------------------------------
    # BENCHMARK 3: Nonlinear Schrödinger Equation (NLS 2D)
    # -------------------------------------------------------------------------
    rhs_nls, u0_nls = get_nls_setup(Nx=4, Ny=4) # Compact grid for rapid optimization optimization
    t_span_nls = (0.0, 0.2)
    for h in h_set:
        err_classical = compute_problem_error(rhs_nls, u0_nls, t_span_nls, h, THETA_CLASSICAL)
        err_parametric = compute_problem_error(rhs_nls, u0_nls, t_span_nls, h, theta)
        
        denom = err_classical if err_classical > 1e-15 else 1e-15
        normalized_errors.append(err_parametric / denom)

    # -------------------------------------------------------------------------
    # BENCHMARK 4: Gravitational Outer Planets System (24-D)
    # -------------------------------------------------------------------------
    u0_planets = np.zeros(24)
    u0_planets[0::4] = np.array([0.0, 5.2, 9.5, 19.2, 30.1, 39.5]) # X-coordinates
    u0_planets[3::4] = np.array([0.0, 0.013, 0.009, 0.006, 0.005, 0.004]) # VY-coordinates
    t_span_planets = (0.0, 5.0)
    for h in h_set:
        err_classical = compute_problem_error(rhs_outer_planets, u0_planets, t_span_planets, h, THETA_CLASSICAL)
        err_parametric = compute_problem_error(rhs_outer_planets, u0_planets, t_span_planets, h, theta)
        
        denom = err_classical if err_classical > 1e-15 else 1e-15
        normalized_errors.append(err_parametric / denom)

    return float(np.mean(normalized_errors))


# =====================================================================
# TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    print("=== Testing loss_function.py ===")
    
    # Test with classical coefficients (the result should be exactly 1.0)
    baseline_loss = compute_loss(THETA_CLASSICAL)
    print(f"Loss computed for THETA_CLASSICAL: {baseline_loss:.4f} -> Expected: 1.0000")
    
    # Test with slightly perturbed coefficients
    perturbed_theta = np.array([0.21, 0.39, 0.62, 0.78])
    perturbed_loss = compute_loss(perturbed_theta)
    print(f"Loss computed for perturbed theta: {perturbed_loss:.6e}")
    print("Loss module successfully coupled with solver and benchmarks!")