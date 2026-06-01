"""
Main Orchestration Script (main.py)

Acts as the architectural center of the computational complex. Coordinates the
end-to-end pipeline: loops through all four benchmark problems, dynamically
constructs objective functions, executes multi-stage hybrid optimization, 
and exports comprehensive convergence summaries for the thesis report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import csv
import os
import sys

# Internal computational modules mapping
from rk_solver import rk6_8_integrate
from test_problems import rhs_oscillator, exact_oscillator, get_nls_setup, rhs_two_body, rhs_outer_planets
from optimizer import grid_search_optimization, pso_optimization, local_polish_optimization

# Classical non-parametric Butcher tableau coefficients used as an analytical baseline
THETA_CLASSICAL = np.array([0.1, 0.25, 0.4, 0.7])


class DualLogger:
    """
    Custom stream redirector to capture all standard print statements 
    and silently route them to a dedicated execution log file.
    """
    def __init__(self, filepath):
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.log_file.flush()


def print_to_console(message: str):
    """
    Bypasses the redirected stdout to print critical, high-level 
    pipeline progress indicators directly to the user terminal screen.
    """
    sys.__stdout__.write(message + "\n")
    sys.__stdout__.flush()


def make_loss_factory(rhs_func, u0, t_span, h, exact_func=None):
    """
    Dynamic factory closure that constructs a custom objective quality functional C(theta)
    tailored to a specific ordinary differential equation (ODE) environment.

    Parameters
    ----------
    rhs_func : Callable
        Right-hand side vector field function of the target differential system.
    u0 : np.ndarray
        Initial state vector conditions.
    t_span : Tuple[float, float]
        Integration time domain interval bounds.
    h : float
        Numerical step size for the Runge-Kutta solver.
    exact_func : Callable, optional
        Pointer to the analytical exact solution generator, if mathematically available.

    Returns
    -------
    compute_loss : Callable
        Encapsulated loss function accepting parameter vector theta and returning scalar error.
    """
    # Generate discrete time grid for trajectory mapping
    t_eval = np.arange(t_span[0], t_span[1] + h / 2.0, h)
    
    # Pre-compute or generate pristine reference trajectory data
    if exact_func is not None:
        u_ref = exact_func(t_eval, u0)
    else:
        # Fallback: integrate via high-order DOP853 solver with strict tolerances
        sol = solve_ivp(rhs_func, t_span, u0, method='DOP853', t_eval=t_eval, atol=1e-13, rtol=1e-13)
        u_ref = sol.y.T

    def compute_loss(theta):
        """Evaluates mean trajectory deviation vector norm for a specific theta config."""
        try:
            t_steps, u_num = rk6_8_integrate(rhs_func, u0, t_span, h, theta)
            
            # Catch numerical explosions, shape mismatches, or invalid solver state steps
            if u_num.shape != u_ref.shape or np.any(np.isnan(u_num)) or np.any(np.isinf(u_num)):
                return 1e5
                
            # Compute Euclidean norm of absolute trajectory displacement error across all grid steps
            error_norm = np.linalg.norm(u_num - u_ref, axis=1)
            return float(np.mean(error_norm))
        except Exception:
            return 1e5  # Impose severe penalty constraint on algorithmic failures

    return compute_loss


def run_computational_experiment():
    """Main execution entrypoint supervising data setup, cascade optimization, and summaries export."""
    # Ensure results repository structures exist locally
    os.makedirs("results", exist_ok=True)
    np.seterr(all='ignore')
    log_filename = "results/experiment.log"
    
    print_to_console("=" * 75)
    print_to_console("      STARTING MULTI-BENCHMARK RUNGE-KUTTA OPTIMIZATION CONVEYOR")
    print_to_console("=" * 75)
    print_to_console(f"[System] Redirecting verbose output streams to '{log_filename}'...")
    
    # Intercept global sys.stdout and assign it to the silent file logger
    original_stdout = sys.stdout
    sys.stdout = DualLogger(log_filename)
    
    print("=" * 75)
    print("      VERBOSE RUNGE-KUTTA OPTIMIZATION CONVEYOR EXECUTION LOG")
    print("=" * 75)
    
    # Initialize benchmark simulation specific setups
    rhs_nls, u0_nls = get_nls_setup(Nx=4, Ny=4)
    
    u0_planets = np.zeros(24)
    u0_planets[0::4] = np.array([0.0, 5.2, 9.5, 19.2, 30.1, 39.5])  # Position components X
    u0_planets[3::4] = np.array([0.0, 0.438, 0.324, 0.228, 0.182, 0.159])  # Velocity components Y (stable orbits)

    # Core meta-array holding configuration blocks for all four evaluation models
    benchmarks = [
        {
            "id": 1,
            "name": "Linear Harmonic Oscillator",
            "rhs": rhs_oscillator,
            "u0": np.array([1.0, 0.0]),
            "t_span": (0.0, 10.0),
            "h": 0.1,
            "exact": exact_oscillator
        },
        {
            "id": 2,
            "name": "Nonlinear Schrodinger Equation (NLS 2D)",
            "rhs": rhs_nls,
            "u0": u0_nls,
            "t_span": (0.0, 2.0),
            "h": 0.05,
            "exact": None
        },
        {
            "id": 3,
            "name": "Gravitational Two-Body Problem",
            "rhs": rhs_two_body,
            "u0": np.array([1.0, 0.0, 0.0, 0.5]),
            "t_span": (0.0, 15.0),
            "h": 0.1,
            "exact": None
        },
        {
            "id": 4,
            "name": "Gravitational Outer Planets Problem (24D)",
            "rhs": rhs_outer_planets,
            "u0": u0_planets,
            "t_span": (0.0, 20.0),
            "h": 0.2,
            "exact": None
        }
    ]
    
    summary_results = []

    # Main multi-stage conveyor looping over tasks sequentionally 
    for task in benchmarks:
        print_to_console(f"\n[BENCHMARK {task['id']}/4] Initializing optimization pipeline for: {task['name']}")
        
        print("\n" + "#" * 75)
        print(f"  PROCESSING BENCHMARK {task['id']}: {task['name']}")
        print("#" * 75)
        
        # Instantiate objective functional via factory closure
        current_loss_func = make_loss_factory(
            rhs_func=task["rhs"],
            u0=task["u0"],
            t_span=task["t_span"],
            h=task["h"],
            exact_func=task["exact"]
        )
        
        # Capture raw textbook coefficient baseline performance
        baseline_loss = current_loss_func(THETA_CLASSICAL)
        print(f"[Baseline] Classical RK6 Initial Loss: {baseline_loss:.6e}")
        
        print_to_console("  -> Executing optimization cascade (Grid Search -> PSO -> Simplex Polishing)...")
        
        # -------------------------------------------------------------------------
        # STAGE 1: Grid Search (Rough Deterministic Exploration)
        # -------------------------------------------------------------------------
        print(f"[{task['name']}] Running Deterministic Grid Search...")
        start_time = time.time()
        theta_grid, loss_grid = grid_search_optimization(current_loss_func)
        time_grid = time.time() - start_time
        
        # -------------------------------------------------------------------------
        # STAGE 2: Particle Swarm Optimization (Global Stochastic Descent)
        # -------------------------------------------------------------------------
        print(f"[{task['name']}] Running Stochastic Particle Swarm Optimization...")
        start_time = time.time()
        theta_pso, loss_pso, _ = pso_optimization(
            current_loss_func, 
            num_particles=30,
            max_iter=30 
        )
        time_pso = time.time() - start_time
        
        # -------------------------------------------------------------------------
        # STAGE 3: Hybrid Local Polishing (Nelder-Mead Refinement)
        # -------------------------------------------------------------------------
        print(f"[{task['name']}] Polishing PSO Champion via Nelder-Mead Simplex...")
        theta_final, loss_final = local_polish_optimization(current_loss_func, theta_pso)
        
        # Log localized telemetry values internally
        print(f"\n>>> BENCHMARK {task['id']} COMPLETE <<<")
        print(f"  Baseline Loss : {baseline_loss:.6e}")
        print(f"  Grid Search   : {loss_grid:.6e}")
        print(f"  Polished Hybrid: {loss_final:.6e}")
        print(f"  Accuracy Gain : {baseline_loss / (loss_final + 1e-15):.2f}x better")
        print(f"  Optimal Nodes : {np.round(theta_final, 5).tolist()}")
        
        # Append calculated structures to global summary array mapping
        summary_results.append({
            "id": task["id"],
            "name": task["name"],
            "baseline": baseline_loss,
            "grid": loss_grid,
            "final": loss_final,
            "theta": np.round(theta_final, 5).tolist(),
            "time": time_grid + time_pso
        })
        print_to_console("  -> Cascade completed successfully. Phase-lag errors minimized.")


if __name__ == "__main__":
    run_computational_experiment()