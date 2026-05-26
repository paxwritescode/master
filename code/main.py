"""
Main Orchestration Script (main.py)

Acts as the architectural center of the computational complex. Coordinates the
end-to-end pipeline: initializes hyper-parameters, runs both deterministic and 
stochastic optimization processes, executes post-processing validation, and 
generates final high-resolution plots for the thesis report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import csv
import os

from rk_solver import rk6_8_integrate, reconstruct_butcher
from test_problems import rhs_two_body
from loss_function import compute_loss, THETA_CLASSICAL
from optimizer import grid_search_optimization, pso_optimization


def run_computational_experiment():
    print("=" * 70)
    print("        STARTING COMPLEX RUNGE-KUTTA OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Deterministic Multi-Stage Grid Search Execution
    # -------------------------------------------------------------------------
    start_grid = time.time()
    theta_grid, loss_grid = grid_search_optimization(compute_loss)
    time_grid = time.time() - start_grid
    
    print("\n" + "-" * 50)
    print(f"[RESULTS] Grid Search Optimal Parameters Discovered:")
    print(f"theta* = {np.round(theta_grid, 5).tolist()}")
    print(f"Minimum Quality Functional C(theta*): {loss_grid:.6e}")
    print(f"Execution Time: {time_grid:.2f} seconds")
    print("-" * 50)
    
    # -------------------------------------------------------------------------
    # STAGE 2: Stochastic Particle Swarm Optimization (PSO) Execution
    # -------------------------------------------------------------------------
    start_pso = time.time()
    theta_pso, loss_pso, pso_history = pso_optimization(
        compute_loss, 
        num_particles=20,
        max_iter=30
    )
    time_pso = time.time() - start_pso
    
    print("\n" + "-" * 50)
    print(f"[RESULTS] Particle Swarm Optimization (PSO) Optimal Parameters:")
    print(f"theta* = {np.round(theta_pso, 5).tolist()}")
    print(f"Minimum Quality Functional C(theta*): {loss_pso:.6e}")
    print(f"Execution Time: {time_pso:.2f} seconds")
    print("-" * 50)
    
    # -------------------------------------------------------------------------
    # STAGE 3: Data Export / Post-Processing (.csv generation)
    # -------------------------------------------------------------------------
    csv_filename = "results/optimization_summary.csv"
    print(f"\n[Post-Processing] Exporting convergence summary to '{csv_filename}'...")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "PSO_Global_Best_Loss", "Grid_Search_Baseline"])
        for idx, pso_loss in enumerate(pso_history):
            writer.writerow([idx, pso_loss, loss_grid])
            
    # -------------------------------------------------------------------------
    # STAGE 4: Extended Validation and Plot Generation
    # -------------------------------------------------------------------------
    print("[Post-Processing] Initiating visual post-processing suite...")
    
    plt.figure(figsize=(14, 6))
    
    # --- PLOT 1: Optimization Swarm Convergence Dynamics ---
    plt.subplot(1, 2, 1)
    plt.plot(pso_history, 'b-o', lw=2.5, label='Modified PSO Swarm Convergence')
    plt.axhline(y=1.0, color='black', linestyle=':', label='Classical RK6 Baseline (Loss = 1.0)')
    plt.axhline(y=loss_grid, color='r', linestyle='--', lw=2, label='Deterministic Grid Search Limit')
    plt.yscale('log')
    plt.xlabel('Swarm Iterations / Generations', fontsize=12)
    plt.ylabel('Generalized Loss Functional C(theta) [Log Scale]', fontsize=12)
    plt.title('Global Optimization Convergence History', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    
    # --- PLOT 2: Long-Term Meta-Validation Trajectory (Two-Body Problem) ---
    u0_validate = np.array([1.0, 0.0, 0.0, 0.5])
    t_span_validate = (0.0, 15.0)
    h_validate = 0.05
    
    t_class, u_class = rk6_8_integrate(rhs_two_body, u0_validate, t_span_validate, h_validate, THETA_CLASSICAL)
    t_opt, u_opt = rk6_8_integrate(rhs_two_body, u0_validate, t_span_validate, h_validate, theta_pso)
    
    sol_ref = solve_ivp(rhs_two_body, t_span_validate, u0_validate, method='DOP853', t_eval=t_opt, atol=1e-13, rtol=1e-13)
    u_ref = sol_ref.y.T
    
    error_classical_series = np.linalg.norm(u_class - u_ref, axis=1)
    error_optimized_series = np.linalg.norm(u_opt - u_ref, axis=1)
    
    plt.subplot(1, 2, 2)
    plt.plot(t_opt, error_classical_series, 'r--', lw=1.8, label='Classical Non-Parametric RK6')
    plt.plot(t_opt, error_optimized_series, 'g-', lw=2.5, label='Optimized Parametric RK(6,8) [PSO]')
    plt.yscale('log')
    plt.xlabel('Temporal Domain Integration Interval (t)', fontsize=12)
    plt.ylabel('Absolute Trajectory Deviation Error ||E(t)||', fontsize=12)
    plt.title('Long-Term Validation on Extended Interval', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    plot_filename = "results/rk_optimization_results.pdf"
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    print(f"[Post-Processing] Publication-quality chart saved as '{plot_filename}'.")
    
    plt.show()
    print("=" * 70)
    print("        COMPUTATIONAL EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_computational_experiment()