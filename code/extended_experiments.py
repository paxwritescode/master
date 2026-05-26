"""
Extended Research and Analysis Module (extended_analysis.py)

Performs advanced numerical experiments for the Master's thesis:
A) Empirical Order of Convergence (Log-Log Error vs Step Size).
B) Invariant Preservation (Hamiltonian Energy Conservation over time).
C) Optimizer Sensitivity Analysis (Swarm size impact on convergence).
D) Spatial Wave Packet Distribution for the 2D Nonlinear Schrodinger Equation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from rk_solver import rk6_8_integrate
from test_problems import rhs_oscillator, exact_oscillator, rhs_two_body, get_nls_setup
from loss_function import compute_loss, THETA_CLASSICAL

THETA_OPTIMIZED = np.array([0.29052, 0.43018, 0.63375, 0.68519])

os.makedirs("results", exist_ok=True)

# =====================================================================
# EXPERIMENT A: Empirical Order of Convergence (Log-Log Scale)
# =====================================================================
def run_convergence_order_experiment():
    print("\n[Experiment A] Computing empirical order of convergence...")
    h_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    
    errors_classical = []
    errors_optimized = []
    
    u0_osc = np.array([1.0, 0.0])
    t_span = (0.0, 4.0)
    
    for h in h_values:
        # Classical RK6
        t_c, u_c = rk6_8_integrate(rhs_oscillator, u0_osc, t_span, h, THETA_CLASSICAL)
        u_ref_c = exact_oscillator(t_c, u0_osc)
        errors_classical.append(np.max(np.linalg.norm(u_c - u_ref_c, axis=1)))
        
        # Optimized Parametric RK(6,8)
        t_o, u_o = rk6_8_integrate(rhs_oscillator, u0_osc, t_span, h, THETA_OPTIMIZED)
        u_ref_o = exact_oscillator(t_o, u0_osc)
        errors_optimized.append(np.max(np.linalg.norm(u_o - u_ref_o, axis=1)))
        
    slope_classical = np.polyfit(np.log(h_values), np.log(errors_classical), 1)[0]
    slope_optimized = np.polyfit(np.log(h_values), np.log(errors_optimized), 1)[0]
    
    plt.figure(figsize=(7, 6))
    plt.loglog(h_values, errors_classical, 'r--o', lw=2, label=f'Classical RK6 (Slope: {slope_classical:.2f})')
    plt.loglog(h_values, errors_optimized, 'g-s', lw=2, label=f'Optimized Parametric (Slope: {slope_optimized:.2f})')
    
    plt.xlabel('Integration Step Size log(h)', fontsize=12)
    plt.ylabel('Maximum Global Error log(||E||)', fontsize=12)
    plt.title('Empirical Convergence Order Verification', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    plt.savefig("results/exp_A_convergence_order.pdf", bbox_inches='tight')
    plt.close()
    print(" -> Saved: 'results/exp_A_convergence_order.pdf'")


# =====================================================================
# EXPERIMENT B: Energy Conservation (Hamiltonian Invariant Error)
# =====================================================================
def run_energy_conservation_experiment():
    print("\n[Experiment B] Tracking Hamiltonian invariant error over time...")
    u0_2body = np.array([1.0, 0.0, 0.0, 0.6]) # Kepler eccentric orbit
    t_span = (0.0, 30.0) # Long tracking interval
    h = 0.04
    
    t_c, u_c = rk6_8_integrate(rhs_two_body, u0_2body, t_span, h, THETA_CLASSICAL)
    t_o, u_o = rk6_8_integrate(rhs_two_body, u0_2body, t_span, h, THETA_OPTIMIZED)
    
    def compute_hamiltonian_energy(trajectory):
        # E = v^2 / 2 - mu / r
        x, y, vx, vy = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
        r = np.sqrt(x**2 + y**2)
        energy = 0.5 * (vx**2 + vy**2) - 1.0 / r
        return energy

    energy_c = compute_hamiltonian_energy(u_c)
    energy_o = compute_hamiltonian_energy(u_o)
    
    delta_energy_c = np.abs(energy_c - energy_c[0])
    delta_energy_o = np.abs(energy_o - energy_o[0])
    
    plt.figure(figsize=(8, 5))
    plt.plot(t_c, delta_energy_c, 'r--', label='Classical RK6 Energy Drift')
    plt.plot(t_o, delta_energy_o, 'g-', lw=2, label='Optimized Parametric Energy Drift')
    
    plt.yscale('log')
    plt.xlabel('Time Domain (t)', fontsize=12)
    plt.ylabel('Energy Invariant Absolute Deviation |E(t) - E(0)|', fontsize=12)
    plt.title('Hamiltonian Energy Preservation Tracking', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    plt.savefig("results/exp_B_energy_preservation.pdf", bbox_inches='tight')
    plt.close()
    print(" -> Saved: 'results/exp_B_energy_preservation.pdf'")


# =====================================================================
# EXPERIMENT C: Optimizer Sensitivity (Swarm Size Comparison)
# =====================================================================
def run_pso_sensitivity_experiment():
    print("\n[Experiment C] Testing optimizer sensitivity to swarm size parameters...")
    from optimizer import pso_optimization
    
    swarm_sizes = [6, 16, 30]
    max_iterations = 12
    
    plt.figure(figsize=(8, 5))
    
    for size in swarm_sizes:
        print(f" -> Evaluating PSO Swarm Size S = {size}...")
        _, _, history = pso_optimization(compute_loss, num_particles=size, max_iter=max_iterations)
        plt.plot(history, label=f'Swarm Size S = {size}', lw=2)
        
    plt.yscale('log')
    plt.xlabel('Swarm Iterations / Generations', fontsize=12)
    plt.ylabel('Objective Loss Value C(theta)', fontsize=12)
    plt.title('PSO Convergence Sensitivity to Swarm Size', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    plt.savefig("results/exp_C_pso_sensitivity.pdf", bbox_inches='tight')
    plt.close()
    print(" -> Saved: 'results/exp_C_pso_sensitivity.pdf'")


# =====================================================================
# EXPERIMENT D: 2D Spatial Wave Packet Distribution (NLS Plot)
# =====================================================================
def run_nls_spatial_distribution_experiment():
    print("\n[Experiment D] Simulating 2D Nonlinear Schrodinger wave distribution...")
    Nx, Ny = 16, 16  # Enhanced grid density for a beautiful dense surface plot
    rhs_nls, u0_nls = get_nls_setup(Nx=Nx, Ny=Ny, dx=0.4, dy=0.4)
    t_span = (0.0, 0.4)
    h = 0.02
    
    t_steps, u_sol = rk6_8_integrate(rhs_nls, u0_nls, t_span, h, THETA_OPTIMIZED)
    
    u_final = u_sol[-1]
    u_real = u_final[:Nx*Ny].reshape((Nx, Ny))
    u_imag = u_final[Nx*Ny:].reshape((Nx, Ny))
    
    psi_amplitude = u_real**2 + u_imag**2
    
    x = np.linspace(-3.2, 3.2, Nx)
    y = np.linspace(-3.2, 3.2, Ny)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, psi_amplitude, cmap='viridis', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=r'Probability Density $|\psi(x,y)|^2$')
    
    ax.set_xlabel('Spatial Axis X', fontsize=11)
    ax.set_ylabel('Spatial Axis Y', fontsize=11)
    ax.set_zlabel(r'$|\psi|^2$', fontsize=11)
    ax.set_title('NLS 2D Wave Packet Distribution at Final Time Layer', fontsize=13, fontweight='bold')
    
    plt.savefig("results/exp_D_nls_spatial_3d.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved: 'results/exp_D_nls_spatial_3d.png'")


# =====================================================================
# COMPREHENSIVE EXPERIMENTAL RUNNER
# =====================================================================
if __name__ == "__main__":
    start_total_time = time.time()
    print("=" * 70)
    print("    RUNNING GRADUATE ANALYSIS EXTENSION FOR MASTER'S THESIS")
    print("=" * 70)
    
    run_convergence_order_experiment()
    run_energy_conservation_experiment()
    run_pso_sensitivity_experiment()
    run_nls_spatial_distribution_experiment()
    
    print("\n" + "=" * 70)
    print(f" ALL THESIS EXPERIMENTS COMPLETED IN {time.time() - start_total_time:.2f} SECONDS")
    print(" CHECK THE 'results/' FOLDER FOR PUBLICATION-QUALITY GRAPHICS!")
    print("=" * 70)