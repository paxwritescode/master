import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

# Direct imports from your existing project modules
from rk_solver import rk6_8_integrate
from test_problems import rhs_oscillator, exact_oscillator, get_nls_setup, rhs_two_body, rhs_outer_planets

# =========================================================================
# GLOBAL MATPLOTLIB CONFIGURATION FOR ACADEMIC PAPERS
# =========================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


def plot_harmonic_oscillator():
    """
    Simulates and visualizes the Linear Harmonic Oscillator problem,
    comparing different step sizes (large vs good) against the exact solution.
    """
    print("\n--- Generating Harmonic Oscillator Visualization ---")
    
    u0 = np.array([1.0, 0.0])
    t_span = (0.0, 10.0)
    
    h_large = 0.5
    h_good = 0.25
    
    theta_optimized = np.array([0.07819, 0.35000, 0.47245, 0.98000])

    print("Integrating Harmonic Oscillator with a large step (h = 0.50)...")
    t_large, u_large = rk6_8_integrate(rhs_oscillator, u0, t_span, h_large, theta_optimized)
    
    print("Integrating Harmonic Oscillator with a good step (h = 0.25)...")
    t_good, u_good = rk6_8_integrate(rhs_oscillator, u0, t_span, h_good, theta_optimized)

    t_fine = np.linspace(t_span[0], t_span[1], 1000)
    u_exact = exact_oscillator(t_fine, u0)

    plt.figure(figsize=(9, 4.5))

    plt.plot(t_fine, u_exact[:, 0], 
             color='black', linestyle='-', linewidth=2, 
             label='Exact Analytical Solution')

    plt.plot(t_good, u_good[:, 0], 
             color='tab:blue', linestyle='-.', linewidth=1.6, 
             label=f'Optimized RK6 with Good Step ($h = {h_good}$)')

    plt.plot(t_large, u_large[:, 0], 
             color='tab:red', linestyle='--', marker='s', markersize=4, linewidth=1.4, 
             label=f'Optimized RK6 with Large Step ($h = {h_large}$)')

    plt.xlabel('Time $t$', fontsize=11)
    plt.ylabel('Oscillator Position $x(t)$', fontsize=11)
    plt.title('Linear Harmonic Oscillator: Step Size Error Accumulation', fontsize=11, pad=10)
    
    plt.xlim(t_span)
    plt.ylim(-1.3, 1.3)
    plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gainsboro')
    plt.tight_layout()

    output_path = "results/visualization/oscillator_visualized.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"Success! Oscillator plot saved to '{output_path}'")
    plt.close()

def plot_nls_components():
    """
    Simulates NLS 2D and visualizes Real and Imaginary components side-by-side,
    comparing the high-precision reference solution with the optimized RK6 method 
    under different step sizes (large vs good) at a fixed physical time.
    """
    print("\n--- Generating Nonlinear Schrödinger Components Step Comparison ---")
    
    # 1. Setup spatial grid and system parameters
    Nx, Ny = 4, 4
    dx, dy = 0.5, 0.5
    rhs_nls, u0_nls = get_nls_setup(Nx=Nx, Ny=Ny, dx=dx, dy=dy)
    
    t_span = (0.0, 2.0)
    target_t = 1.0  # The exact physical time layer to visualize
    
    # Define two distinct step sizes for step-controlled convergence check
    h_large = 0.2
    h_good = 0.02
    
    theta_optimized = np.array([0.09010, 0.23948, 0.42150, 0.87405])

    # 2. Compute high-precision reference baseline via DOP853
    print("Computing NLS 2D high-precision reference via DOP853...")
    # Use a dense time mesh for DOP853 to hit the target physical time perfectly
    t_eval_ref = np.linspace(t_span[0], t_span[1], 401)
    ref_sol = solve_ivp(rhs_nls, t_span, u0_nls, method='DOP853', t_eval=t_eval_ref, atol=1e-13, rtol=1e-13)
    idx_ref = np.argmin(np.abs(ref_sol.t - target_t))
    u_ref_t = ref_sol.y.T[idx_ref]

    # 3. Compute numerical trajectories with your optimized method using different step sizes
    print(f"Integrating NLS 2D system with a Large Step (h = {h_large})...")
    t_steps_large, u_sol_large = rk6_8_integrate(rhs_nls, u0_nls, t_span, h_large, theta_optimized)
    idx_large = np.argmin(np.abs(t_steps_large - target_t))
    u_large_t = u_sol_large[idx_large]

    print(f"Integrating NLS 2D system with a Good Step (h = {h_good})...")
    t_steps_good, u_sol_good = rk6_8_integrate(rhs_nls, u0_nls, t_span, h_good, theta_optimized)
    idx_good = np.argmin(np.abs(t_steps_good - target_t))
    u_good_t = u_sol_good[idx_good]

    # 4. Reconstruct 2D structures from the flattened state vectors
    real_ref_2d = u_ref_t[:Nx*Ny].reshape((Nx, Ny))
    imag_ref_2d = u_ref_t[Nx*Ny:].reshape((Nx, Ny))
    
    real_large_2d = u_large_t[:Nx*Ny].reshape((Nx, Ny))
    imag_large_2d = u_large_t[Nx*Ny:].reshape((Nx, Ny))
    
    real_good_2d = u_good_t[:Nx*Ny].reshape((Nx, Ny))
    imag_good_2d = u_good_t[Nx*Ny:].reshape((Nx, Ny))

    # Define coarse spatial coordinates for the 4 nodes
    x_coarse = np.array([i * dx - (Nx * dx) / 2.0 for i in range(Nx)])
    y_slice_idx = Ny // 2
    
    # Extract a 1D slice across the central Y-axis row
    r_ref_slice = real_ref_2d[:, y_slice_idx]
    i_ref_slice = imag_ref_2d[:, y_slice_idx]
    
    r_large_slice = real_large_2d[:, y_slice_idx]
    i_large_slice = imag_large_2d[:, y_slice_idx]
    
    r_good_slice = real_good_2d[:, y_slice_idx]
    i_good_slice = imag_good_2d[:, y_slice_idx]

    # 5. Use Cubic Spline interpolation to generate pristine, smooth wave representations
    x_fine = np.linspace(x_coarse.min(), x_coarse.max(), 300)
    
    cs_r_ref = CubicSpline(x_coarse, r_ref_slice)
    cs_i_ref = CubicSpline(x_coarse, i_ref_slice)
    
    cs_r_large = CubicSpline(x_coarse, r_large_slice)
    cs_i_large = CubicSpline(x_coarse, i_large_slice)
    
    cs_r_good = CubicSpline(x_coarse, r_good_slice)
    cs_i_good = CubicSpline(x_coarse, i_good_slice)

    # =========================================================================
    # HORIZONTAL SUBPLOTS CONSTRUCTION (Side-by-Side Layout)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # -------------------------------------------------------------------------
    # LEFT SUBPLOT: Real Part Step Size Comparison
    # -------------------------------------------------------------------------
    ax1.plot(x_fine, cs_r_ref(x_fine), color='black', linestyle='-', linewidth=2, 
             label='Reference Profile (DOP853)')
    ax1.plot(x_fine, cs_r_good(x_fine), color='tab:blue', linestyle='-.', linewidth=1.6, 
             label=f'Optimized RK6, $h = {h_good}$')
    ax1.plot(x_fine, cs_r_large(x_fine), color='tab:red', linestyle='--', linewidth=1.4, 
             label=f'Optimized RK6, $h = {h_large}$')
    ax1.scatter(x_coarse, r_large_slice, color='darkred', marker='o', s=35, zorder=5, 
                label='Solver Grid Nodes')
    
    ax1.set_xlabel('Spatial Coordinate $x$', fontsize=11)
    ax1.set_ylabel('Real Part $\\mathfrak{R}(u)$', fontsize=11)
    ax1.set_title('Real Part $\\mathfrak{R}(u)$ of the Dark Soliton', fontsize=12, pad=10)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # RIGHT SUBPLOT: Imaginary Part Step Size Comparison
    # -------------------------------------------------------------------------
    ax2.plot(x_fine, cs_i_ref(x_fine), color='black', linestyle='-', linewidth=2, 
             label='Reference Profile (DOP853)')
    ax2.plot(x_fine, cs_i_good(x_fine), color='tab:blue', linestyle='-.', linewidth=1.6, 
             label=f'Optimized RK6 (Good Step, $h = {h_good}$)')
    ax2.plot(x_fine, cs_i_large(x_fine), color='tab:red', linestyle='--', linewidth=1.4, 
             label=f'Optimized RK6 (Large Step, $h = {h_large}$)')
    ax2.scatter(x_coarse, i_large_slice, color='darkred', marker='x', s=35, zorder=5, 
                label='Actual Solver Grid Nodes')
    
    ax2.set_xlabel('Spatial Coordinate $x$', fontsize=11)
    ax2.set_ylabel('Imaginary Part $\\mathfrak{I}(u)$', fontsize=11)
    ax2.set_title('Imaginary Part $\\mathfrak{I}(u)$ of the Dark Soliton', fontsize=12, pad=10)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Global superior title configuration showing current physical time
    plt.suptitle(f'Oscillatory Behavior of the Dark Soliton (Real & Imaginary Components), t = {target_t}', 
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save high-quality vector graphic for LaTeX
    output_path = "results/visualization/nls_components_visualized.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"Success! Side-by-side NLS component step comparison saved to '{output_path}'")
    plt.close()
    
def plot_two_body_orbit():
    """
    Simulates and visualizes the Gravitational Two-Body (Kepler) problem.
    Plots the spatial orbit (Y vs X) on the left, and the explicit temporal 
    coordinate oscillations (X vs t) on the right to demonstrate phase-lag tracking.
    """
    print("\n--- Generating Gravitational Two-Body Orbit & Oscillations ---")
    
    # 1. Setup system parameters (State vector: [x, y, vx, vy])
    u0 = np.array([1.0, 0.0, 0.0, 0.9])
    t_span = (0.0, 15.0)
    
    h_large = 0.4
    h_good = 0.04
    theta_optimized = np.array([0.09010, 0.23948, 0.42150, 0.87405])

    # 2. Compute numerical integrations via your optimized RK6 solver
    print("Integrating Kepler system with a large step (h = 0.25)...")
    t_large, u_large = rk6_8_integrate(rhs_two_body, u0, t_span, h_large, theta_optimized)
    
    print("Integrating Kepler system with a good step (h = 0.05)...")
    t_good, u_good = rk6_8_integrate(rhs_two_body, u0, t_span, h_good, theta_optimized)

    # 3. Compute reference baseline via high-precision DOP853
    print("Computing high-precision Kepler reference via DOP853...")
    t_fine = np.linspace(t_span[0], t_span[1], 1000)
    ref_sol = solve_ivp(rhs_two_body, t_span, u0, method='DOP853', t_eval=t_fine, atol=1e-13, rtol=1e-13)
    u_exact = ref_sol.y.T

    # =========================================================================
    # SIDE-BY-SIDE PLOTS: ORBIT MAP (LEFT) AND COORDINATE OSCILLATIONS (RIGHT)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # -------------------------------------------------------------------------
    # LEFT SUBPLOT: Spatial Trajectory (The Orbit)
    # -------------------------------------------------------------------------
    ax1.plot(u_exact[:, 0], u_exact[:, 1], color='black', linestyle='-', linewidth=2, 
             label='Exact Elliptical Orbit (DOP853)')
    ax1.plot(u_good[:, 0], u_good[:, 1], color='tab:blue', linestyle='-.', linewidth=1.6, 
             label=f'Optimized RK6 ($h = {h_good}$)')
    ax1.plot(u_large[:, 0], u_large[:, 1], color='tab:red', linestyle='--', linewidth=1.4, 
             label=f'Optimized RK6 ($h = {h_large}$)')
    ax1.scatter([0], [0], color='orange', s=100, zorder=5, label='Central Mass')
    
    ax1.set_xlabel('x$', fontsize=11)
    ax1.set_ylabel('y$', fontsize=11)
    ax1.set_title('Space Domain: Trajectory Orbit Stability', fontsize=12, pad=10)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal') # Keep the circle perfectly round
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # RIGHT SUBPLOT: Temporal Oscillations (The Coordinate Waves)
    # -------------------------------------------------------------------------
    ax2.plot(t_fine, u_exact[:, 0], color='black', linestyle='-', linewidth=2, 
             label='Exact Solution')
    ax2.plot(t_good, u_good[:, 0], color='tab:blue', linestyle='-.', linewidth=1.6, 
             label=f'Optimized RK6 ($h = {h_good}$)')
    ax2.plot(t_large, u_large[:, 0], color='tab:red', linestyle='--', marker='s', markevery=3, markersize=4, linewidth=1.4, 
             label=f'Optimized RK6 ($h = {h_large}$)')
    
    ax2.set_xlabel('$Time $t$', fontsize=11)
    ax2.set_ylabel('$x(t)$', fontsize=11)
    ax2.set_title('Time Domain: Coordinate $x(t)$ Oscillations', fontsize=12, pad=10)
    ax2.set_xlim(t_span)
    ax2.set_ylim(-1.4, 1.4)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Dynamics of the Gravitational Two-Body Kepler Problem', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save high-quality vector graphic for LaTeX
    output_path = "results/visualization/twobody_orbit_comparison.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"Success! Twin Kepler plot saved to '{output_path}'")
    plt.close()

def plot_outer_planets_orbits():
    """
    Simulates and visualizes the 24-D N-Body Outer Planets problem.
    Plots the spatial concentric orbits (Y vs X) on the left, and the temporal
    coordinate oscillations x(t) for selected planets on the right to match the 
    thesis focus on oscillatory solutions.
    """
    print("\n--- Generating Outer Planets N-Body Visualization ---")
    
    # 1. Setup initial conditions (6 bodies x 4 variables = 24 dimensions)
    u0_planets = np.zeros(24)
    u0_planets[0::4] = np.array([0.0, 5.2, 9.5, 19.2, 30.1, 39.5])      # Initial X positions (AU)
    u0_planets[3::4] = np.array([0.0, 0.438, 0.324, 0.228, 0.182, 0.159]) # Initial Vy velocities
    
    # We set t_span to 100.0 to observe multiple complete wave cycles for inner planets
    t_span = (0.0, 175.0) 
    h = 0.1
    theta_optimized = np.array([0.09010, 0.23948, 0.42150, 0.87405])

    # 2. Integrate using the optimized RK6 solver
    print("Integrating 24-D Outer Planets system...")
    t_steps, u_sol = rk6_8_integrate(rhs_outer_planets, u0_planets, t_span, h, theta_optimized)

    # 3. Create Side-by-Side Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6))

    # Metadata for plotting the solar system bodies
    body_names = ['Sun', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    body_colors = ['gold', 'chocolate', 'goldenrod', 'lightseagreen', 'royalblue', 'purple']
    body_sizes = [100, 50, 45, 35, 35, 15] # Visual hierarchy

    # -------------------------------------------------------------------------
    # LEFT SUBPLOT: Space Domain (Concentric Orbits Map)
    # -------------------------------------------------------------------------
    for i in range(6):
        idx = i * 4
        x_coords = u_sol[:, idx]
        y_coords = u_sol[:, idx + 1]
        
        # Plot orbital trajectory paths
        ax1.plot(x_coords, y_coords, color=body_colors[i], linestyle='-', linewidth=1.3, alpha=0.8)
        # Mark current planet positions at the final time layer T = 100.0
        ax1.scatter(x_coords[-1], y_coords[-1], color=body_colors[i], s=body_sizes[i], zorder=4, label=body_names[i])

    ax1.set_xlabel('$X$ (AU)', fontsize=11)
    ax1.set_ylabel('$Y$ (AU)', fontsize=11)
    ax1.set_title('Space Domain: Concentric Planetary Orbits', fontsize=12, pad=10)
    # ax1.set_xlim(-45, 45)
    # ax1.set_ylim(-45, 45)
    ax1.set_aspect('equal') # Crucial to ensure orbits remain circular/elliptical without stretching
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=9)

    # -------------------------------------------------------------------------
    # RIGHT SUBPLOT: Time Domain (Planetary Position Oscillations)
    # -------------------------------------------------------------------------
    # Extracting and plotting Jupiter (i=1, index 4) and Saturn (i=2, index 8) 
    # to show distinct frequencies and clean sinusoidal oscillations.
    ax2.plot(t_steps, u_sol[:, 4], color='chocolate', linestyle='-', linewidth=1.6, label='Jupiter $x_1(t)$')
    ax2.plot(t_steps, u_sol[:, 8], color='goldenrod', linestyle='--', linewidth=1.6, label='Saturn $x_2(t)$')

    ax2.set_xlabel('$t$', fontsize=11)
    ax2.set_ylabel('x(t)$ (AU)', fontsize=11)
    ax2.set_title('Time Domain: Selected Coordinate Oscillations', fontsize=12, pad=10)
    # ax2.set_xlim(t_span)
    # ax2.set_ylim(-12, 12) # Fits Jupiter (~5.2 AU) and Saturn (~9.5 AU) amplitudes perfectly
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=10)

    # Main combined figure title
    plt.suptitle('N-Body Outer Planets Simulation', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save high-quality vector graphic for LaTeX
    output_path = "results/visualization/nbody_planets_orbits.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"Success! Side-by-side N-body planetary plot saved to '{output_path}'")
    plt.close()

def main():
    """
    Main orchestration function ensuring directories are built and pipelines are triggered.
    """
    os.makedirs("results", exist_ok=True)
    
    print("=========================================================================")
    plot_harmonic_oscillator()
    plot_nls_components()
    plot_two_body_orbit()
    plot_outer_planets_orbits()
    
    print("=========================================================================")
    print("All tasks completed successfully! Check the 'results/' folder for PDFs.")


if __name__ == "__main__":
    main()