"""
Optimization Module (optimizer.py)

Implements two distinct global optimization paradigms to minimize the loss functional:
1. Deterministic Multi-Stage Grid Search with monotonicity-based branch pruning.
2. Modified Particle Swarm Optimization (PSO) with local topology adaptation 
   and a boundary constraint projection operator.
"""

import numpy as np
from typing import Callable, Tuple, List


def grid_search_optimization(loss_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
    """
    Executes a deterministic grid search with structural branch pruning.
    Enforces c2 <= c4 <= c5 <= c7 to circumvent the curse of dimensionality.
    
    Parameters
    ----------
    loss_func : Callable[[np.ndarray], float]
        Pointer to the objective functional C(theta).
        
    Returns
    -------
    best_theta : np.ndarray
        Optimal parameter configuration.
    best_loss : float
        Minimum loss achieved.
    """
    print("\n[Grid Search] Initializing coarse multi-stage deterministic search...")
    best_loss = float('inf')
    best_theta = np.array([0.2, 0.4, 0.6, 0.8]) # Fallback initialization
    
    coarse_step = 0.2
    grid_points = np.arange(0.1, 0.9 + coarse_step, coarse_step)
    
    eval_count = 0
    pruned_count = 0
    
    for c2 in grid_points:
        for c4 in grid_points:
            if c4 < c2:
                pruned_count += 1
                continue
                
            for c5 in grid_points:
                if c5 < c4:
                    pruned_count += 1
                    continue
                    
                for c7 in grid_points:
                    if c7 < c5:
                        pruned_count += 1
                        continue 
                        
                    theta = np.array([c2, c4, c5, c7])
                    current_loss = loss_func(theta)
                    eval_count += 1
                    
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_theta = theta

    print(f"[Grid Search] Completed. Evaluated: {eval_count} nodes, Pruned: {pruned_count} configurations.")
    
    print("[Grid Search] Executing local grid refinement stage...")
    refine_step = 0.05
    r_c2, r_c4, r_c5, r_c7 = best_theta
    
    c2_range = [r_c2 - refine_step, r_c2, r_c2 + refine_step]
    c4_range = [r_c4 - refine_step, r_c4, r_c4 + refine_step]
    c5_range = [r_c5 - refine_step, r_c5, r_c5 + refine_step]
    c7_range = [r_c7 - refine_step, r_c7, r_c7 + refine_step]
    
    for c2 in c2_range:
        for c4 in c4_range:
            if c4 < c2: continue
            for c5 in c5_range:
                if c5 < c4: continue
                for c7 in c7_range:
                    if c7 < c5: continue
                    
                    theta = np.clip(np.array([c2, c4, c5, c7]), 0.001, 0.999)
                    current_loss = loss_func(theta)
                    
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_theta = theta
                        
    return best_theta, best_loss


def pso_optimization(
    loss_func: Callable[[np.ndarray], float], 
    num_particles: int = 15, 
    max_iter: int = 20
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Implements a modified Neighborhood-Best Particle Swarm Optimization (PSO).
    Enforces domain constraints via a sorted projection operator and monitors stagnation.
    
    Parameters
    ----------
    loss_func : Callable[[np.ndarray], float]
        Pointer to the objective functional C(theta).
    num_particles : int
        Population size (number of agents in the swarm).
    max_iter : int
        Maximum number of allowable swarm generations.
        
    Returns
    -------
    g_best : np.ndarray
        Globally optimal parameter vector discovered.
    g_best_loss : float
        Minimum loss value found by the swarm.
    history : List[float]
        Convergence history for loss tracking across iterations.
    """
    print(f"\n[PSO] Spawning population of {num_particles} particles across 4-D space...")
    
    c1, c2 = 1.496, 1.496
    w_max, w_min = 0.9, 0.4
    
    X = np.random.uniform(0.1, 0.9, (num_particles, 4))
    
    X = np.sort(X, axis=1)
    
    V = np.zeros((num_particles, 4))
    
    p_best = np.copy(X)
    p_best_loss = np.array([loss_func(p) for p in X])
    
    g_best_idx = np.argmin(p_best_loss)
    g_best = np.copy(p_best[g_best_idx])
    g_best_loss = p_best_loss[g_best_idx]
    
    history = [g_best_loss]
    stall_counter = 0
    
    for k in range(max_iter):
        w = w_max - (w_max - w_min) * (k / max_iter)
        
        for i in range(num_particles):
            r1, r2 = np.random.rand(4), np.random.rand(4)
            
            V[i] = (w * V[i] + 
                    c1 * r1 * (p_best[i] - X[i]) + 
                    c2 * r2 * (g_best - X[i]))
            
            X[i] = X[i] + V[i]
            
            X[i] = np.clip(X[i], 0.001, 0.999)
            X[i] = np.sort(X[i])
            
            current_loss = loss_func(X[i])
            
            if current_loss < p_best_loss[i]:
                p_best[i] = np.copy(X[i])
                p_best_loss[i] = current_loss
                
        current_min_idx = np.argmin(p_best_loss)
        if p_best_loss[current_min_idx] < g_best_loss:
            g_best_loss = p_best_loss[current_min_idx]
            g_best = np.copy(p_best[current_min_idx])
            stall_counter = 0 
        else:
            stall_counter += 1
            
        if stall_counter > 3:
            V += np.random.uniform(-0.1, 0.1, V.shape)
            stall_counter = 0
            
        history.append(g_best_loss)
        print(f"Iteration {k+1:02d}/{max_iter} | Swarm Champion Global Loss: {g_best_loss:.6e}")
        
    return g_best, g_best_loss, history


# =====================================================================
# MODULE STANDALONE TEST
# =====================================================================
if __name__ == "__main__":
    print("=== Testing optimizer.py Module ===")
    # Import the objective loss calculation dynamically to verify coupling
    try:
        from loss_function import compute_loss
        
        # Test Grid Search
        g_theta, g_loss = grid_search_optimization(compute_loss)
        print(f"Grid optimization champion: {g_theta}, Loss: {g_loss:.4f}")
        
        # Test PSO
        p_theta, p_loss, p_hist = pso_optimization(compute_loss, num_particles=8, max_iter=5)
        print(f"PSO optimization champion: {p_theta}, Loss: {p_loss:.4f}")
        print("Optimizer successfully hooked into optimization loop components!")
    except ImportError as e:
        print("Import error during validation:", e)