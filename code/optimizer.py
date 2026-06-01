"""
Optimization Module (optimizer.py)

Implements two distinct global optimization paradigms to minimize the loss functional:
1. Deterministic Multi-Stage Grid Search with monotonicity-based branch pruning.
2. Modified Particle Swarm Optimization (PSO) with local topology adaptation 
   and a boundary constraint projection operator.
"""

import numpy as np
from typing import Callable, Tuple, List

BOUNDS_MIN = np.array([0.01, 0.17, 0.20, 0.52])
BOUNDS_MAX = np.array([0.15, 0.35, 0.495, 0.98]) # Freedom for parameters c5 and c7!

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
    
    def _search_subspace(c2_space, c4_space, c5_space, c7_space, b_theta, b_loss):
        evals, prunes = 0, 0
        for c2 in c2_space:
            for c4 in c4_space:
                if c4 <= c2:
                    prunes += 1
                    continue
                for c5 in c5_space:
                    if c5 <= c4:
                        prunes += 1
                        continue
                    for c7 in c7_space:
                        if c7 <= c5:
                            prunes += 1
                            continue
                        
                        theta = np.array([c2, c4, c5, c7])
                        current_loss = loss_func(theta)
                        evals += 1
                        
                        if current_loss < b_loss:
                            b_loss = current_loss
                            b_theta = theta
        return b_theta, b_loss, evals, prunes
    
    print("\n[Grid Search] Initializing coarse multi-stage deterministic search...")
    best_loss = float('inf')
    best_theta = np.array([0.1, 0.25, 0.4, 0.7])
    
    c2_grid = np.linspace(BOUNDS_MIN[0], BOUNDS_MAX[0], 3)
    c4_grid = np.linspace(BOUNDS_MIN[1], BOUNDS_MAX[1], 3)
    c5_grid = np.linspace(BOUNDS_MIN[2], BOUNDS_MAX[2], 3)
    c7_grid = np.linspace(BOUNDS_MIN[3], BOUNDS_MAX[3], 4)
        
    best_theta, best_loss, eval_count, pruned_count = _search_subspace(
        c2_grid, c4_grid, c5_grid, c7_grid, best_theta, best_loss
    )

    print(f"[Grid Search] Completed coarse stage. Evaluated: {eval_count} safe configurations.")
    print(f"[Grid Search] Coarse Optimum: {best_theta} | Loss: {best_loss:.6e}")
    
    print("[Grid Search] Executing multi-stage adaptive local refinement...")
    refine_step = 0.04
    num_refine_stages = 3
    refine_eval_count = 0
    refine_pruned_count = 0
    
    for stage in range(num_refine_stages):
        r_c2, r_c4, r_c5, r_c7 = best_theta
        
        c2_range = np.unique(np.clip([r_c2 - refine_step, r_c2, r_c2 + refine_step], BOUNDS_MIN[0], BOUNDS_MAX[0]))
        c4_range = np.unique(np.clip([r_c4 - refine_step, r_c4, r_c4 + refine_step], BOUNDS_MIN[1], BOUNDS_MAX[1]))
        c5_range = np.unique(np.clip([r_c5 - refine_step, r_c5, r_c5 + refine_step], BOUNDS_MIN[2], BOUNDS_MAX[2]))
        c7_range = np.unique(np.clip([r_c7 - refine_step, r_c7, r_c7 + refine_step], BOUNDS_MIN[3], BOUNDS_MAX[3]))
            
        best_theta, best_loss, stage_eval, stage_pruned = _search_subspace(
            c2_range, c4_range, c5_range, c7_range, best_theta, best_loss
        )
        
        refine_eval_count += stage_eval
        refine_pruned_count += stage_pruned
        
        print(f"  -> Stage {stage+1}/{num_refine_stages} | Step: {refine_step:.4f} | Current Best Loss: {best_loss:.6e}")
        refine_step /= 2.0
                        
    print(f"[Grid Search] Refinement completed. Extra evaluations: {refine_eval_count}, Extra pruned: {refine_pruned_count}")
    print(f"[Grid Search] Total processed configurations: {eval_count + refine_eval_count} | Total bypassed: {pruned_count + refine_pruned_count}")
    
    return best_theta, best_loss


def pso_optimization(
    loss_func: Callable[[np.ndarray], float], 
    num_particles: int, 
    max_iter: int
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Implements a modified bounded Particle Swarm Optimization (PSO).
    Enforces isolated coordinate bounds via projection and applies velocity clamping.
    """
    print(f"\n[PSO] Spawning population of {num_particles} particles across partitioned 4-D space...")
    
    c1, c2 = 1.49618, 1.49618
    w_max, w_min = 0.9, 0.4 
    
    v_max = 0.2 * (BOUNDS_MAX - BOUNDS_MIN)
    
    X = np.zeros((num_particles, 4))
    for j in range(4):
        X[:, j] = np.random.uniform(BOUNDS_MIN[j], BOUNDS_MAX[j], num_particles)
        
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
            
            V[i] = np.clip(V[i], -v_max, v_max)
            
            X[i] = X[i] + V[i]
            
            X[i] = np.clip(X[i], BOUNDS_MIN, BOUNDS_MAX)
            
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
            
        if stall_counter >= 4:
            V += np.random.uniform(-0.05, 0.05, V.shape) * (BOUNDS_MAX - BOUNDS_MIN)
            stall_counter = 0
            
        history.append(g_best_loss)
        print(f"Iteration {k+1:02d}/{max_iter} | Swarm Champion Global Loss: {g_best_loss:.6e}")
        
    return g_best, g_best_loss, history

def local_polish_optimization(
    loss_func: Callable[[np.ndarray], float], 
    start_theta: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Executes a local Nelder-Mead simplex refinement (polishing) 
    starting from the PSO champion within strict boundary constraints.
    
    Parameters
    ----------
    loss_func : Callable[[np.ndarray], float]
        Pointer to the objective functional C(theta).
    start_theta : np.ndarray
        Starting parameter vector (usually the best found by PSO).
        
    Returns
    -------
    polished_theta : np.ndarray
        Locally optimized parameters at the absolute bottom of the valley.
    polished_loss : float
        The refined minimum loss value.
    """
    print("\n======================================================================")
    print("[HYBRID] Starting high-precision local polishing of the PSO champion...")
    print("======================================================================")
    
    import scipy.optimize as opt

    def bounded_loss_wrapper(theta):
        if np.any(theta < BOUNDS_MIN) or np.any(theta > BOUNDS_MAX):
            return 1e5
        return loss_func(theta)

    res = opt.minimize(
        bounded_loss_wrapper, 
        start_theta, 
        method='Nelder-Mead', 
        options={
            'maxiter': 200,
            'xatol': 1e-7,
            'fatol': 1e-7
        }
    )
    
    if res.success:
        print(f"[HYBRID] Polishing successfully converged in {res.nit} iterations.")
    else:
        print("[HYBRID] Warning: Polishing reached iteration limit or stopped early.")
        
    return res.x, res.fun

if __name__ == "__main__":
    print("=== Testing optimizer.py Module ===")
    try:
        from loss_function import compute_loss
        
        g_theta, g_loss = grid_search_optimization(compute_loss)
        print(f"Grid optimization champion: {g_theta}, Loss: {g_loss:.4f}")
        
        p_theta, p_loss, p_hist = pso_optimization(compute_loss, num_particles=8, max_iter=5)
        print(f"PSO optimization champion: {p_theta}, Loss: {p_loss:.4f}")
        print("Optimizer successfully integrated into the complex pipeline!")
    except ImportError as e:
        print("Import error during standalone test verification:", e)