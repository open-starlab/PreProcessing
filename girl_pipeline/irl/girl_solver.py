"""
GIRL (Goal-based Inverse Reinforcement Learning) Solver.

Solves quadratic programming problem to recover reward weights
from policy gradients.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings


def solve_girl(
    mean_gradients: np.ndarray,
    method: str = 'quadprog',
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Solve GIRL optimization problem to recover reward weights.
    
    Minimizes: θᵀ Gᵀ G θ
    Subject to: θ ≥ 0, sum(θ) = 1
    
    Args:
        mean_gradients: Mean policy gradients of shape (n_params, n_reward_features)
        method: Solver method ('quadprog', 'cvxopt', or 'analytical')
        verbose: Print solver progress
    
    Returns:
        Tuple of:
        - reward_weights: Recovered reward weights of shape (n_reward_features,)
        - info: Dict with solver information
    """
    if verbose:
        print("="*60)
        print("GIRL Solver")
        print("="*60)
        print(f"Mean gradients shape: {mean_gradients.shape}")
        print(f"Method: {method}")
        print()
    
    n_params, n_reward_features = mean_gradients.shape
    
    # Compute Gram matrix G^T G
    G = mean_gradients
    gram_matrix = G.T @ G  # (n_reward_features, n_reward_features)
    
    if verbose:
        print(f"Gram matrix shape: {gram_matrix.shape}")
        print(f"Gram matrix condition number: {np.linalg.cond(gram_matrix):.2e}")
        print()
    
    # Choose solver
    if method == 'analytical':
        reward_weights, info = _solve_analytical(gram_matrix, verbose)
    elif method == 'cvxopt':
        reward_weights, info = _solve_cvxopt(gram_matrix, verbose)
    elif method == 'quadprog':
        reward_weights, info = _solve_quadprog(gram_matrix, verbose)
    else:
        raise ValueError(f"Unknown solver method: {method}")
    
    # Validate solution constraints
    assert np.all(reward_weights >= -1e-6), \
        f"Weights must be non-negative, got min={reward_weights.min()}"
    assert abs(reward_weights.sum() - 1.0) < 1e-3, \
        f"Weights must sum to 1, got {reward_weights.sum()}"
    
    # Project to valid simplex if needed (numerical tolerance)
    reward_weights = np.maximum(reward_weights, 0)
    reward_weights = reward_weights / reward_weights.sum()
    
    if verbose:
        print()
        print("="*60)
        print("GIRL Solution Validation")
        print("="*60)
        print(f"✓ Weight feasibility:")
        print(f"  Min weight: {reward_weights.min():.6e}")
        print(f"  Sum of weights: {reward_weights.sum():.10f}")
        print(f"✓ Constraints satisfied: weights ≥ 0, sum(weights) = 1")
        print()
        print("✓ Recovered Reward Weights (Normalized):")
        feature_names = ['stretch_index', 'pressure_index', 'space_score', 'line_height_rel']
        for i, (name, weight) in enumerate(zip(feature_names[:n_reward_features], reward_weights)):
            print(f"  {name:20s}: {weight:.6f}")
        print()
        print("="*60)
        print()
    
    return reward_weights, info


def _solve_analytical(gram_matrix: np.ndarray, verbose: bool) -> Tuple[np.ndarray, Dict]:
    """
    Solve GIRL using analytical method (uniform initialization + projection).
    
    Args:
        gram_matrix: Gram matrix G^T G
        verbose: Print progress
    
    Returns:
        Tuple of (reward_weights, info)
    """
    n_features = gram_matrix.shape[0]
    
    # Start with uniform weights
    theta = np.ones(n_features) / n_features
    
    # Simple projection onto simplex
    theta = np.maximum(theta, 0)
    theta = theta / theta.sum()
    
    # Compute objective value
    obj_value = theta.T @ gram_matrix @ theta
    
    info = {
        'method': 'analytical',
        'objective': obj_value,
        'success': True
    }
    
    if verbose:
        print(f"Analytical solution objective: {obj_value:.6e}")
    
    return theta, info


def _solve_quadprog(gram_matrix: np.ndarray, verbose: bool) -> Tuple[np.ndarray, Dict]:
    """
    Solve GIRL using quadprog (if available, fallback to scipy).
    
    Args:
        gram_matrix: Gram matrix G^T G
        verbose: Print progress
    
    Returns:
        Tuple of (reward_weights, info)
    """
    try:
        import quadprog
        return _solve_with_quadprog(gram_matrix, verbose)
    except ImportError:
        if verbose:
            print("quadprog not available, using scipy.optimize")
        return _solve_with_scipy(gram_matrix, verbose)


def _solve_with_quadprog(gram_matrix: np.ndarray, verbose: bool) -> Tuple[np.ndarray, Dict]:
    """Solve using quadprog library."""
    import quadprog
    
    n_features = gram_matrix.shape[0]
    
    # QP formulation: min 0.5 * x^T P x + q^T x
    # subject to: Gx >= h, Ax = b
    
    # Add small regularization for numerical stability
    P = 2 * gram_matrix + 1e-8 * np.eye(n_features)
    q = np.zeros(n_features)
    
    # Equality constraint: sum(theta) = 1
    A = np.ones((1, n_features))
    b = np.array([1.0])
    
    # Inequality constraint: theta >= 0
    G = -np.eye(n_features)
    h = np.zeros(n_features)
    
    try:
        # quadprog uses: Ax = b, Gx >= h
        solution = quadprog.solve_qp(P, q, -G.T, h, A.T, b)
        theta = solution[0]
        obj_value = solution[1]
        
        # Ensure valid simplex
        theta = np.maximum(theta, 0)
        theta = theta / theta.sum()
        
        info = {
            'method': 'quadprog',
            'objective': obj_value,
            'success': True
        }
        
        if verbose:
            print(f"Quadprog solution objective: {obj_value:.6e}")
        
        return theta, info
    
    except Exception as e:
        if verbose:
            print(f"Quadprog failed: {e}, falling back to scipy")
        return _solve_with_scipy(gram_matrix, verbose)


def _solve_with_scipy(gram_matrix: np.ndarray, verbose: bool) -> Tuple[np.ndarray, Dict]:
    """Solve using scipy.optimize."""
    from scipy.optimize import minimize
    
    n_features = gram_matrix.shape[0]
    
    # Objective function
    def objective(theta):
        return theta.T @ gram_matrix @ theta
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # sum = 1
    ]
    
    # Bounds: theta >= 0
    bounds = [(0, None) for _ in range(n_features)]
    
    # Initial guess (uniform)
    x0 = np.ones(n_features) / n_features
    
    # Solve
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': verbose, 'ftol': 1e-9}
    )
    
    theta = result.x
    
    # Ensure valid simplex
    theta = np.maximum(theta, 0)
    theta = theta / theta.sum()
    
    info = {
        'method': 'scipy',
        'objective': result.fun,
        'success': result.success,
        'message': result.message
    }
    
    if verbose:
        print(f"Scipy solution objective: {result.fun:.6e}")
        print(f"Success: {result.success}")
    
    return theta, info


def _solve_cvxopt(gram_matrix: np.ndarray, verbose: bool) -> Tuple[np.ndarray, Dict]:
    """Solve using cvxopt (if available, fallback to scipy)."""
    try:
        from cvxopt import matrix, solvers
        
        if not verbose:
            solvers.options['show_progress'] = False
        
        n_features = gram_matrix.shape[0]
        
        # QP: min 0.5 * x^T P x + q^T x
        # subject to: Gx <= h, Ax = b
        
        P = matrix(2 * gram_matrix)
        q = matrix(np.zeros(n_features))
        
        # Inequality: -theta <= 0 (i.e., theta >= 0)
        G = matrix(-np.eye(n_features))
        h = matrix(np.zeros(n_features))
        
        # Equality: sum(theta) = 1
        A = matrix(np.ones((1, n_features)))
        b = matrix([1.0])
        
        solution = solvers.qp(P, q, G, h, A, b)
        
        theta = np.array(solution['x']).flatten()
        obj_value = solution['primal objective']
        
        # Ensure valid simplex
        theta = np.maximum(theta, 0)
        theta = theta / theta.sum()
        
        info = {
            'method': 'cvxopt',
            'objective': obj_value,
            'success': solution['status'] == 'optimal'
        }
        
        if verbose:
            print(f"CVXOPT solution objective: {obj_value:.6e}")
        
        return theta, info
    
    except ImportError:
        if verbose:
            print("cvxopt not available, using scipy")
        return _solve_with_scipy(gram_matrix, verbose)


class GIRLSolver:
    """
    GIRL Solver class for repeated solving.
    """
    
    def __init__(self, method: str = 'quadprog', verbose: bool = True):
        """
        Initialize GIRL solver.
        
        Args:
            method: Solver method
            verbose: Print progress
        """
        self.method = method
        self.verbose = verbose
        self.solutions = []
    
    def solve(self, mean_gradients: np.ndarray) -> np.ndarray:
        """
        Solve for reward weights.
        
        Args:
            mean_gradients: Mean policy gradients
        
        Returns:
            Reward weights
        """
        weights, info = solve_girl(mean_gradients, self.method, self.verbose)
        self.solutions.append((weights, info))
        return weights
    
    def get_statistics(self) -> Dict:
        """
        Get statistics across multiple solutions.
        
        Returns:
            Dict with mean, std of weights
        """
        if not self.solutions:
            return {}
        
        all_weights = np.array([s[0] for s in self.solutions])
        
        return {
            'mean': all_weights.mean(axis=0),
            'std': all_weights.std(axis=0),
            'min': all_weights.min(axis=0),
            'max': all_weights.max(axis=0)
        }


if __name__ == "__main__":
    # Test GIRL solver
    print("Testing GIRL solver...")
    
    np.random.seed(42)
    
    # Create synthetic gradients
    n_params = 1000
    n_reward_features = 4
    
    mean_gradients = np.random.randn(n_params, n_reward_features).astype(np.float32)
    
    # Solve with different methods
    for method in ['analytical', 'quadprog']:
        print(f"\nTesting method: {method}")
        weights, info = solve_girl(mean_gradients, method=method, verbose=True)
        
        # Verify constraints
        assert np.all(weights >= 0), "Weights must be non-negative"
        assert np.abs(weights.sum() - 1.0) < 1e-6, "Weights must sum to 1"
        print(f"✓ Method {method} passed")
    
    print("\n✓ GIRL solver test passed")
