import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Individual:
    rnvec: np.ndarray
    factorial_costs: float = None
    constraint_violation: float = None


class DaS_KT:
    """Domain-Adaptive Selection for Knowledge Transfer.
    
    Learns which dimensions should be transferred between task pairs
    based on transfer success.
    """
    
    def __init__(self, num_tasks: int, max_dim: int, eta: float = 0.05, warmup: int = 10):
        """
        Args:
            num_tasks: Number of tasks
            max_dim: Maximum dimension across all tasks
            eta: Learning rate for weight updates
            warmup: Number of generations before using learned weights
        """
        self.num_tasks = num_tasks
        self.max_dim = max_dim
        self.eta = eta
        self.warmup = warmup
        
        # Dimension weights for each task pair (src -> dst)
        # Shape: [num_tasks, num_tasks, max_dim]
        self.dim_weights = np.ones((num_tasks, num_tasks, max_dim))
        self.gen_count = 0
        
        # Normalization
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1 for each task pair."""
        for src in range(self.num_tasks):
            for dst in range(self.num_tasks):
                total = self.dim_weights[src, dst].sum()
                if total > 0:
                    self.dim_weights[src, dst] /= total
    
    def select_dimensions(self, src_task: int, dst_task: int, divD: int, rng=None) -> np.ndarray:
        """Select divD dimensions based on learned weights.
        
        Args:
            src_task: Source task index
            dst_task: Destination task index
            divD: Number of dimensions to select
            rng: Random number generator
            
        Returns:
            Array of selected dimension indices
        """
        rng = np.random.default_rng(rng)
        
        if self.gen_count < self.warmup:
            # Warmup: uniform random selection
            return rng.choice(self.max_dim, min(divD, self.max_dim), replace=False)
        
        # Select dimensions with probability proportional to weights
        weights = self.dim_weights[src_task, dst_task]
        probs = weights / weights.sum()
        
        # Sample without replacement
        dims = rng.choice(self.max_dim, min(divD, self.max_dim), replace=False, p=probs)
        return dims
    
    def update_weights(self, src_task: int, dst_task: int, transferred_dims: np.ndarray, reward: float):
        """Update dimension weights based on transfer success.
        
        Args:
            src_task: Source task index
            dst_task: Destination task index
            transferred_dims: Dimensions that were transferred
            reward: Quality of offspring (higher = better)
        """
        if self.gen_count < self.warmup:
            return  # No updates during warmup
        
        for dim in transferred_dims:
            # Policy gradient update
            current_weight = self.dim_weights[src_task, dst_task, dim]
            grad = reward / (current_weight + 1e-10)
            
            # Exponential update
            self.dim_weights[src_task, dst_task, dim] *= np.exp(self.eta * grad)
        
        # Clip to prevent overflow
        self.dim_weights[src_task, dst_task] = np.clip(
            self.dim_weights[src_task, dst_task], 1e-10, 1e10
        )
        
        # Renormalize
        self._normalize_weights()
    
    def next_generation(self):
        """Increment generation counter."""
        self.gen_count += 1
    
    def get_weights(self, src_task: int, dst_task: int) -> np.ndarray:
        """Get current dimension weights for a task pair."""
        return self.dim_weights[src_task, dst_task].copy()
