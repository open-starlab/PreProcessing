"""
GIRL Pipeline - Goal-based Inverse Reinforcement Learning

Main pipeline for recovering reward weights from expert demonstrations.
"""

__version__ = "1.0.0"

from .data_loader import load_sar_dataset, get_action_names
from .bc.model import DefenseBC, create_defense_bc_model
from .bc.train_bc import train_bc_model
from .gradients.compute_gradients import compute_policy_gradients
from .irl.girl_solver import solve_girl, GIRLSolver
from .utils.cross_validation import cross_validate_girl

__all__ = [
    'load_sar_dataset',
    'get_action_names',
    'DefenseBC',
    'create_defense_bc_model',
    'train_bc_model',
    'compute_policy_gradients',
    'solve_girl',
    'GIRLSolver',
    'cross_validate_girl',
]
