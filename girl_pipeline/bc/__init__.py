"""Behavior cloning module for GIRL pipeline."""

from .model import DefenseBC
from .train_bc import train_bc_model

__all__ = ['DefenseBC', 'train_bc_model']
