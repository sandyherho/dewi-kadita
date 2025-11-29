"""Core solver components for Couzin model."""

from .solver import CouzinSolver
from .systems import CouzinSystem
from .metrics import (
    compute_school_cohesion_entropy,
    compute_polarization_entropy,
    compute_depth_stratification_entropy,
    compute_angular_momentum_entropy,
    compute_nearest_neighbor_entropy,
    compute_velocity_correlation_entropy,
    compute_school_shape_entropy,
    compute_oceanic_schooling_index,
    compute_order_index,
    compute_all_metrics,
    compute_metrics_timeseries
)

__all__ = [
    "CouzinSolver",
    "CouzinSystem",
    "compute_school_cohesion_entropy",
    "compute_polarization_entropy",
    "compute_depth_stratification_entropy",
    "compute_angular_momentum_entropy",
    "compute_nearest_neighbor_entropy",
    "compute_velocity_correlation_entropy",
    "compute_school_shape_entropy",
    "compute_oceanic_schooling_index",
    "compute_order_index",
    "compute_all_metrics",
    "compute_metrics_timeseries"
]
