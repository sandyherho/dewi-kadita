"""dewi-kadita: Idealized 3D Couzin Model Fish Schooling Simulator with Oceanic Entropy Metrics"""

__version__ = "0.0.4"
__author__ = "Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Alfita P. Handayani, Karina Sujatmiko, Kamaluddin Kasim, Rusmawan Suwarman, Dasapta E. Irawan"
__email__ = "sandy.herho@email.ucr.edu"
__license__ = "MIT"

from .core.solver import CouzinSolver
from .core.systems import CouzinSystem
from .core.metrics import (
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
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    "CouzinSolver",
    "CouzinSystem",
    "ConfigManager",
    "DataHandler",
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
