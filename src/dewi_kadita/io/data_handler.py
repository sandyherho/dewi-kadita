"""Data handler for saving simulation results to CSV and NetCDF."""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class DataHandler:
    """Handle saving simulation data to various formats."""
    
    @staticmethod
    def save_order_parameters_csv(filepath: str, result: Dict[str, Any]):
        """
        Save polarization and rotation time series to CSV.
        
        Args:
            filepath: Output file path
            result: Simulation result dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            'Time': result['time'],
            'Polarization': result['polarization'],
            'Rotation': result['rotation']
        })
        
        df.to_csv(filepath, index=False, float_format='%.8e')
    
    @staticmethod
    def save_entropy_metrics_csv(filepath: str, metrics: Dict[str, Any]):
        """
        Save oceanic entropy metrics time series to CSV.
        
        Args:
            filepath: Output file path
            metrics: Metrics dictionary from compute_metrics_timeseries
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        columns = {
            'Time': metrics['time'],
            'Polarization': metrics['polarization'],
            'Rotation': metrics['rotation'],
            'Spread': metrics['spread'],
            'Order_Index': metrics['order_index'],
            'School_Cohesion_Entropy': metrics['school_cohesion_entropy'],
            'Polarization_Entropy': metrics['polarization_entropy'],
            'Depth_Stratification_Entropy': metrics['depth_stratification_entropy'],
            'Angular_Momentum_Entropy': metrics['angular_momentum_entropy'],
            'Nearest_Neighbor_Entropy': metrics['nearest_neighbor_entropy'],
            'Velocity_Correlation_Entropy': metrics['velocity_correlation_entropy'],
            'School_Shape_Entropy': metrics['school_shape_entropy'],
            'Oceanic_Schooling_Index': metrics['oceanic_schooling_index']
        }
        
        df = pd.DataFrame(columns)
        df.to_csv(filepath, index=False, float_format='%.8e')
    
    @staticmethod
    def save_entropy_summary_csv(filepath: str, metrics: Dict[str, Any]):
        """
        Save summary statistics of entropy metrics to CSV.
        
        Args:
            filepath: Output file path
            metrics: Metrics dictionary from compute_metrics_timeseries
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        metric_names = [
            'polarization',
            'rotation',
            'spread',
            'order_index',
            'school_cohesion_entropy',
            'polarization_entropy',
            'depth_stratification_entropy',
            'angular_momentum_entropy',
            'nearest_neighbor_entropy',
            'velocity_correlation_entropy',
            'school_shape_entropy',
            'oceanic_schooling_index'
        ]
        
        rows = []
        for name in metric_names:
            rows.append({
                'Metric': name,
                'Mean_2nd_Half': metrics.get(f'{name}_mean', np.nan),
                'Std_2nd_Half': metrics.get(f'{name}_std', np.nan),
                'Final_Value': metrics.get(f'{name}_final', np.nan),
                'Min': metrics.get(f'{name}_min', np.nan),
                'Max': metrics.get(f'{name}_max', np.nan)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, float_format='%.8e')
    
    @staticmethod
    def save_trajectory_csv(filepath: str, result: Dict[str, Any], step: int = -1):
        """
        Save fish positions and velocities at a given step.
        
        Args:
            filepath: Output file path
            result: Simulation result dictionary
            step: Time step index (-1 for final)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        pos = result['positions'][step]
        vel = result['velocities'][step]
        
        df = pd.DataFrame({
            'Fish_ID': np.arange(pos.shape[0]),
            'x': pos[:, 0],
            'y': pos[:, 1],
            'z': pos[:, 2],
            'vx': vel[:, 0],
            'vy': vel[:, 1],
            'vz': vel[:, 2]
        })
        
        df.to_csv(filepath, index=False, float_format='%.8e')
    
    @staticmethod
    def save_netcdf(
        filepath: str,
        result: Dict[str, Any],
        config: Dict[str, Any],
        metrics: Dict[str, Any] = None
    ):
        """
        Save complete simulation data to NetCDF format.
        
        Args:
            filepath: Output file path
            result: Simulation result dictionary
            config: Configuration dictionary
            metrics: Optional metrics dictionary from compute_metrics_timeseries
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            # Dimensions
            n_time = len(result['time'])
            n_fish = result['positions'].shape[1]
            
            nc.createDimension('time', n_time)
            nc.createDimension('fish', n_fish)
            nc.createDimension('spatial', 3)
            
            # Time coordinate
            nc_time = nc.createVariable('time', 'f8', ('time',), zlib=True)
            nc_time[:] = result['time']
            nc_time.units = "simulation_time_units"
            nc_time.long_name = "simulation_time"
            nc_time.axis = "T"
            
            # Fish coordinate
            nc_fish = nc.createVariable('fish', 'i4', ('fish',), zlib=True)
            nc_fish[:] = np.arange(n_fish)
            nc_fish.long_name = "fish_index"
            nc_fish.units = "1"
            
            # Spatial coordinate
            nc_spatial = nc.createVariable('spatial', 'i4', ('spatial',), zlib=True)
            nc_spatial[:] = np.array([0, 1, 2])
            nc_spatial.long_name = "spatial_dimension"
            nc_spatial.comment = "0=x, 1=y, 2=z (depth)"
            
            # Order parameters
            nc_pol = nc.createVariable('polarization', 'f8', ('time',), zlib=True)
            nc_pol[:] = result['polarization']
            nc_pol.units = "dimensionless"
            nc_pol.long_name = "school_polarization_order_parameter"
            nc_pol.valid_range = [0.0, 1.0]
            
            nc_rot = nc.createVariable('rotation', 'f8', ('time',), zlib=True)
            nc_rot[:] = result['rotation']
            nc_rot.units = "dimensionless"
            nc_rot.long_name = "school_rotation_order_parameter"
            nc_rot.valid_range = [0.0, 1.0]
            
            # Positions
            nc_pos = nc.createVariable('positions', 'f8', ('time', 'fish', 'spatial'), zlib=True)
            nc_pos[:] = result['positions']
            nc_pos.units = "simulation_length_units"
            nc_pos.long_name = "fish_positions"
            
            # Velocities
            nc_vel = nc.createVariable('velocities', 'f8', ('time', 'fish', 'spatial'), zlib=True)
            nc_vel[:] = result['velocities']
            nc_vel.units = "simulation_velocity_units"
            nc_vel.long_name = "fish_velocities"
            
            # Individual components
            for i, coord in enumerate(['x', 'y', 'z']):
                nc_var = nc.createVariable(coord, 'f8', ('time', 'fish'), zlib=True)
                nc_var[:] = result['positions'][:, :, i]
                nc_var.units = "simulation_length_units"
                nc_var.long_name = f"fish_{coord}_position"
                
                nc_var = nc.createVariable(f'v{coord}', 'f8', ('time', 'fish'), zlib=True)
                nc_var[:] = result['velocities'][:, :, i]
                nc_var.units = "simulation_velocity_units"
                nc_var.long_name = f"fish_{coord}_velocity"
            
            # Oceanic entropy metrics
            if metrics is not None:
                # Order index (new)
                nc_var = nc.createVariable('order_index', 'f8', ('time',), zlib=True)
                nc_var[:] = metrics['order_index']
                nc_var.units = "dimensionless"
                nc_var.long_name = "order_index"
                nc_var.valid_range = [0.0, 1.0]
                nc_var.description = "Simple order index: max(P, M). High = ordered."
                
                entropy_vars = [
                    ('school_cohesion_entropy', 'School cohesion entropy (NND distribution)'),
                    ('polarization_entropy', 'Polarization entropy (heading distribution)'),
                    ('depth_stratification_entropy', 'Depth stratification entropy'),
                    ('angular_momentum_entropy', 'Angular momentum entropy (milling)'),
                    ('nearest_neighbor_entropy', 'k-NN distance entropy'),
                    ('velocity_correlation_entropy', 'Velocity correlation entropy (pairwise alignment)'),
                    ('school_shape_entropy', 'School shape entropy (PCA)'),
                    ('oceanic_schooling_index', 'Composite Oceanic Schooling Index (high = disorder)')
                ]
                
                for var_name, description in entropy_vars:
                    nc_var = nc.createVariable(var_name, 'f8', ('time',), zlib=True)
                    nc_var[:] = metrics[var_name]
                    nc_var.units = "dimensionless"
                    nc_var.long_name = var_name
                    nc_var.valid_range = [0.0, 1.0]
                    nc_var.description = description
                
                # Spread
                nc_var = nc.createVariable('spread', 'f8', ('time',), zlib=True)
                nc_var[:] = metrics['spread']
                nc_var.units = "simulation_length_units"
                nc_var.long_name = "school_spread_rms"
            
            # Global attributes
            system = result['system']
            nc.title = "3D Couzin Model Fish Schooling Simulation"
            nc.scenario_name = config.get('scenario_name', 'unknown')
            nc.institution = "dewi-kadita"
            nc.source = "dewi-kadita v0.0.1"
            nc.history = f"Created {datetime.now().isoformat()}"
            nc.Conventions = "CF-1.8"
            
            # System parameters
            nc.n_fish = int(system.n_fish)
            nc.box_size = float(system.box_size)
            nc.speed = float(system.speed)
            nc.r_repulsion = float(system.r_repulsion)
            nc.r_orientation = float(system.r_orientation)
            nc.r_attraction = float(system.r_attraction)
            nc.max_turn = float(system.max_turn)
            nc.noise = float(system.noise)
            nc.blind_angle = float(system.blind_angle)
            
            # Simulation parameters
            nc.n_steps = int(result['n_steps'])
            nc.dt = float(result['dt'])
            nc.save_interval = int(result['save_interval'])
            
            # Summary statistics
            nc.final_polarization = float(result['final_polarization'])
            nc.mean_polarization = float(result['mean_polarization'])
            nc.final_rotation = float(result['final_rotation'])
            nc.mean_rotation = float(result['mean_rotation'])
            
            if metrics is not None:
                nc.final_order_index = float(metrics['order_index_final'])
                nc.mean_order_index = float(metrics['order_index_mean'])
                nc.final_oceanic_schooling_index = float(metrics['oceanic_schooling_index_final'])
                nc.mean_oceanic_schooling_index = float(metrics['oceanic_schooling_index_mean'])
            
            # References
            nc.author = "Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Alfita P. Handayani, Karina A. Sujatmiko, Kamaluddin Kasim, Rusmawan Suwarman, Dasapta E. Irawan"
            nc.email = "sandy.herho@email.ucr.edu"
            nc.license = "MIT"
