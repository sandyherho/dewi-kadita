"""
3D Couzin Model Solver with Numba Acceleration.

Implements the Couzin zone-based dynamics with optional JIT compilation
for high-performance simulation of fish schooling behavior.

The model uses three behavioral zones:
    - Zone of Repulsion (ZOR): Avoid collisions (priority)
    - Zone of Orientation (ZOO): Align with neighbors
    - Zone of Attraction (ZOA): Move toward distant fish

References:
    Couzin, I.D., et al. (2002). Collective memory and spatial sorting
    in animal groups. Journal of Theoretical Biology, 218(1), 1-11.
"""

import numpy as np
from typing import Dict, Any, Optional
from tqdm import tqdm

from .systems import CouzinSystem

# Try to import numba, fall back to pure numpy if unavailable
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True)
def _compute_distance_pbc(pos1: np.ndarray, pos2: np.ndarray, box_size: float) -> float:
    """Compute distance with periodic boundary conditions."""
    delta = pos1 - pos2
    delta = delta - box_size * np.round(delta / box_size)
    return np.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)


@jit(nopython=True, cache=True)
def _compute_delta_pbc(pos1: np.ndarray, pos2: np.ndarray, box_size: float) -> np.ndarray:
    """Compute displacement vector with periodic boundary conditions."""
    delta = pos2 - pos1  # Vector from pos1 to pos2
    delta = delta - box_size * np.round(delta / box_size)
    return delta


@jit(nopython=True, cache=True)
def _is_in_visual_field(vel: np.ndarray, delta: np.ndarray, blind_angle_rad: float) -> bool:
    """
    Check if neighbor is within visual field (not in blind spot behind).
    
    Args:
        vel: Current fish velocity (direction)
        delta: Vector from current fish to neighbor
        blind_angle_rad: Half-angle of rear blind cone in radians
    
    Returns:
        True if neighbor is visible
    """
    if blind_angle_rad <= 0:
        return True
    
    vel_norm = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    delta_norm = np.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
    
    if vel_norm < 1e-10 or delta_norm < 1e-10:
        return True
    
    # Dot product gives cos(angle)
    cos_angle = (vel[0]*delta[0] + vel[1]*delta[1] + vel[2]*delta[2]) / (vel_norm * delta_norm)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    # If cos_angle < cos(pi - blind_angle), neighbor is in blind spot
    # cos(pi - x) = -cos(x)
    cos_threshold = -np.cos(blind_angle_rad)
    
    return cos_angle > cos_threshold


@jit(nopython=True, cache=True)
def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    return vec / norm


@jit(nopython=True, cache=True)
def _rotate_toward(current_dir: np.ndarray, target_dir: np.ndarray, 
                   max_angle: float, noise: float) -> np.ndarray:
    """
    Rotate current direction toward target with maximum angle constraint and noise.
    
    Args:
        current_dir: Current unit direction vector
        target_dir: Target unit direction vector
        max_angle: Maximum rotation angle in radians
        noise: Angular noise standard deviation
    
    Returns:
        New unit direction vector
    """
    # Handle zero vectors
    target_norm = np.sqrt(target_dir[0]**2 + target_dir[1]**2 + target_dir[2]**2)
    if target_norm < 1e-10:
        target_dir = current_dir.copy()
    else:
        target_dir = target_dir / target_norm
    
    # Compute angle between current and target
    dot = current_dir[0]*target_dir[0] + current_dir[1]*target_dir[1] + current_dir[2]*target_dir[2]
    dot = max(-1.0, min(1.0, dot))
    angle = np.arccos(dot)
    
    # If already aligned, just add noise
    if angle < 1e-10:
        new_dir = current_dir.copy()
    elif angle <= max_angle:
        # Can reach target direction
        new_dir = target_dir.copy()
    else:
        # Interpolate: rotate by max_angle toward target
        # Using spherical linear interpolation approximation
        t = max_angle / angle
        new_dir = current_dir * (1 - t) + target_dir * t
        norm = np.sqrt(new_dir[0]**2 + new_dir[1]**2 + new_dir[2]**2)
        if norm > 1e-10:
            new_dir = new_dir / norm
        else:
            new_dir = current_dir.copy()
    
    # Add angular noise (random rotation)
    if noise > 0:
        # Generate random perturbation
        noise_vec = np.array([
            np.random.randn() * noise,
            np.random.randn() * noise,
            np.random.randn() * noise
        ])
        new_dir = new_dir + noise_vec
        norm = np.sqrt(new_dir[0]**2 + new_dir[1]**2 + new_dir[2]**2)
        if norm > 1e-10:
            new_dir = new_dir / norm
    
    return new_dir


@jit(nopython=True, parallel=True, cache=True)
def _update_velocities_numba(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: float,
    r_repulsion: float,
    r_orientation: float,
    r_attraction: float,
    max_turn: float,
    noise: float,
    blind_angle_rad: float,
    speed: float,
    orientation_weight: float
) -> np.ndarray:
    """
    Numba-accelerated velocity update with Couzin zone dynamics.
    
    Implements the three-zone model:
    1. If any neighbors in ZOR: move away (repulsion only)
    2. Else: combine orientation and attraction responses
    
    Args:
        positions: (N, 3) fish positions
        velocities: (N, 3) fish velocities
        box_size: Simulation box size
        r_repulsion: Zone of repulsion radius
        r_orientation: Zone of orientation radius  
        r_attraction: Zone of attraction radius
        max_turn: Maximum turning angle per step
        noise: Angular noise magnitude
        blind_angle_rad: Rear blind angle in radians
        speed: Constant swimming speed
        orientation_weight: Weight for orientation influence (0-1)
                           Lower values reduce alignment, promoting milling
    
    Returns:
        (N, 3) updated velocity array
    """
    n = positions.shape[0]
    new_vel = np.zeros((n, 3), dtype=np.float64)
    
    for i in prange(n):
        # Current direction
        vel_i = velocities[i]
        vel_norm = np.sqrt(vel_i[0]**2 + vel_i[1]**2 + vel_i[2]**2)
        if vel_norm < 1e-10:
            vel_norm = 1.0
        current_dir = vel_i / vel_norm
        
        # Accumulate zone responses
        d_repulsion = np.zeros(3)
        d_orientation = np.zeros(3)
        d_attraction = np.zeros(3)
        
        n_repulsion = 0
        n_orientation = 0
        n_attraction = 0
        
        for j in range(n):
            if i == j:
                continue
            
            # Compute displacement with PBC
            delta = _compute_delta_pbc(positions[i], positions[j], box_size)
            dist = np.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
            
            if dist < 1e-10:
                continue
            
            # Check visual field (is neighbor visible?)
            if not _is_in_visual_field(vel_i, delta, blind_angle_rad):
                continue
            
            # Unit vector toward neighbor
            delta_unit = delta / dist
            
            # Zone of Repulsion (highest priority)
            if dist < r_repulsion:
                # Move AWAY from neighbor
                d_repulsion[0] -= delta_unit[0]
                d_repulsion[1] -= delta_unit[1]
                d_repulsion[2] -= delta_unit[2]
                n_repulsion += 1
            
            # Zone of Orientation (only if outside ZOR)
            elif dist < r_orientation:
                # Align with neighbor's direction
                vel_j = velocities[j]
                vel_j_norm = np.sqrt(vel_j[0]**2 + vel_j[1]**2 + vel_j[2]**2)
                if vel_j_norm > 1e-10:
                    d_orientation[0] += vel_j[0] / vel_j_norm
                    d_orientation[1] += vel_j[1] / vel_j_norm
                    d_orientation[2] += vel_j[2] / vel_j_norm
                    n_orientation += 1
            
            # Zone of Attraction (only if outside ZOO)
            elif dist < r_attraction:
                # Move TOWARD neighbor
                d_attraction[0] += delta_unit[0]
                d_attraction[1] += delta_unit[1]
                d_attraction[2] += delta_unit[2]
                n_attraction += 1
        
        # Compute desired direction based on zone responses
        if n_repulsion > 0:
            # Repulsion takes priority - ignore other zones
            desired_dir = _normalize(d_repulsion)
        elif n_orientation > 0 or n_attraction > 0:
            # Combine orientation and attraction with orientation weight
            combined = np.zeros(3)
            if n_orientation > 0:
                # Apply orientation weight (key for torus formation)
                combined[0] += d_orientation[0] * orientation_weight
                combined[1] += d_orientation[1] * orientation_weight
                combined[2] += d_orientation[2] * orientation_weight
            if n_attraction > 0:
                combined[0] += d_attraction[0]
                combined[1] += d_attraction[1]
                combined[2] += d_attraction[2]
            desired_dir = _normalize(combined)
        else:
            # No neighbors - keep current direction
            desired_dir = current_dir.copy()
        
        # Rotate toward desired direction with constraints
        new_dir = _rotate_toward(current_dir, desired_dir, max_turn, noise)
        
        # Apply speed
        new_vel[i, 0] = new_dir[0] * speed
        new_vel[i, 1] = new_dir[1] * speed
        new_vel[i, 2] = new_dir[2] * speed
    
    return new_vel


def _update_velocities_numpy(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: float,
    r_repulsion: float,
    r_orientation: float,
    r_attraction: float,
    max_turn: float,
    noise: float,
    blind_angle_rad: float,
    speed: float,
    orientation_weight: float
) -> np.ndarray:
    """
    NumPy-based velocity update (fallback when Numba unavailable).
    
    This is a vectorized but slower implementation for compatibility.
    """
    n = positions.shape[0]
    new_vel = np.zeros((n, 3), dtype=np.float64)
    
    # Compute pairwise distances with PBC
    delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    delta = delta - box_size * np.round(delta / box_size)
    dists = np.linalg.norm(delta, axis=2)
    
    # Normalize velocities
    vel_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    vel_norms[vel_norms < 1e-10] = 1.0
    unit_vel = velocities / vel_norms
    
    for i in range(n):
        current_dir = unit_vel[i]
        
        # Zone masks (excluding self)
        not_self = np.ones(n, dtype=bool)
        not_self[i] = False
        
        # Visual field check (simplified)
        delta_i = -delta[i]  # Vectors from i to others
        delta_norms = dists[i].copy()
        delta_norms[delta_norms < 1e-10] = 1.0
        delta_unit = delta_i / delta_norms[:, np.newaxis]
        
        # Dot product for visual field
        if blind_angle_rad > 0:
            cos_angles = np.sum(current_dir * delta_unit, axis=1)
            cos_threshold = -np.cos(blind_angle_rad)
            visible = cos_angles > cos_threshold
        else:
            visible = np.ones(n, dtype=bool)
        visible[i] = False
        
        # Zone assignments
        in_zor = (dists[i] < r_repulsion) & visible & not_self
        in_zoo = (dists[i] >= r_repulsion) & (dists[i] < r_orientation) & visible & not_self
        in_zoa = (dists[i] >= r_orientation) & (dists[i] < r_attraction) & visible & not_self
        
        # Compute zone responses
        if np.any(in_zor):
            d_rep = -np.sum(delta_unit[in_zor], axis=0)
            d_rep_norm = np.linalg.norm(d_rep)
            desired_dir = d_rep / d_rep_norm if d_rep_norm > 1e-10 else current_dir
        else:
            d_orient = np.sum(unit_vel[in_zoo], axis=0) * orientation_weight if np.any(in_zoo) else np.zeros(3)
            d_attract = np.sum(delta_unit[in_zoa], axis=0) if np.any(in_zoa) else np.zeros(3)
            combined = d_orient + d_attract
            combined_norm = np.linalg.norm(combined)
            desired_dir = combined / combined_norm if combined_norm > 1e-10 else current_dir
        
        # Rotation with max turn constraint
        dot = np.clip(np.dot(current_dir, desired_dir), -1, 1)
        angle = np.arccos(dot)
        
        if angle < 1e-10:
            new_dir = current_dir.copy()
        elif angle <= max_turn:
            new_dir = desired_dir.copy()
        else:
            t = max_turn / angle
            new_dir = current_dir * (1 - t) + desired_dir * t
            new_dir /= np.linalg.norm(new_dir)
        
        # Add noise
        if noise > 0:
            noise_vec = np.random.randn(3) * noise
            new_dir = new_dir + noise_vec
            new_dir /= np.linalg.norm(new_dir)
        
        new_vel[i] = new_dir * speed
    
    return new_vel


class CouzinSolver:
    """
    Solver for the 3D Couzin Fish Schooling Model.
    
    Simulates collective motion with zone-based behavioral rules
    producing emergent schooling patterns.
    """
    
    def __init__(self, dt: float = 0.1, use_numba: bool = True):
        """
        Initialize solver.
        
        Args:
            dt: Time step
            use_numba: Use Numba JIT acceleration if available
        """
        self.dt = dt
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        if use_numba and not NUMBA_AVAILABLE:
            import warnings
            warnings.warn(
                "Numba not available, falling back to NumPy implementation. "
                "Install numba for 10-100x speedup: pip install numba"
            )
    
    def solve(
        self,
        system: CouzinSystem,
        n_steps: int = 1000,
        save_interval: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run Couzin fish schooling simulation.
        
        Args:
            system: CouzinSystem instance
            n_steps: Number of simulation steps
            save_interval: Save state every N steps
            verbose: Print progress information
        
        Returns:
            Dictionary with simulation results including:
            - time: Time array
            - positions: Position history (T, N, 3)
            - velocities: Velocity history (T, N, 3)
            - polarization: Polarization time series
            - rotation: Rotation time series
            - Various statistics
        """
        n_saves = n_steps // save_interval + 1
        n_fish = system.n_fish
        
        # Storage arrays
        positions_history = np.zeros((n_saves, n_fish, 3))
        velocities_history = np.zeros((n_saves, n_fish, 3))
        polarization = np.zeros(n_saves)
        rotation = np.zeros(n_saves)
        time_array = np.zeros(n_saves)
        
        # Initial state
        positions_history[0] = system.positions.copy()
        velocities_history[0] = system.velocities.copy()
        polarization[0] = system.compute_polarization()
        rotation[0] = system.compute_rotation()
        time_array[0] = 0.0
        
        save_idx = 1
        
        if verbose:
            print(f"      Running {n_steps} steps with dt={self.dt}")
            print(f"      Numba acceleration: {'enabled' if self.use_numba else 'disabled'}")
            print(f"      Orientation weight: {system.orientation_weight:.2f}")
            print(f"      Initial polarization: {polarization[0]:.4f}")
            print(f"      Initial rotation: {rotation[0]:.4f}")
        
        # Select update function
        update_func = _update_velocities_numba if self.use_numba else _update_velocities_numpy
        
        # Warmup JIT compilation
        if self.use_numba and verbose:
            print("      Compiling Numba kernels (first run only)...")
            _ = update_func(
                system.positions[:10].copy(),
                system.velocities[:10].copy(),
                system.box_size,
                system.r_repulsion,
                system.r_orientation,
                system.r_attraction,
                system.max_turn,
                system.noise,
                system.blind_angle_rad,
                system.speed,
                system.orientation_weight
            )
        
        # Main simulation loop
        iterator = tqdm(
            range(1, n_steps + 1),
            desc="      Simulating",
            ncols=70,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
            disable=not verbose
        )
        
        for step in iterator:
            # Update velocities (zone-based dynamics)
            system.velocities = update_func(
                system.positions,
                system.velocities,
                system.box_size,
                system.r_repulsion,
                system.r_orientation,
                system.r_attraction,
                system.max_turn,
                system.noise,
                system.blind_angle_rad,
                system.speed,
                system.orientation_weight
            )
            
            # Update positions
            system.positions += system.velocities * self.dt
            
            # Apply periodic boundaries
            system.apply_periodic_boundary()
            
            # Save state
            if step % save_interval == 0:
                positions_history[save_idx] = system.positions.copy()
                velocities_history[save_idx] = system.velocities.copy()
                polarization[save_idx] = system.compute_polarization()
                rotation[save_idx] = system.compute_rotation()
                time_array[save_idx] = step * self.dt
                save_idx += 1
        
        # Compile results
        half_idx = save_idx // 2
        result = {
            'time': time_array[:save_idx],
            'positions': positions_history[:save_idx],
            'velocities': velocities_history[:save_idx],
            'polarization': polarization[:save_idx],
            'rotation': rotation[:save_idx],
            'system': system,
            'n_steps': n_steps,
            'dt': self.dt,
            'save_interval': save_interval,
            'final_polarization': polarization[save_idx - 1],
            'final_rotation': rotation[save_idx - 1],
            'mean_polarization': np.mean(polarization[half_idx:save_idx]),
            'std_polarization': np.std(polarization[half_idx:save_idx]),
            'mean_rotation': np.mean(rotation[half_idx:save_idx]),
            'std_rotation': np.std(rotation[half_idx:save_idx]),
            'max_polarization': np.max(polarization[:save_idx]),
            'min_polarization': np.min(polarization[:save_idx]),
            'max_rotation': np.max(rotation[:save_idx]),
            'min_rotation': np.min(rotation[:save_idx])
        }
        
        if verbose:
            print(f"\n      Final polarization: {result['final_polarization']:.4f}")
            print(f"      Final rotation: {result['final_rotation']:.4f}")
            print(f"      Mean polarization (2nd half): {result['mean_polarization']:.4f}")
            print(f"      Mean rotation (2nd half): {result['mean_rotation']:.4f}")
        
        return result
