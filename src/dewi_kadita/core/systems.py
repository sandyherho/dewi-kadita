"""
3D Couzin Model System Definition for Fish Schooling.

The Couzin model (2002) describes collective animal motion through
three concentric behavioral zones:
    1. Zone of Repulsion (ZOR): Collision avoidance (highest priority)
    2. Zone of Orientation (ZOO): Alignment with neighbors
    3. Zone of Attraction (ZOA): Cohesion with distant neighbors

This produces four distinct collective states:
    - Swarm: Disordered aggregation
    - Torus: Rotating mill
    - Dynamic Parallel: Aligned but fluctuating
    - Highly Parallel: Stable aligned motion

References:
    Couzin, I.D., et al. (2002). Collective memory and spatial sorting
    in animal groups. Journal of Theoretical Biology, 218(1), 1-11.
"""

import numpy as np
from typing import Tuple, Optional


class CouzinSystem:
    """
    3D Couzin Model System for Fish Schooling.
    
    Attributes:
        n_fish: Number of fish in the school
        box_size: Side length of cubic simulation volume
        speed: Constant swimming speed of all fish
        r_repulsion: Radius of zone of repulsion (ZOR)
        r_orientation: Radius of zone of orientation (ZOO)
        r_attraction: Radius of zone of attraction (ZOA)
        max_turn: Maximum turning angle per timestep (radians)
        noise: Angular noise magnitude (radians)
        blind_angle: Rear blind angle in degrees (0-180)
        orientation_weight: Weight for orientation influence (0-1), default 1.0
        positions: (N, 3) array of fish positions
        velocities: (N, 3) array of fish velocities (unit vectors * speed)
    """
    
    def __init__(
        self,
        n_fish: int = 150,
        box_size: float = 25.0,
        speed: float = 1.0,
        r_repulsion: float = 1.0,
        r_orientation: float = 5.0,
        r_attraction: float = 10.0,
        max_turn: float = 0.3,
        noise: float = 0.1,
        blind_angle: float = 30.0,
        orientation_weight: float = 1.0,
        torus_init: bool = False,
        torus_radius: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Couzin fish schooling system.
        
        Args:
            n_fish: Number of fish
            box_size: Side length of cubic tank/ocean volume (L)
            speed: Constant swimming speed (v0)
            r_repulsion: Zone of repulsion radius (r_r)
            r_orientation: Zone of orientation radius (r_o)
            r_attraction: Zone of attraction radius (r_a)
            max_turn: Maximum turning angle per step in radians (θ_max)
            noise: Angular noise standard deviation in radians (σ)
            blind_angle: Rear blind angle in degrees (α)
            orientation_weight: Weight for orientation zone influence (0-1)
                               Lower values reduce alignment, promoting milling
            torus_init: If True, initialize fish in circular arrangement
                       with tangential velocities (for torus/milling state)
            torus_radius: Radius of initial circle for torus_init
                         If None, defaults to box_size * 0.25
            seed: Random seed for reproducibility
        """
        # Validate zone hierarchy
        if not (r_repulsion < r_orientation <= r_attraction):
            raise ValueError(
                f"Zone radii must satisfy r_r < r_o <= r_a, "
                f"got r_r={r_repulsion}, r_o={r_orientation}, r_a={r_attraction}"
            )
        
        self.n_fish = n_fish
        self.box_size = box_size
        self.speed = speed
        self.r_repulsion = r_repulsion
        self.r_orientation = r_orientation
        self.r_attraction = r_attraction
        self.max_turn = max_turn
        self.noise = noise
        self.blind_angle = blind_angle
        self.blind_angle_rad = np.radians(blind_angle)
        self.orientation_weight = np.clip(orientation_weight, 0.0, 1.0)
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize positions and velocities
        if torus_init:
            # Toroidal/circular initialization for milling behavior
            radius = torus_radius if torus_radius is not None else box_size * 0.25
            self.positions, self.velocities = self._initialize_torus(radius)
        else:
            # Standard random initialization
            self.positions = np.random.rand(n_fish, 3) * box_size
            vel = np.random.randn(n_fish, 3)
            norms = np.linalg.norm(vel, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.velocities = (vel / norms) * speed
    
    def _initialize_torus(self, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize fish in a toroidal/circular arrangement.
        
        Fish are placed on a circle in the XY plane centered in the box,
        with velocities tangent to the circle. This seeds rotational
        motion for torus/milling behavior.
        
        Args:
            radius: Radius of the initial circle
        
        Returns:
            Tuple of (positions, velocities) arrays
        """
        center = np.array([self.box_size / 2, self.box_size / 2, self.box_size / 2])
        
        positions = np.zeros((self.n_fish, 3))
        velocities = np.zeros((self.n_fish, 3))
        
        for i in range(self.n_fish):
            # Distribute angles uniformly with small noise
            theta = 2 * np.pi * i / self.n_fish + np.random.randn() * 0.1
            
            # Position on circle (XY plane) with radial noise
            r = radius + np.random.randn() * (radius * 0.15)
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            # Small vertical spread
            z = center[2] + np.random.randn() * (radius * 0.3)
            
            positions[i] = [x, y, z]
            
            # Tangential velocity (perpendicular to radius, in XY plane)
            # This creates initial rotation
            vx = -np.sin(theta)
            vy = np.cos(theta)
            vz = np.random.randn() * 0.1  # Small vertical component
            
            vel = np.array([vx, vy, vz])
            vel = vel / np.linalg.norm(vel) * self.speed
            velocities[i] = vel
        
        # Apply periodic boundary conditions
        positions = np.mod(positions, self.box_size)
        
        return positions, velocities
    
    def compute_polarization(self) -> float:
        """
        Compute polarization order parameter (alignment measure).
        
        P = (1/N) * |sum(v_i / |v_i|)|
        
        P ~ 0: disordered (random directions)
        P ~ 1: highly aligned (parallel motion)
        
        Returns:
            Polarization value between 0 and 1
        """
        # Normalize velocities to unit vectors
        norms = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        unit_vel = self.velocities / norms
        
        # Average unit velocity
        mean_vel = np.mean(unit_vel, axis=0)
        return np.linalg.norm(mean_vel)
    
    def compute_rotation(self) -> float:
        """
        Compute rotation order parameter (milling measure).
        
        M = (1/N) * |sum(v_i x r_i^c)|
        
        where r_i^c is unit vector from fish i to group centroid.
        
        M ~ 0: no rotation (parallel or random)
        M ~ 1: rotating mill (torus formation)
        
        Returns:
            Rotation value between 0 and 1
        """
        # Compute centroid
        centroid = np.mean(self.positions, axis=0)
        
        # Vectors from centroid to each fish
        r_to_fish = self.positions - centroid
        r_norms = np.linalg.norm(r_to_fish, axis=1, keepdims=True)
        r_norms[r_norms < 1e-10] = 1.0
        r_unit = r_to_fish / r_norms
        
        # Unit velocities
        v_norms = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        v_norms[v_norms < 1e-10] = 1.0
        v_unit = self.velocities / v_norms
        
        # Cross product for angular momentum direction
        cross = np.cross(r_unit, v_unit)
        
        # Mean angular momentum
        mean_cross = np.mean(cross, axis=0)
        return np.linalg.norm(mean_cross)
    
    def compute_centroid(self) -> np.ndarray:
        """
        Compute group centroid position.
        
        Returns:
            (3,) array of centroid coordinates
        """
        return np.mean(self.positions, axis=0)
    
    def compute_spread(self) -> float:
        """
        Compute group spread (standard deviation from centroid).
        
        Returns:
            RMS distance from centroid
        """
        centroid = self.compute_centroid()
        distances = np.linalg.norm(self.positions - centroid, axis=1)
        return np.sqrt(np.mean(distances ** 2))
    
    def compute_density(self) -> float:
        """
        Compute fish density.
        
        Returns:
            Number density (fish per unit volume)
        """
        return self.n_fish / (self.box_size ** 3)
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current system state.
        
        Returns:
            Tuple of (positions, velocities) arrays
        """
        return self.positions.copy(), self.velocities.copy()
    
    def set_state(self, positions: np.ndarray, velocities: np.ndarray):
        """
        Set system state.
        
        Args:
            positions: (N, 3) array of positions
            velocities: (N, 3) array of velocities
        """
        self.positions = positions.copy()
        self.velocities = velocities.copy()
    
    def apply_periodic_boundary(self):
        """Apply periodic boundary conditions to positions."""
        self.positions = np.mod(self.positions, self.box_size)
    
    def __repr__(self) -> str:
        return (
            f"CouzinSystem(N={self.n_fish}, L={self.box_size}, "
            f"v0={self.speed}, r_r={self.r_repulsion}, "
            f"r_o={self.r_orientation}, r_a={self.r_attraction}, "
            f"θ_max={self.max_turn:.2f}, σ={self.noise:.2f}, "
            f"blind={self.blind_angle}°, orient_wt={self.orientation_weight:.2f})"
        )
