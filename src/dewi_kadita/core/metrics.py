"""
Oceanic Entropy and Complexity Metrics for Fish Schooling.

This module implements specialized information-theoretic measures
designed for marine collective behavior analysis, distinct from
generic spatial entropy measures.

Measures implemented (unique to fish schooling):
1. School Cohesion Entropy - Nearest-neighbor distance distribution
2. Polarization Entropy - Angular distribution of headings
3. Depth Stratification Entropy - Vertical position distribution
4. Angular Momentum Entropy - Rotation state distribution
5. Nearest Neighbor Entropy - k-NN distance variability
6. Velocity Correlation Entropy - Pairwise velocity alignment distribution
7. School Shape Entropy - Principal component ratios
8. Oceanic Schooling Index (OSI) - Composite metric

References:
    - Couzin, I.D., et al. (2002). Journal of Theoretical Biology.
    - Partridge, B.L. (1982). Scientific American, 246(6), 114-123.
    - Handegard, N.O., et al. (2012). Current Biology, 22(13), 1213-1217.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.spatial import cKDTree, ConvexHull
from scipy.stats import entropy as scipy_entropy


def compute_school_cohesion_entropy(
    positions: np.ndarray,
    box_size: float,
    n_bins: int = 25
) -> float:
    """
    Compute entropy of nearest-neighbor distance distribution.
    
    This metric captures school cohesion patterns. Tight schools have
    narrow NND distributions (low entropy), while dispersed groups
    have broad distributions (high entropy).
    
    Ecological relevance: Predator defense efficiency, schooling tightness.
    
    Args:
        positions: (N, 3) array of fish positions
        box_size: Side length of cubic volume
        n_bins: Number of histogram bins
    
    Returns:
        Normalized cohesion entropy in [0, 1]
    """
    n = len(positions)
    if n < 2:
        return 0.5
    
    # Build KD-tree with periodic boundaries
    tree = cKDTree(positions, boxsize=box_size)
    
    # Get nearest neighbor distances (k=2 because first neighbor is self)
    distances, _ = tree.query(positions, k=2)
    nnd = distances[:, 1]  # Nearest neighbor distances
    
    # Histogram of NND
    max_dist = box_size / 2  # Maximum meaningful distance
    hist, _ = np.histogram(nnd, bins=n_bins, range=(0, max_dist))
    
    # Normalize to probability
    prob = hist / hist.sum() if hist.sum() > 0 else hist
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.5
    
    # Shannon entropy
    H = -np.sum(prob * np.log(prob))
    
    # Normalize by maximum entropy
    H_max = np.log(n_bins)
    
    return H / H_max if H_max > 0 else 0.5


def compute_polarization_entropy(
    velocities: np.ndarray,
    n_bins_theta: int = 12,
    n_bins_phi: int = 24
) -> float:
    """
    Compute entropy of velocity heading distribution on unit sphere.
    
    Measures how spread out fish orientations are. Highly aligned
    schools have low entropy; disorganized groups have high entropy.
    
    Ecological relevance: Migration coordination, predator evasion readiness.
    
    Args:
        velocities: (N, 3) array of fish velocities
        n_bins_theta: Bins for polar angle [0, π]
        n_bins_phi: Bins for azimuthal angle [0, 2π]
    
    Returns:
        Normalized polarization entropy in [0, 1]
    """
    # Normalize to unit vectors
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    unit_vel = velocities / norms
    
    # Convert to spherical coordinates
    theta = np.arccos(np.clip(unit_vel[:, 2], -1, 1))  # [0, π]
    phi = np.arctan2(unit_vel[:, 1], unit_vel[:, 0]) + np.pi  # [0, 2π]
    
    # 2D histogram weighted by solid angle
    theta_edges = np.linspace(0, np.pi, n_bins_theta + 1)
    phi_edges = np.linspace(0, 2 * np.pi, n_bins_phi + 1)
    
    hist, _, _ = np.histogram2d(theta, phi, bins=[theta_edges, phi_edges])
    
    # Weight by sin(theta) for proper spherical measure
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    weights = np.sin(theta_centers)[:, np.newaxis]
    
    weighted_hist = hist * weights
    
    # Probability distribution
    total = weighted_hist.sum()
    if total == 0:
        return 0.5
    
    prob = weighted_hist.flatten() / total
    prob = prob[prob > 0]
    
    # Shannon entropy
    H = -np.sum(prob * np.log(prob))
    
    # Maximum entropy (uniform on sphere)
    H_max = np.log(n_bins_theta * n_bins_phi)
    
    return H / H_max if H_max > 0 else 0.5


def compute_depth_stratification_entropy(
    positions: np.ndarray,
    box_size: float,
    n_bins: int = 15
) -> float:
    """
    Compute entropy of vertical (z-axis) position distribution.
    
    Fish schools often stratify by depth following thermoclines,
    oxygen gradients, or light levels. This metric captures
    the uniformity of depth distribution.
    
    Ecological relevance: Thermocline tracking, vertical migration patterns.
    
    Args:
        positions: (N, 3) array of fish positions
        box_size: Side length of cubic volume
        n_bins: Number of depth bins
    
    Returns:
        Normalized depth entropy in [0, 1]
    """
    # Extract z-coordinates (depth)
    depths = positions[:, 2]
    
    # Histogram
    hist, _ = np.histogram(depths, bins=n_bins, range=(0, box_size))
    
    # Probability
    prob = hist / hist.sum() if hist.sum() > 0 else hist
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.5
    
    # Shannon entropy
    H = -np.sum(prob * np.log(prob))
    H_max = np.log(n_bins)
    
    return H / H_max if H_max > 0 else 0.5


def compute_angular_momentum_entropy(
    positions: np.ndarray,
    velocities: np.ndarray,
    n_bins: int = 20
) -> float:
    """
    Compute entropy of individual angular momentum contributions.
    
    Measures the distribution of how each fish contributes to
    collective rotation (milling behavior). Uniform contributions
    indicate coherent milling; variable contributions indicate
    mixed states.
    
    Ecological relevance: Milling behavior detection, torus formation.
    
    Args:
        positions: (N, 3) fish positions
        velocities: (N, 3) fish velocities
        n_bins: Number of histogram bins
    
    Returns:
        Normalized angular momentum entropy in [0, 1]
    """
    n = len(positions)
    if n < 3:
        return 0.5
    
    # Compute centroid
    centroid = np.mean(positions, axis=0)
    
    # Vectors from centroid
    r = positions - centroid
    r_norms = np.linalg.norm(r, axis=1, keepdims=True)
    r_norms[r_norms < 1e-10] = 1.0
    r_unit = r / r_norms
    
    # Normalize velocities
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    v_norms[v_norms < 1e-10] = 1.0
    v_unit = velocities / v_norms
    
    # Angular momentum per fish (magnitude of r × v)
    cross = np.cross(r_unit, v_unit)
    L_mag = np.linalg.norm(cross, axis=1)  # [0, 1] range
    
    # Histogram
    hist, _ = np.histogram(L_mag, bins=n_bins, range=(0, 1))
    
    prob = hist / hist.sum() if hist.sum() > 0 else hist
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.5
    
    H = -np.sum(prob * np.log(prob))
    H_max = np.log(n_bins)
    
    return H / H_max if H_max > 0 else 0.5


def compute_nearest_neighbor_entropy(
    positions: np.ndarray,
    box_size: float,
    k: int = 5,
    n_bins: int = 20
) -> float:
    """
    Compute entropy of k-nearest neighbor distance variability.
    
    Extends cohesion entropy to consider local density structure.
    Fish in natural schools maintain consistent spacing with
    multiple neighbors.
    
    Ecological relevance: Social network structure, information transfer.
    
    Args:
        positions: (N, 3) fish positions
        box_size: Simulation volume size
        k: Number of nearest neighbors to consider
        n_bins: Histogram bins
    
    Returns:
        Normalized k-NN entropy in [0, 1]
    """
    n = len(positions)
    if n < k + 1:
        return 0.5
    
    # KD-tree with PBC
    tree = cKDTree(positions, boxsize=box_size)
    
    # Get k nearest neighbors
    distances, _ = tree.query(positions, k=k+1)
    knn_dists = distances[:, 1:]  # Exclude self
    
    # Compute coefficient of variation for each fish
    means = np.mean(knn_dists, axis=1)
    stds = np.std(knn_dists, axis=1)
    cv = stds / (means + 1e-10)
    
    # Histogram of CV values
    hist, _ = np.histogram(cv, bins=n_bins, range=(0, 2))
    
    prob = hist / hist.sum() if hist.sum() > 0 else hist
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.5
    
    H = -np.sum(prob * np.log(prob))
    H_max = np.log(n_bins)
    
    return H / H_max if H_max > 0 else 0.5


def compute_velocity_correlation_entropy(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: float,
    n_bins: int = 20
) -> float:
    """
    Compute entropy of pairwise velocity correlation distribution.
    
    FIXED VERSION: Measures the distribution of pairwise velocity
    dot products (correlations). 
    
    - Highly aligned schools: all correlations ~ 1 → concentrated 
      distribution → LOW entropy
    - Disordered schools: correlations spread from -1 to 1 → 
      broad distribution → HIGH entropy
    
    Ecological relevance: Information transfer efficiency, collective alignment.
    
    Args:
        positions: (N, 3) fish positions
        velocities: (N, 3) fish velocities
        box_size: Simulation volume size
        n_bins: Number of histogram bins
    
    Returns:
        Normalized velocity correlation entropy in [0, 1]
    """
    n = len(positions)
    if n < 2:
        return 0.5
    
    # Normalize velocities
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    v_norms[v_norms < 1e-10] = 1.0
    v_unit = velocities / v_norms
    
    # Compute pairwise velocity correlations (dot products)
    # Result: correlation matrix where entry (i,j) = v_i · v_j
    # For aligned school: all values ~ 1
    # For disordered: values spread from -1 to 1
    correlations = np.dot(v_unit, v_unit.T)
    
    # Get upper triangle (exclude diagonal and avoid double counting)
    triu_idx = np.triu_indices(n, k=1)
    all_corr = correlations[triu_idx]
    
    # Shift correlations from [-1, 1] to [0, 1] for histogram
    corr_shifted = (all_corr + 1.0) / 2.0
    
    # Histogram of correlation values
    hist, _ = np.histogram(corr_shifted, bins=n_bins, range=(0, 1))
    
    prob = hist / hist.sum() if hist.sum() > 0 else hist
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.5
    
    # Shannon entropy
    H = -np.sum(prob * np.log(prob))
    H_max = np.log(n_bins)
    
    return H / H_max if H_max > 0 else 0.5


def compute_school_shape_entropy(
    positions: np.ndarray
) -> float:
    """
    Compute entropy based on school shape (principal component ratios).
    
    Fish schools adopt various shapes (spherical, elongated, flat disc)
    depending on behavioral state and environment. This metric captures
    shape variability through PCA eigenvalue ratios.
    
    Ecological relevance: Morphological adaptation, predator response.
    
    Args:
        positions: (N, 3) fish positions
    
    Returns:
        Normalized shape entropy in [0, 1]
    """
    n = len(positions)
    if n < 4:
        return 0.5
    
    # Center positions
    centered = positions - np.mean(positions, axis=0)
    
    # Covariance matrix
    cov = np.cov(centered.T)
    
    # Eigenvalues (principal variances)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid zeros
    
    # Normalize eigenvalues to sum to 1 (probability-like)
    ev_norm = eigenvalues / eigenvalues.sum()
    
    # Shannon entropy of eigenvalue distribution
    H = -np.sum(ev_norm * np.log(ev_norm))
    
    # Maximum entropy for 3 components
    H_max = np.log(3)
    
    return H / H_max if H_max > 0 else 0.5


def compute_oceanic_schooling_index(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: float
) -> Dict[str, float]:
    """
    Compute composite Oceanic Schooling Index (OSI) combining all metrics.
    
    The OSI is specifically designed for fish school analysis, weighting
    metrics by their ecological relevance to marine collective behavior.
    
    OSI = Σ(w_i * H_i) where weights emphasize schooling-specific measures.
    
    Interpretation:
        - OSI ~ 0: Highly ordered school (low entropy)
        - OSI ~ 1: Disordered aggregation (high entropy)
    
    Args:
        positions: (N, 3) fish positions
        velocities: (N, 3) fish velocities
        box_size: Simulation volume size
    
    Returns:
        Dictionary with all entropy measures and composite index
    """
    # Compute all metrics
    H_cohesion = compute_school_cohesion_entropy(positions, box_size)
    H_polarization = compute_polarization_entropy(velocities)
    H_depth = compute_depth_stratification_entropy(positions, box_size)
    H_angular = compute_angular_momentum_entropy(positions, velocities)
    H_knn = compute_nearest_neighbor_entropy(positions, box_size)
    H_vel_corr = compute_velocity_correlation_entropy(positions, velocities, box_size)
    H_shape = compute_school_shape_entropy(positions)
    
    # Weights optimized for fish schooling (sum to 1)
    # Higher weight on polarization and cohesion as primary schooling indicators
    weights = {
        'cohesion': 0.18,
        'polarization': 0.28,  # Increased - most important for alignment
        'depth': 0.08,
        'angular': 0.08,
        'knn': 0.10,
        'vel_corr': 0.18,  # Now properly computed
        'shape': 0.10
    }
    
    # Composite index: high entropy = high OSI = disorder
    OSI = (
        weights['cohesion'] * H_cohesion +
        weights['polarization'] * H_polarization +
        weights['depth'] * H_depth +
        weights['angular'] * H_angular +
        weights['knn'] * H_knn +
        weights['vel_corr'] * H_vel_corr +
        weights['shape'] * H_shape
    )
    
    return {
        'school_cohesion_entropy': H_cohesion,
        'polarization_entropy': H_polarization,
        'depth_stratification_entropy': H_depth,
        'angular_momentum_entropy': H_angular,
        'nearest_neighbor_entropy': H_knn,
        'velocity_correlation_entropy': H_vel_corr,
        'school_shape_entropy': H_shape,
        'oceanic_schooling_index': OSI
    }


def compute_order_index(polarization: float, rotation: float) -> float:
    """
    Compute simple Order Index based on order parameters.
    
    A simpler, more interpretable alternative to entropy-based OSI.
    
    Args:
        polarization: Polarization order parameter P in [0, 1]
        rotation: Rotation order parameter M in [0, 1]
    
    Returns:
        Order Index in [0, 1] where:
        - High value = ordered (either aligned or milling)
        - Low value = disordered swarm
    """
    return max(polarization, rotation)


def compute_all_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size: float,
    speed: float = 1.0
) -> Dict[str, float]:
    """
    Compute all metrics for a single time step.
    
    Args:
        positions: (N, 3) fish positions
        velocities: (N, 3) fish velocities
        box_size: Simulation volume size
        speed: Fish swimming speed (for normalization)
    
    Returns:
        Dictionary with all metrics
    """
    n = len(positions)
    
    # Standard order parameters
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    v_norms[v_norms < 1e-10] = 1.0
    unit_vel = velocities / v_norms
    
    # Polarization
    polarization = np.linalg.norm(np.mean(unit_vel, axis=0))
    
    # Rotation
    centroid = np.mean(positions, axis=0)
    r = positions - centroid
    r_norms = np.linalg.norm(r, axis=1, keepdims=True)
    r_norms[r_norms < 1e-10] = 1.0
    r_unit = r / r_norms
    cross = np.cross(r_unit, unit_vel)
    rotation = np.linalg.norm(np.mean(cross, axis=0))
    
    # Oceanic entropy measures
    oceanic = compute_oceanic_schooling_index(positions, velocities, box_size)
    
    # Simple order index
    order_index = compute_order_index(polarization, rotation)
    
    # Group spread
    spread = np.sqrt(np.mean(np.linalg.norm(r, axis=1) ** 2))
    
    return {
        'polarization': polarization,
        'rotation': rotation,
        'spread': spread,
        'order_index': order_index,
        **oceanic
    }


def compute_metrics_timeseries(
    positions_history: np.ndarray,
    velocities_history: np.ndarray,
    time_array: np.ndarray,
    box_size: float,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute all metrics for entire simulation trajectory.
    
    Args:
        positions_history: (T, N, 3) position trajectory
        velocities_history: (T, N, 3) velocity trajectory
        time_array: (T,) time values
        box_size: Simulation volume size
        verbose: Print progress
    
    Returns:
        Dictionary with time series of all metrics and summary statistics
    """
    from tqdm import tqdm
    
    n_times = len(time_array)
    
    # Metric keys
    metrics_keys = [
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
    
    results = {key: np.zeros(n_times) for key in metrics_keys}
    results['time'] = time_array.copy()
    
    iterator = tqdm(
        range(n_times),
        desc="      Computing metrics",
        ncols=70,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
        disable=not verbose
    )
    
    for t in iterator:
        metrics = compute_all_metrics(
            positions_history[t],
            velocities_history[t],
            box_size
        )
        
        for key in metrics_keys:
            results[key][t] = metrics[key]
    
    # Compute summary statistics (second half for equilibrium)
    half_idx = n_times // 2
    for key in metrics_keys:
        results[f'{key}_mean'] = np.mean(results[key][half_idx:])
        results[f'{key}_std'] = np.std(results[key][half_idx:])
        results[f'{key}_final'] = results[key][-1]
        results[f'{key}_min'] = np.min(results[key])
        results[f'{key}_max'] = np.max(results[key])
    
    return results
