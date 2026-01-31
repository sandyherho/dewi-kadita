# `dewi-kadita`: A Python Library for 3D Idealized Couzin Model Fish Schooling Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/dewi-kadita.svg)](https://pypi.org/project/dewi-kadita/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1106385474.svg)](https://doi.org/10.5281/zenodo.17760126)


[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)
[![Numba](https://img.shields.io/badge/Numba-%2300A3E0.svg?logo=numba&logoColor=white)](https://numba.pydata.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%23000000.svg)](https://python-pillow.org/)
[![tqdm](https://img.shields.io/badge/tqdm-%23FFC107.svg)](https://tqdm.github.io/)

*Named after Kanjeng Ratu Kidul, the legendary Queen of the Southern Sea in Javanese mythology*

A Python library for simulating fish schooling behavior using the 3D Couzin model with Numba JIT acceleration and comprehensive oceanic entropy metrics designed specifically for marine collective motion analysis.

![Animation](https://raw.githubusercontent.com/sandyherho/dewi-kadita/main/.assets/case4_highly_parallel_animation.gif)

## Model

The Couzin model (2002) describes collective animal motion through three concentric behavioral zones around each individual:

### Zone-Based Behavior

Each fish $i$ responds to neighbors based on their distance:

**Zone of Repulsion (ZOR)** — radius $r_r$: Collision avoidance takes priority
$$\mathbf{d}_i^{(r)} = -\sum_{j \in ZOR} \frac{\mathbf{r}_{ij}}{|\mathbf{r}_{ij}|}$$

**Zone of Orientation (ZOO)** — radius $r_o > r_r$: Align with neighbors' headings
$$\mathbf{d}_i^{(o)} = \sum_{j \in ZOO} \hat{\mathbf{v}}_j$$

**Zone of Attraction (ZOA)** — radius $r_a > r_o$: Move toward distant neighbors
$$\mathbf{d}_i^{(a)} = \sum_{j \in ZOA} \frac{\mathbf{r}_{ij}}{|\mathbf{r}_{ij}|}$$

### Update Rules

The desired direction combines zone responses with priority to repulsion:

$$\mathbf{d}_i = \begin{cases} 
\mathbf{d}_i^{(r)} & \text{if } |\mathbf{d}_i^{(r)}| > 0 \\
\mathbf{d}_i^{(o)} + \mathbf{d}_i^{(a)} & \text{otherwise}
\end{cases}$$

Velocity updates with angular noise and turning rate constraint $\theta_{max}$:

$$\mathbf{v}_i(t + \Delta t) = v_0 \cdot \text{rotate}(\hat{\mathbf{v}}_i, \mathbf{d}_i, \theta_{max}, \sigma)$$

Position updates with periodic boundaries:

$$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i(t + \Delta t) \cdot \Delta t$$

### Key Parameters

| Parameter | Symbol | Description | Typical Range |
|:---------:|:------:|:------------|:-------------:|
| N | $N$ | Number of fish | 50-1000 |
| L | $L$ | Tank/ocean volume size | 10-100 |
| v0 | $v_0$ | Swimming speed | 0.5-3.0 |
| r_r | $r_r$ | Repulsion zone radius | 0.5-2.0 |
| r_o | $r_o$ | Orientation zone radius | 2.0-10.0 |
| r_a | $r_a$ | Attraction zone radius | 5.0-20.0 |
| θ_max | $\theta_{max}$ | Maximum turning angle | 0.1-0.8 rad |
| σ | $\sigma$ | Angular noise | 0.0-0.5 rad |
| blind_angle | $\alpha$ | Rear blind angle | 0-90° |

### Collective States

The Couzin model produces four distinct collective states:

| State | Description | Characteristics |
|:-----:|:------------|:----------------|
| **Swarm** | Disordered aggregation | Low polarization, high density |
| **Torus** | Rotating mill | Circular motion, medium polarization |
| **Dynamic Parallel** | Aligned but fluctuating | High polarization, direction changes |
| **Highly Parallel** | Stable aligned motion | Very high polarization, stable direction |

### Order Parameters

**Polarization** (alignment measure):
$$P = \frac{1}{N} \left| \sum_{i=1}^{N} \hat{\mathbf{v}}_i \right|$$

**Rotation** (milling measure):
$$M = \frac{1}{N} \left| \sum_{i=1}^{N} \hat{\mathbf{v}}_i \times \hat{\mathbf{r}}_i^c \right|$$

where $\hat{\mathbf{r}}_i^c$ is the unit vector from fish $i$ to the group centroid.

## Oceanic Entropy Metrics

This library implements specialized information-theoretic measures designed for marine collective behavior:

| Metric | Description | Ecological Relevance |
|:------:|:------------|:---------------------|
| **School Cohesion Entropy** | Nearest-neighbor distance distribution | Predator defense efficiency |
| **Polarization Entropy** | Angular distribution of headings | Migration coordination |
| **Depth Stratification Entropy** | Vertical position distribution | Thermocline tracking |
| **Angular Momentum Entropy** | Rotation state distribution | Milling behavior detection |
| **Nearest Neighbor Entropy** | k-NN distance variability | Social network structure |
| **Velocity Correlation Entropy** | Spatial velocity autocorrelation | Information transfer |
| **School Shape Entropy** | Principal component ratios | Morphological adaptation |
| **Oceanic Schooling Index** | Composite weighted metric | Overall school health |

### Oceanic Schooling Index (OSI)

The composite metric specifically designed for fish school analysis:

$$\text{OSI} = w_1 H_{cohesion} + w_2 H_{polar} + w_3 H_{depth} + w_4 H_{angular} + w_5 H_{nn} + w_6 (1 - H_{vel}) + w_7 H_{shape}$$

where weights are optimized for marine schooling behavior assessment.

## Installation

**From PyPI:**
```bash
pip install dewi-kadita
```

**From source:**
```bash
git clone https://github.com/sandyherho/dewi-kadita.git
cd dewi-kadita
pip install .
```

**Development installation with Poetry:**
```bash
git clone https://github.com/sandyherho/dewi-kadita.git
cd dewi-kadita
poetry install
```

## Quick Start

**CLI:**
```bash
dewi-kadita case1          # Run swarm phase scenario
dewi-kadita case2          # Run torus/milling scenario
dewi-kadita case3          # Run dynamic parallel scenario
dewi-kadita case4          # Run highly parallel scenario
dewi-kadita --all          # Run all test cases
dewi-kadita case1 --no-entropy  # Skip entropy computation
```

**Python API:**
```python
from dewi_kadita import CouzinSystem, CouzinSolver
from dewi_kadita import compute_metrics_timeseries

# Create system with 150 fish
system = CouzinSystem(
    n_fish=150,
    box_size=25.0,
    speed=1.0,
    r_repulsion=1.0,
    r_orientation=5.0,
    r_attraction=10.0,
    max_turn=0.3,
    noise=0.1,
    blind_angle=30.0
)

# Initialize solver with Numba acceleration
solver = CouzinSolver(dt=0.1, use_numba=True)

# Run simulation for 1000 steps
result = solver.solve(system, n_steps=1000)

# Compute oceanic entropy metrics
metrics = compute_metrics_timeseries(
    result['positions'],
    result['velocities'],
    result['time'],
    system.box_size
)

print(f"Final polarization: {result['final_polarization']:.4f}")
print(f"Final rotation: {result['final_rotation']:.4f}")
print(f"Final OSI: {metrics['oceanic_schooling_index_final']:.4f}")
```

## Features

- **High-performance**: Numba JIT compilation for 10-100x speedup
- **3D simulation**: Full three-dimensional fish schooling dynamics
- **Couzin zones**: Proper implementation of repulsion, orientation, attraction zones
- **Blind angle**: Realistic rear visual field limitation
- **Maximum turning rate**: Biologically realistic angular velocity constraint
- **Oceanic entropy metrics**: Specialized measures for marine collective behavior
- **Multiple output formats**: CSV, NetCDF (CF-compliant), PNG, GIF
- **Stunning oceanic visualization**: Deep sea aesthetic with bioluminescent effects
- **Configurable scenarios**: Text-based configuration files

## Output Files

The library generates:

- **CSV files**: 
  - `*_order_parameters.csv` - Polarization and rotation time series
  - `*_entropy_timeseries.csv` - All oceanic entropy measures over time
  - `*_entropy_summary.csv` - Statistics (mean, std, min, max, final)
  - `*_final_state.csv` - Final fish positions and velocities
- **NetCDF**: Full trajectory data with all metrics and CF-compliant metadata
- **PNG**: Static summary plots with oceanic entropy visualization
- **GIF**: Animated 3D visualization with underwater aesthetic


## Dependencies

- **numpy** >= 1.20.0
- **scipy** >= 1.7.0
- **matplotlib** >= 3.3.0
- **pandas** >= 1.3.0
- **netCDF4** >= 1.5.0
- **numba** >= 0.53.0
- **Pillow** >= 8.0.0
- **tqdm** >= 4.60.0


## License

MIT 2026 © [Sandy H. S. Herho](mailto:sandy.herho@email.ucr.edu), [Iwan P. Anwar](mailto:iwanpanwar@itb.ac.id), [Faruq Khadami](mailto:fkhadami@itb.ac.id), [Alfita P. Handayani](mailto:alfita@itb.ac.id), [Karina A. Sujatmiko](mailto:karinaas@itb.ac.id), Kamaluddin Kasim, [Rusmawan Suwarman](mailto:rusmawan@itb.ac.id), and [Dasapta E. Irawan](mailto:dasaptaerwin@itb.ac.id)

## Citation

```bibtex
@software{herhoEtAl2026_dewi_kadita,
  title   = {{\texttt{dewi-kadita}: A Python library for 3D idealized Couzin model fish schooling simulation}},
  author  = {Herho, Sandy H. S. and Anwar, Iwan P. and Khadami, Faruq and Handayani, Alfita P. and Sujatmiko, Karina A. and Kasim, Kamaluddin and Suwarman, Rusmawan and Irawan, Dasapta E.},
  year    = {2026},
  url     = {https://github.com/sandyherho/dewi-kadita}
}
```
