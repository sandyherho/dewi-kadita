"""
Stunning Oceanic Visualization for 3D Couzin Fish Schooling Model.

Features deep sea aesthetic with bioluminescent effects, caustic light
patterns, and dramatic underwater atmosphere.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

from PIL import Image, ImageFilter, ImageEnhance
import io


class Animator:
    """
    Create stunning oceanic visualizations for fish schooling
    with bioluminescent deep sea aesthetic.
    """
    
    # Deep Ocean Theme Color Palette
    COLOR_ABYSS = '#020B1A'           # Deepest ocean background
    COLOR_DEEP_BLUE = '#051530'        # Deep water
    COLOR_TWILIGHT = '#0A2647'         # Twilight zone
    COLOR_MESOPELAGIC = '#144272'      # Mesopelagic blue
    
    COLOR_BIOLUM_CYAN = '#00F5FF'      # Bioluminescent cyan
    COLOR_BIOLUM_GREEN = '#39FF14'     # Bioluminescent green
    COLOR_BIOLUM_BLUE = '#00D4FF'      # Electric blue
    COLOR_BIOLUM_PURPLE = '#BF40BF'    # Deep purple glow
    
    COLOR_FISH_SILVER = '#C0C0C0'      # Fish silver
    COLOR_FISH_BLUE = '#4169E1'        # Fish blue
    
    COLOR_SUNLIGHT = '#FFE5B4'         # Filtered sunlight
    COLOR_CAUSTICS = '#87CEEB'         # Caustic light patterns
    
    COLOR_TEXT = '#B8D4E3'             # Soft blue text
    COLOR_TITLE = '#E8F4F8'            # Bright title
    COLOR_ACCENT = '#00CED1'           # Dark cyan accent
    COLOR_GRID = '#1A3A5C'             # Subtle grid
    
    def __init__(self, fps: int = 30, dpi: int = 150):
        """
        Initialize animator with oceanic theme.
        
        Args:
            fps: Frames per second for animations
            dpi: Resolution for output images
        """
        self.fps = fps
        self.dpi = dpi
        self._setup_oceanic_style()
        self._create_ocean_colormap()
    
    def _setup_oceanic_style(self):
        """Setup matplotlib deep ocean styling."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': self.COLOR_ABYSS,
            'axes.facecolor': self.COLOR_DEEP_BLUE,
            'axes.edgecolor': self.COLOR_GRID,
            'axes.labelcolor': self.COLOR_TEXT,
            'axes.titlecolor': self.COLOR_TITLE,
            'xtick.color': self.COLOR_TEXT,
            'ytick.color': self.COLOR_TEXT,
            'text.color': self.COLOR_TEXT,
            'grid.color': self.COLOR_GRID,
            'grid.alpha': 0.3,
            'legend.facecolor': self.COLOR_TWILIGHT,
            'legend.edgecolor': self.COLOR_GRID,
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 13,
            'axes.titlesize': 15,
            'lines.antialiased': True,
            'figure.autolayout': False,
        })
    
    def _create_ocean_colormap(self):
        """Create custom ocean depth colormap."""
        colors = [
            (0.0, self.COLOR_ABYSS),
            (0.2, self.COLOR_DEEP_BLUE),
            (0.4, self.COLOR_TWILIGHT),
            (0.6, self.COLOR_MESOPELAGIC),
            (0.8, '#2980B9'),
            (1.0, '#5DADE2')
        ]
        
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
        
        cmap_colors = [(pos, hex_to_rgb(color)) for pos, color in colors]
        
        # Create colormap
        cdict = {'red': [], 'green': [], 'blue': []}
        for pos, (r, g, b) in cmap_colors:
            cdict['red'].append((pos, r, r))
            cdict['green'].append((pos, g, g))
            cdict['blue'].append((pos, b, b))
        
        self.ocean_cmap = LinearSegmentedColormap('OceanDepth', cdict)
    
    def _add_underwater_glow(self, image: Image.Image, intensity: float = 1.3) -> Image.Image:
        """Add ethereal underwater glow effect."""
        # Convert to RGB if image has alpha channel (RGBA)
        if image.mode == 'RGBA':
            # Preserve alpha channel
            alpha = image.split()[3]
            rgb_image = image.convert('RGB')
        else:
            alpha = None
            rgb_image = image.convert('RGB')
        
        # Create glow layer
        glow = rgb_image.filter(ImageFilter.GaussianBlur(radius=4))
        
        # Enhance blue channel for underwater feel
        r, g, b = glow.split()
        b = ImageEnhance.Brightness(b).enhance(1.2)
        glow = Image.merge('RGB', (r, g, b))
        
        # Blend with original
        result = Image.blend(rgb_image, glow, alpha=0.2)
        
        # Restore alpha channel if it existed
        if alpha is not None:
            result = result.convert('RGBA')
            result.putalpha(alpha)
        
        return result
    
    def _add_caustics_effect(self, ax, box_size: float, time_phase: float):
        """Add subtle caustic light patterns (simulated)."""
        # This creates a subtle wave-like pattern suggestion
        # Real caustics would require ray tracing
        pass  # Simplified for performance
    
    def create_static_plot(
        self,
        result: Dict[str, Any],
        filepath: str,
        title: str = "3D Couzin Fish School",
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Create stunning oceanic static plot with metrics visualization.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Plot title
            metrics: Optional oceanic entropy metrics dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Layout based on metrics availability
        if metrics is not None:
            fig = plt.figure(figsize=(20, 12), facecolor=self.COLOR_ABYSS)
            fig.suptitle(f'{title}\nOceanic Fish Schooling Dynamics',
                        fontsize=20, fontweight='bold', color=self.COLOR_TITLE, y=0.98)
            
            # Create grid: 2x3
            ax1 = fig.add_subplot(231, facecolor=self.COLOR_DEEP_BLUE)
            ax2 = fig.add_subplot(232, projection='3d', facecolor=self.COLOR_ABYSS)
            ax3 = fig.add_subplot(233, facecolor=self.COLOR_DEEP_BLUE)
            ax4 = fig.add_subplot(234, facecolor=self.COLOR_DEEP_BLUE)
            ax5 = fig.add_subplot(235, facecolor=self.COLOR_DEEP_BLUE)
            ax6 = fig.add_subplot(236, facecolor=self.COLOR_DEEP_BLUE)
        else:
            fig = plt.figure(figsize=(16, 7), facecolor=self.COLOR_ABYSS)
            fig.suptitle(f'{title}',
                        fontsize=20, fontweight='bold', color=self.COLOR_TITLE, y=0.98)
            ax1 = fig.add_subplot(121, facecolor=self.COLOR_DEEP_BLUE)
            ax2 = fig.add_subplot(122, projection='3d', facecolor=self.COLOR_ABYSS)
        
        time = result['time']
        polarization = result['polarization']
        rotation = result['rotation']
        
        # Plot 1: Order parameters time series
        ax1.plot(time, polarization, color=self.COLOR_BIOLUM_CYAN, lw=2.5, 
                alpha=0.9, label='Polarization (P)')
        ax1.fill_between(time, 0, polarization, color=self.COLOR_BIOLUM_CYAN, alpha=0.15)
        
        ax1.plot(time, rotation, color=self.COLOR_BIOLUM_GREEN, lw=2.5, 
                alpha=0.9, label='Rotation (M)')
        ax1.fill_between(time, 0, rotation, color=self.COLOR_BIOLUM_GREEN, alpha=0.15)
        
        ax1.axhline(y=result['mean_polarization'], color=self.COLOR_BIOLUM_CYAN,
                   linestyle='--', lw=1.5, alpha=0.6)
        ax1.axhline(y=result['mean_rotation'], color=self.COLOR_BIOLUM_GREEN,
                   linestyle='--', lw=1.5, alpha=0.6)
        
        ax1.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Order Parameter', fontsize=14, fontweight='bold')
        ax1.set_title('School Coordination', fontsize=14, fontweight='bold', 
                     color=self.COLOR_TITLE, pad=10)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(time[0], time[-1])
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.8)
        ax1.grid(True, alpha=0.3, color=self.COLOR_GRID)
        
        for spine in ax1.spines.values():
            spine.set_color(self.COLOR_GRID)
        
        # Plot 2: 3D fish school visualization
        pos = result['positions'][-1]
        vel = result['velocities'][-1]
        L = result['system'].box_size
        
        # Color by depth (z-coordinate)
        depths = pos[:, 2]
        depth_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-10)
        colors = plt.cm.cool(depth_norm)
        
        # Draw fish as quivers with bioluminescent colors
        ax2.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                  vel[:, 0], vel[:, 1], vel[:, 2],
                  length=1.2, normalize=True, 
                  color=self.COLOR_BIOLUM_CYAN, alpha=0.8,
                  arrow_length_ratio=0.3, linewidth=0.8)
        
        # Add scatter points for fish bodies
        ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=depth_norm, cmap='cool', s=15, alpha=0.6)
        
        ax2.set_xlim(0, L)
        ax2.set_ylim(0, L)
        ax2.set_zlim(0, L)
        ax2.set_xlabel('X', fontsize=12, fontweight='bold', color=self.COLOR_TEXT)
        ax2.set_ylabel('Y', fontsize=12, fontweight='bold', color=self.COLOR_TEXT)
        ax2.set_zlabel('Depth', fontsize=12, fontweight='bold', color=self.COLOR_TEXT)
        ax2.set_title(f'Final State (P={result["final_polarization"]:.3f}, M={result["final_rotation"]:.3f})',
                     fontsize=14, fontweight='bold', color=self.COLOR_TITLE, pad=10)
        
        # Style 3D panes for underwater look
        ax2.xaxis.pane.fill = True
        ax2.yaxis.pane.fill = True
        ax2.zaxis.pane.fill = True
        ax2.xaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
        ax2.yaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
        ax2.zaxis.pane.set_facecolor(self.COLOR_MESOPELAGIC)
        ax2.xaxis.pane.set_alpha(0.8)
        ax2.yaxis.pane.set_alpha(0.8)
        ax2.zaxis.pane.set_alpha(0.6)
        ax2.tick_params(colors=self.COLOR_TEXT, labelsize=9)
        
        # Additional plots if metrics available
        if metrics is not None:
            # Plot 3: Oceanic Schooling Index
            ax3.plot(metrics['time'], metrics['oceanic_schooling_index'],
                    color=self.COLOR_BIOLUM_BLUE, lw=2.5, alpha=0.9)
            ax3.fill_between(metrics['time'], 0, metrics['oceanic_schooling_index'],
                            color=self.COLOR_BIOLUM_BLUE, alpha=0.2)
            ax3.axhline(y=metrics['oceanic_schooling_index_mean'], 
                       color=self.COLOR_ACCENT, linestyle='--', lw=1.5, alpha=0.7,
                       label=f"Mean: {metrics['oceanic_schooling_index_mean']:.3f}")
            
            ax3.set_xlabel('Time', fontsize=14, fontweight='bold')
            ax3.set_ylabel('OSI', fontsize=14, fontweight='bold')
            ax3.set_title('Oceanic Schooling Index', fontsize=14, fontweight='bold',
                         color=self.COLOR_TITLE, pad=10)
            ax3.set_ylim(0, 1.05)
            ax3.legend(loc='upper right', fontsize=10)
            ax3.grid(True, alpha=0.3)
            for spine in ax3.spines.values():
                spine.set_color(self.COLOR_GRID)
            
            # Plot 4: Key entropy measures over time
            ax4.plot(metrics['time'], metrics['polarization_entropy'],
                    color=self.COLOR_BIOLUM_CYAN, lw=2, alpha=0.9, label='Polarization H')
            ax4.plot(metrics['time'], metrics['school_cohesion_entropy'],
                    color=self.COLOR_BIOLUM_GREEN, lw=2, alpha=0.9, label='Cohesion H')
            ax4.plot(metrics['time'], metrics['angular_momentum_entropy'],
                    color=self.COLOR_BIOLUM_PURPLE, lw=2, alpha=0.9, label='Angular H')
            
            ax4.set_xlabel('Time', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Entropy', fontsize=14, fontweight='bold')
            ax4.set_title('Key Entropy Measures', fontsize=14, fontweight='bold',
                         color=self.COLOR_TITLE, pad=10)
            ax4.set_ylim(0, 1.05)
            ax4.legend(loc='upper right', fontsize=9)
            ax4.grid(True, alpha=0.3)
            for spine in ax4.spines.values():
                spine.set_color(self.COLOR_GRID)
            
            # Plot 5: School spread over time
            ax5.plot(metrics['time'], metrics['spread'],
                    color=self.COLOR_SUNLIGHT, lw=2.5, alpha=0.9)
            ax5.fill_between(metrics['time'], 0, metrics['spread'],
                            color=self.COLOR_SUNLIGHT, alpha=0.15)
            
            ax5.set_xlabel('Time', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Spread (RMS)', fontsize=14, fontweight='bold')
            ax5.set_title('School Spread', fontsize=14, fontweight='bold',
                         color=self.COLOR_TITLE, pad=10)
            ax5.grid(True, alpha=0.3)
            for spine in ax5.spines.values():
                spine.set_color(self.COLOR_GRID)
            
            # Plot 6: Final entropy summary (radar-like bar chart)
            entropy_names = ['Cohesion', 'Polar.', 'Depth', 'Angular', 'k-NN', 'Vel.Corr', 'Shape']
            entropy_values = [
                metrics['school_cohesion_entropy_final'],
                metrics['polarization_entropy_final'],
                metrics['depth_stratification_entropy_final'],
                metrics['angular_momentum_entropy_final'],
                metrics['nearest_neighbor_entropy_final'],
                metrics['velocity_correlation_entropy_final'],
                metrics['school_shape_entropy_final']
            ]
            
            colors = [
                self.COLOR_BIOLUM_CYAN,
                self.COLOR_BIOLUM_GREEN,
                self.COLOR_BIOLUM_BLUE,
                self.COLOR_BIOLUM_PURPLE,
                self.COLOR_ACCENT,
                self.COLOR_SUNLIGHT,
                self.COLOR_CAUSTICS
            ]
            
            bars = ax6.bar(entropy_names, entropy_values, color=colors, alpha=0.85,
                          edgecolor='white', linewidth=0.5)
            
            ax6.set_ylabel('Entropy', fontsize=14, fontweight='bold')
            ax6.set_title('Final Oceanic Entropy Profile', fontsize=14, fontweight='bold',
                         color=self.COLOR_TITLE, pad=10)
            ax6.set_ylim(0, 1.1)
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, entropy_values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9,
                        color=self.COLOR_TEXT, fontweight='bold')
            
            ax6.tick_params(axis='x', rotation=45, labelsize=9)
            for spine in ax6.spines.values():
                spine.set_color(self.COLOR_GRID)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filepath, dpi=self.dpi, facecolor=self.COLOR_ABYSS,
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
    
    def create_animation(self, result: Dict[str, Any], filepath: str,
                         title: str = "3D Fish School", skip: int = None,
                         duration_seconds: float = 18.0):
        """
        Create stunning animated 3D visualization with oceanic theme
        and dramatic underwater camera movement.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Animation title
            skip: Frame skip (auto-calculated if None)
            duration_seconds: Target duration in seconds
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        positions = result['positions']
        velocities = result['velocities']
        time = result['time']
        polarization = result['polarization']
        rotation = result['rotation']
        system = result['system']
        L = system.box_size
        N = system.n_fish
        
        n_frames_total = len(time)
        
        # Calculate frame skip for target duration
        target_frames = int(duration_seconds * self.fps)
        target_frames = min(target_frames, 360)
        skip = max(1, n_frames_total // target_frames) if skip is None else skip
        
        frame_indices = np.arange(0, n_frames_total, skip)
        n_frames = len(frame_indices)
        
        print(f"      Generating {n_frames} oceanic frames...")
        
        anim_dpi = 120
        frames = []
        
        # Dramatic underwater camera motion
        t_norm = np.linspace(0, 1, n_frames)
        
        # Full 360° rotation with smooth acceleration
        azim_start, azim_end = 30, 390
        ease = lambda t: t * t * (3 - 2 * t)  # Smooth step
        azims = azim_start + (azim_end - azim_start) * ease(t_norm)
        
        # Elevation: diving and rising motion
        elevs = 20 + 20 * np.sin(2 * np.pi * t_norm)  # 0° to 40°
        
        for i, idx in enumerate(tqdm(frame_indices,
                                      desc="      Rendering",
                                      ncols=70,
                                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}')):
            
            fig = plt.figure(figsize=(14, 11), facecolor=self.COLOR_ABYSS, dpi=anim_dpi)
            ax = fig.add_subplot(111, projection='3d', facecolor=self.COLOR_ABYSS)
            
            pos = positions[idx]
            vel = velocities[idx]
            
            ax.view_init(elev=elevs[i], azim=azims[i])
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_zlim(0, L)
            
            # Color fish by depth for underwater depth perception
            depths = pos[:, 2]
            depth_norm = (depths - 0) / L
            
            # Draw fish with bioluminescent trail effect
            ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                     vel[:, 0], vel[:, 1], vel[:, 2],
                     length=1.0, normalize=True,
                     color=self.COLOR_BIOLUM_CYAN, alpha=0.85,
                     linewidth=0.9, arrow_length_ratio=0.25)
            
            # Fish body scatter with depth coloring
            scatter_colors = plt.cm.cool(1 - depth_norm)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=scatter_colors, s=12, alpha=0.5)
            
            # Clean axis labels
            ax.set_xlabel('X', fontsize=11, fontweight='bold', 
                         color=self.COLOR_TEXT, labelpad=8)
            ax.set_ylabel('Y', fontsize=11, fontweight='bold', 
                         color=self.COLOR_TEXT, labelpad=8)
            ax.set_zlabel('Depth', fontsize=11, fontweight='bold', 
                         color=self.COLOR_TEXT, labelpad=8)
            
            # Title with ocean emoji
            ax.set_title(f'{title}',
                        fontsize=18, fontweight='bold', 
                        color=self.COLOR_TITLE, pad=25)
            
            # Style 3D panes for deep ocean look
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
            ax.yaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
            ax.zaxis.pane.set_facecolor(self.COLOR_MESOPELAGIC)
            ax.xaxis.pane.set_edgecolor(self.COLOR_GRID)
            ax.yaxis.pane.set_edgecolor(self.COLOR_GRID)
            ax.zaxis.pane.set_edgecolor(self.COLOR_GRID)
            ax.xaxis.pane.set_alpha(0.85)
            ax.yaxis.pane.set_alpha(0.85)
            ax.zaxis.pane.set_alpha(0.7)
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=9)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            
            # Parameter info box (top-left) with oceanic styling
            param_text = (
                f'N = {N}\n'
                f'L = {L:.1f}\n'
                f'r_r = {system.r_repulsion:.1f}\n'
                f'r_o = {system.r_orientation:.1f}\n'
                f'r_a = {system.r_attraction:.1f}\n'
                f'θ_max = {system.max_turn:.2f}'
            )
            ax.text2D(0.02, 0.98, param_text,
                     transform=ax.transAxes, fontsize=10, fontweight='bold',
                     color=self.COLOR_TEXT, ha='left', va='top',
                     alpha=0.9, family='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', 
                              facecolor=self.COLOR_TWILIGHT,
                              edgecolor=self.COLOR_GRID, alpha=0.9))
            
            # Time and order parameters (bottom center) with bioluminescent glow
            info_text = f't = {time[idx]:.1f}  │  P = {polarization[idx]:.3f}  │  M = {rotation[idx]:.3f}'
            ax.text2D(0.5, 0.02, info_text,
                     transform=ax.transAxes, fontsize=13, fontweight='bold',
                     color=self.COLOR_BIOLUM_CYAN, ha='center', va='bottom',
                     alpha=0.95)
            
            fig.tight_layout()
            
            # Render to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=anim_dpi,
                       facecolor=self.COLOR_ABYSS, edgecolor='none')
            buf.seek(0)
            frame_img = Image.open(buf).copy()
            
            # Add underwater glow effect
            frame_img = self._add_underwater_glow(frame_img, intensity=1.2)
            
            frames.append(frame_img)
            buf.close()
            plt.close(fig)
        
        # Smooth playback
        frame_duration_ms = int(1000 / self.fps)
        
        print(f"      Saving oceanic GIF ({n_frames} frames)...")
        
        frames[0].save(
            str(filepath),
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=True,
            quality=85
        )
        
        print(f"      Done! Saved to {filepath.name}")
