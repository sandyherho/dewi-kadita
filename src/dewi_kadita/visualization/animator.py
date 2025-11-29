"""
Oceanic Visualization for 3D Couzin Fish Schooling Model.

Features refined deep sea aesthetic with smooth bioluminescent effects,
graceful camera movement, and sophisticated underwater atmosphere.
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
    Create elegant oceanic visualizations for fish schooling
    with refined deep sea aesthetic and graceful animations.
    """
    
    # Refined Deep Ocean Color Palette
    COLOR_ABYSS = '#010A14'           # Deepest ocean background
    COLOR_DEEP_BLUE = '#041525'        # Deep water
    COLOR_TWILIGHT = '#082540'         # Twilight zone
    COLOR_MESOPELAGIC = '#0F3460'      # Mesopelagic blue
    COLOR_MIDWATER = '#1A4A7A'         # Mid-water blue
    
    # Elegant bioluminescent tones (softer, more refined)
    COLOR_BIOLUM_CYAN = '#4DD0E1'      # Soft bioluminescent cyan
    COLOR_BIOLUM_TEAL = '#26A69A'      # Elegant teal
    COLOR_BIOLUM_BLUE = '#42A5F5'      # Soft electric blue
    COLOR_BIOLUM_AQUA = '#00ACC1'      # Aquamarine
    COLOR_BIOLUM_PURPLE = '#7E57C2'    # Soft purple glow
    
    # Fish coloring
    COLOR_FISH_PRIMARY = '#80DEEA'     # Primary fish color
    COLOR_FISH_SECONDARY = '#B2EBF2'   # Secondary highlights
    
    # Ambient lighting
    COLOR_SUNBEAM = '#E8F5E9'          # Filtered sunlight
    COLOR_CAUSTICS = '#B3E5FC'         # Caustic light patterns
    
    # UI Elements
    COLOR_TEXT = '#B0BEC5'             # Soft gray-blue text
    COLOR_TITLE = '#ECEFF1'            # Clean white title
    COLOR_ACCENT = '#4DB6AC'           # Teal accent
    COLOR_GRID = '#1E3A5F'             # Subtle grid
    COLOR_PANEL = '#0D2137'            # Info panel background
    
    def __init__(self, fps: int = 30, dpi: int = 150):
        """
        Initialize animator with elegant oceanic theme.
        
        Args:
            fps: Frames per second for animations
            dpi: Resolution for output images
        """
        self.fps = fps
        self.dpi = dpi
        self._setup_elegant_style()
        self._create_ocean_colormap()
    
    def _setup_elegant_style(self):
        """Setup matplotlib with refined ocean styling."""
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
            'grid.alpha': 0.25,
            'legend.facecolor': self.COLOR_PANEL,
            'legend.edgecolor': self.COLOR_GRID,
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'lines.antialiased': True,
            'figure.autolayout': False,
        })
    
    def _create_ocean_colormap(self):
        """Create custom ocean depth colormap with smooth gradients."""
        colors = [
            (0.0, self.COLOR_ABYSS),
            (0.15, self.COLOR_DEEP_BLUE),
            (0.35, self.COLOR_TWILIGHT),
            (0.55, self.COLOR_MESOPELAGIC),
            (0.75, self.COLOR_MIDWATER),
            (1.0, '#2E7D9A')
        ]
        
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
        
        cmap_colors = [(pos, hex_to_rgb(color)) for pos, color in colors]
        
        cdict = {'red': [], 'green': [], 'blue': []}
        for pos, (r, g, b) in cmap_colors:
            cdict['red'].append((pos, r, r))
            cdict['green'].append((pos, g, g))
            cdict['blue'].append((pos, b, b))
        
        self.ocean_cmap = LinearSegmentedColormap('OceanDepth', cdict)
    
    def _add_underwater_glow(self, image: Image.Image, intensity: float = 1.2) -> Image.Image:
        """Add subtle ethereal underwater glow effect."""
        # Handle RGBA images properly
        if image.mode == 'RGBA':
            alpha = image.split()[3]
            rgb_image = image.convert('RGB')
        else:
            alpha = None
            rgb_image = image.convert('RGB')
        
        # Create soft glow layer
        glow = rgb_image.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Subtle blue channel enhancement for underwater atmosphere
        r, g, b = glow.split()
        b = ImageEnhance.Brightness(b).enhance(1.1)
        g = ImageEnhance.Brightness(g).enhance(1.05)
        glow = Image.merge('RGB', (r, g, b))
        
        # Gentle blend with original
        result = Image.blend(rgb_image, glow, alpha=0.15)
        
        # Restore alpha channel if present
        if alpha is not None:
            result = result.convert('RGBA')
            result.putalpha(alpha)
        
        return result
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Smooth cubic easing function for graceful motion."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def _ease_in_out_sine(self, t: float) -> float:
        """Sinusoidal easing for natural, flowing motion."""
        return -(np.cos(np.pi * t) - 1) / 2
    
    def create_static_plot(
        self,
        result: Dict[str, Any],
        filepath: str,
        title: str = "3D Couzin Fish School",
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Create elegant static visualization with metrics.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Plot title
            metrics: Optional oceanic entropy metrics dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean title (remove any emojis if present)
        clean_title = ''.join(c for c in title if ord(c) < 128)
        
        if metrics is not None:
            fig = plt.figure(figsize=(20, 12), facecolor=self.COLOR_ABYSS)
            fig.suptitle(f'{clean_title}\nCollective Motion Dynamics',
                        fontsize=18, fontweight='normal', color=self.COLOR_TITLE, 
                        y=0.97, fontfamily='sans-serif')
            
            ax1 = fig.add_subplot(231, facecolor=self.COLOR_DEEP_BLUE)
            ax2 = fig.add_subplot(232, projection='3d', facecolor=self.COLOR_ABYSS)
            ax3 = fig.add_subplot(233, facecolor=self.COLOR_DEEP_BLUE)
            ax4 = fig.add_subplot(234, facecolor=self.COLOR_DEEP_BLUE)
            ax5 = fig.add_subplot(235, facecolor=self.COLOR_DEEP_BLUE)
            ax6 = fig.add_subplot(236, facecolor=self.COLOR_DEEP_BLUE)
        else:
            fig = plt.figure(figsize=(16, 7), facecolor=self.COLOR_ABYSS)
            fig.suptitle(clean_title,
                        fontsize=18, fontweight='normal', color=self.COLOR_TITLE, 
                        y=0.97, fontfamily='sans-serif')
            ax1 = fig.add_subplot(121, facecolor=self.COLOR_DEEP_BLUE)
            ax2 = fig.add_subplot(122, projection='3d', facecolor=self.COLOR_ABYSS)
        
        time = result['time']
        polarization = result['polarization']
        rotation = result['rotation']
        
        # Plot 1: Order parameters with elegant styling
        ax1.plot(time, polarization, color=self.COLOR_BIOLUM_CYAN, lw=2, 
                alpha=0.9, label='Polarization (P)')
        ax1.fill_between(time, 0, polarization, color=self.COLOR_BIOLUM_CYAN, alpha=0.1)
        
        ax1.plot(time, rotation, color=self.COLOR_BIOLUM_TEAL, lw=2, 
                alpha=0.9, label='Rotation (M)')
        ax1.fill_between(time, 0, rotation, color=self.COLOR_BIOLUM_TEAL, alpha=0.1)
        
        ax1.axhline(y=result['mean_polarization'], color=self.COLOR_BIOLUM_CYAN,
                   linestyle='--', lw=1, alpha=0.5)
        ax1.axhline(y=result['mean_rotation'], color=self.COLOR_BIOLUM_TEAL,
                   linestyle='--', lw=1, alpha=0.5)
        
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Order Parameter', fontsize=12)
        ax1.set_title('School Coordination', fontsize=13, color=self.COLOR_TITLE, pad=10)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(time[0], time[-1])
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.7)
        ax1.grid(True, alpha=0.2, color=self.COLOR_GRID)
        
        for spine in ax1.spines.values():
            spine.set_color(self.COLOR_GRID)
            spine.set_linewidth(0.5)
        
        # Plot 2: 3D visualization with refined aesthetics
        pos = result['positions'][-1]
        vel = result['velocities'][-1]
        L = result['system'].box_size
        
        depths = pos[:, 2]
        depth_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-10)
        
        # Elegant quiver visualization
        ax2.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                  vel[:, 0], vel[:, 1], vel[:, 2],
                  length=1.0, normalize=True, 
                  color=self.COLOR_BIOLUM_CYAN, alpha=0.75,
                  arrow_length_ratio=0.3, linewidth=0.6)
        
        # Subtle scatter for fish positions
        ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=depth_norm, cmap='cool', s=12, alpha=0.5)
        
        ax2.set_xlim(0, L)
        ax2.set_ylim(0, L)
        ax2.set_zlim(0, L)
        ax2.set_xlabel('X', fontsize=11, color=self.COLOR_TEXT)
        ax2.set_ylabel('Y', fontsize=11, color=self.COLOR_TEXT)
        ax2.set_zlabel('Z', fontsize=11, color=self.COLOR_TEXT)
        ax2.set_title(f'Final State  (P = {result["final_polarization"]:.3f},  M = {result["final_rotation"]:.3f})',
                     fontsize=13, color=self.COLOR_TITLE, pad=10)
        
        # Refined 3D pane styling
        ax2.xaxis.pane.fill = True
        ax2.yaxis.pane.fill = True
        ax2.zaxis.pane.fill = True
        ax2.xaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
        ax2.yaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
        ax2.zaxis.pane.set_facecolor(self.COLOR_MESOPELAGIC)
        ax2.xaxis.pane.set_alpha(0.7)
        ax2.yaxis.pane.set_alpha(0.7)
        ax2.zaxis.pane.set_alpha(0.5)
        ax2.tick_params(colors=self.COLOR_TEXT, labelsize=8)
        
        if metrics is not None:
            # Plot 3: Oceanic Schooling Index
            ax3.plot(metrics['time'], metrics['oceanic_schooling_index'],
                    color=self.COLOR_BIOLUM_BLUE, lw=2, alpha=0.9)
            ax3.fill_between(metrics['time'], 0, metrics['oceanic_schooling_index'],
                            color=self.COLOR_BIOLUM_BLUE, alpha=0.15)
            ax3.axhline(y=metrics['oceanic_schooling_index_mean'], 
                       color=self.COLOR_ACCENT, linestyle='--', lw=1, alpha=0.6,
                       label=f"Mean: {metrics['oceanic_schooling_index_mean']:.3f}")
            
            ax3.set_xlabel('Time', fontsize=12)
            ax3.set_ylabel('OSI', fontsize=12)
            ax3.set_title('Oceanic Schooling Index', fontsize=13, color=self.COLOR_TITLE, pad=10)
            ax3.set_ylim(0, 1.05)
            ax3.legend(loc='upper right', fontsize=9, framealpha=0.7)
            ax3.grid(True, alpha=0.2)
            for spine in ax3.spines.values():
                spine.set_color(self.COLOR_GRID)
                spine.set_linewidth(0.5)
            
            # Plot 4: Key entropy measures
            ax4.plot(metrics['time'], metrics['polarization_entropy'],
                    color=self.COLOR_BIOLUM_CYAN, lw=1.8, alpha=0.9, label='Polarization H')
            ax4.plot(metrics['time'], metrics['school_cohesion_entropy'],
                    color=self.COLOR_BIOLUM_TEAL, lw=1.8, alpha=0.9, label='Cohesion H')
            ax4.plot(metrics['time'], metrics['angular_momentum_entropy'],
                    color=self.COLOR_BIOLUM_PURPLE, lw=1.8, alpha=0.9, label='Angular H')
            
            ax4.set_xlabel('Time', fontsize=12)
            ax4.set_ylabel('Entropy', fontsize=12)
            ax4.set_title('Key Entropy Measures', fontsize=13, color=self.COLOR_TITLE, pad=10)
            ax4.set_ylim(0, 1.05)
            ax4.legend(loc='upper right', fontsize=9, framealpha=0.7)
            ax4.grid(True, alpha=0.2)
            for spine in ax4.spines.values():
                spine.set_color(self.COLOR_GRID)
                spine.set_linewidth(0.5)
            
            # Plot 5: School spread
            ax5.plot(metrics['time'], metrics['spread'],
                    color=self.COLOR_SUNBEAM, lw=2, alpha=0.9)
            ax5.fill_between(metrics['time'], 0, metrics['spread'],
                            color=self.COLOR_SUNBEAM, alpha=0.1)
            
            ax5.set_xlabel('Time', fontsize=12)
            ax5.set_ylabel('Spread (RMS)', fontsize=12)
            ax5.set_title('School Spread', fontsize=13, color=self.COLOR_TITLE, pad=10)
            ax5.grid(True, alpha=0.2)
            for spine in ax5.spines.values():
                spine.set_color(self.COLOR_GRID)
                spine.set_linewidth(0.5)
            
            # Plot 6: Final entropy profile
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
                self.COLOR_BIOLUM_TEAL,
                self.COLOR_BIOLUM_BLUE,
                self.COLOR_BIOLUM_PURPLE,
                self.COLOR_ACCENT,
                self.COLOR_SUNBEAM,
                self.COLOR_CAUSTICS
            ]
            
            bars = ax6.bar(entropy_names, entropy_values, color=colors, alpha=0.8,
                          edgecolor='none', linewidth=0)
            
            ax6.set_ylabel('Entropy', fontsize=12)
            ax6.set_title('Final Entropy Profile', fontsize=13, color=self.COLOR_TITLE, pad=10)
            ax6.set_ylim(0, 1.1)
            ax6.grid(True, alpha=0.2, axis='y')
            
            for bar, val in zip(bars, entropy_values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                        color=self.COLOR_TEXT)
            
            ax6.tick_params(axis='x', rotation=45, labelsize=8)
            for spine in ax6.spines.values():
                spine.set_color(self.COLOR_GRID)
                spine.set_linewidth(0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filepath, dpi=self.dpi, facecolor=self.COLOR_ABYSS,
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
    
    def create_animation(self, result: Dict[str, Any], filepath: str,
                         title: str = "3D Fish School", skip: int = None,
                         duration_seconds: float = 20.0):
        """
        Create elegant animated 3D visualization with graceful camera movement.
        
        Features smooth, slow rotation with refined underwater aesthetics
        and sophisticated visual design.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Animation title (emojis will be removed)
            skip: Frame skip (auto-calculated if None)
            duration_seconds: Target duration in seconds
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean title - remove any emojis or special characters
        clean_title = ''.join(c for c in title if ord(c) < 128).strip()
        
        positions = result['positions']
        velocities = result['velocities']
        time = result['time']
        polarization = result['polarization']
        rotation = result['rotation']
        system = result['system']
        L = system.box_size
        N = system.n_fish
        
        n_frames_total = len(time)
        
        # Calculate frame parameters for smooth animation
        target_frames = int(duration_seconds * self.fps)
        target_frames = min(target_frames, 400)
        skip = max(1, n_frames_total // target_frames) if skip is None else skip
        
        frame_indices = np.arange(0, n_frames_total, skip)
        n_frames = len(frame_indices)
        
        print(f"      Generating {n_frames} frames with graceful motion...")
        
        anim_dpi = 120
        frames = []
        
        # Graceful camera motion parameters
        t_norm = np.linspace(0, 1, n_frames)
        
        # Slow, elegant 180-degree rotation (half turn for smooth loop feel)
        azim_start, azim_end = 35, 215
        
        # Apply smooth easing for graceful acceleration/deceleration
        eased_t = np.array([self._ease_in_out_sine(t) for t in t_norm])
        azims = azim_start + (azim_end - azim_start) * eased_t
        
        # Gentle, subtle elevation breathing (very slow oscillation)
        elevs = 25 + 8 * np.sin(np.pi * t_norm)  # Subtle 17° to 33° range
        
        for i, idx in enumerate(tqdm(frame_indices,
                                      desc="      Rendering",
                                      ncols=70,
                                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}')):
            
            fig = plt.figure(figsize=(14, 11), facecolor=self.COLOR_ABYSS, dpi=anim_dpi)
            ax = fig.add_subplot(111, projection='3d', facecolor=self.COLOR_ABYSS)
            
            pos = positions[idx]
            vel = velocities[idx]
            
            # Apply graceful camera position
            ax.view_init(elev=elevs[i], azim=azims[i])
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_zlim(0, L)
            
            # Depth-based coloring for visual depth perception
            depths = pos[:, 2]
            depth_norm = depths / L
            
            # Elegant fish visualization with refined colors
            ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
                     vel[:, 0], vel[:, 1], vel[:, 2],
                     length=0.9, normalize=True,
                     color=self.COLOR_FISH_PRIMARY, alpha=0.8,
                     linewidth=0.7, arrow_length_ratio=0.25)
            
            # Subtle position markers with depth gradient
            scatter_colors = plt.cm.cool(1 - depth_norm)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=scatter_colors, s=10, alpha=0.4, edgecolors='none')
            
            # Clean, minimal axis labels
            ax.set_xlabel('X', fontsize=10, color=self.COLOR_TEXT, labelpad=6)
            ax.set_ylabel('Y', fontsize=10, color=self.COLOR_TEXT, labelpad=6)
            ax.set_zlabel('Z', fontsize=10, color=self.COLOR_TEXT, labelpad=6)
            
            # Elegant title without emojis
            ax.set_title(clean_title,
                        fontsize=16, fontweight='normal', 
                        color=self.COLOR_TITLE, pad=20,
                        fontfamily='sans-serif')
            
            # Refined 3D pane styling for depth
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
            ax.yaxis.pane.set_facecolor(self.COLOR_TWILIGHT)
            ax.zaxis.pane.set_facecolor(self.COLOR_MESOPELAGIC)
            ax.xaxis.pane.set_edgecolor(self.COLOR_GRID)
            ax.yaxis.pane.set_edgecolor(self.COLOR_GRID)
            ax.zaxis.pane.set_edgecolor(self.COLOR_GRID)
            ax.xaxis.pane.set_alpha(0.8)
            ax.yaxis.pane.set_alpha(0.8)
            ax.zaxis.pane.set_alpha(0.6)
            
            # Hide tick labels for cleaner look
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=8)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            
            # Elegant parameter info panel (top-left)
            param_text = (
                f'N = {N}\n'
                f'L = {L:.1f}\n'
                f'r_r = {system.r_repulsion:.1f}\n'
                f'r_o = {system.r_orientation:.1f}\n'
                f'r_a = {system.r_attraction:.1f}'
            )
            ax.text2D(0.02, 0.96, param_text,
                     transform=ax.transAxes, fontsize=9,
                     color=self.COLOR_TEXT, ha='left', va='top',
                     alpha=0.85, family='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', 
                              facecolor=self.COLOR_PANEL,
                              edgecolor=self.COLOR_GRID, 
                              alpha=0.85,
                              linewidth=0.5))
            
            # Clean status bar (bottom center)
            info_text = f't = {time[idx]:.1f}     P = {polarization[idx]:.3f}     M = {rotation[idx]:.3f}'
            ax.text2D(0.5, 0.03, info_text,
                     transform=ax.transAxes, fontsize=11,
                     color=self.COLOR_ACCENT, ha='center', va='bottom',
                     alpha=0.9, fontfamily='monospace')
            
            fig.tight_layout()
            
            # Render frame to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=anim_dpi,
                       facecolor=self.COLOR_ABYSS, edgecolor='none')
            buf.seek(0)
            frame_img = Image.open(buf).copy()
            
            # Apply subtle underwater glow effect
            frame_img = self._add_underwater_glow(frame_img, intensity=1.1)
            
            frames.append(frame_img)
            buf.close()
            plt.close(fig)
        
        # Smooth playback timing
        frame_duration_ms = int(1000 / self.fps)
        
        print(f"      Saving animation ({n_frames} frames)...")
        
        frames[0].save(
            str(filepath),
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=True,
            quality=90
        )
        
        print(f"      Complete: {filepath.name}")
