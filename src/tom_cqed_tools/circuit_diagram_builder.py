import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from typing import Union, List, Optional

class CircuitDiagramBuilder:
    """
    A programmatic builder for Circuit QED sequence diagrams.
    """
    def __init__(self, modes: List[str], height: float = 3.5, theme: str = 'transparent'):
        """
        Initializes the CircuitDiagramBuilder.
        
        Args:
            modes (List[str]): List of mode names (e.g., ['Transmon', 'Buffer']).
            height (float): Total height of the figure.
            theme (str): Theme for alpha transparency. Options are 'solid' or 'transparent'.
        """
        self.modes = {name: i for i, name in enumerate(reversed(modes))}
        
        # Strictly locked to matplotlib's tab10 palette
        self.colors = {
            'blue': 'tab:blue', 'orange': 'tab:orange', 'green': 'tab:green',
            'red': 'tab:red', 'purple': 'tab:purple', 'brown': 'tab:brown',
            'pink': 'tab:pink', 'gray': 'tab:gray', 'olive': 'tab:olive', 
            'cyan': 'tab:cyan',
            'maroon': 'tab:brown'
        }
        
        # "Tom's Defaults": Alpha profile mapped to specific colors
        themes = {
            'solid': {}, # Empty dict defaults everything to 1.0
            'transparent': {
                'green': 0.7, 'red': 0.5, 'purple': 0.7, 
                'orange': 0.7, 'brown': 0.5, 'gray': 0.3
            }
        }
        self.alpha_theme = themes.get(theme, themes['solid'])
        
        self.t = 0.5
        self.fig, self.ax = plt.subplots(figsize=(12, height))
        self.ax.axis('off')

    def advance(self, dt: float = 1.2):
        """
        Moves the time cursor forward by the specified amount.
        
        Args:
            dt (float): The amount of time to advance the cursor.
        """
        self.t += dt

    def _draw_box(self, y: float, width: float, height: float, text: str, color: str, 
                  is_meas: bool = False, alpha: Optional[float] = None):
        """
        Internal method to draw a box (pulse, delay, gate, measurement).
        """
        # Use theme default if alpha is not explicitly passed
        if alpha is None:
            alpha = self.alpha_theme.get(color, 1.0)
            
        base_color = self.colors.get(color, 'tab:gray')
        rgb = np.array(mcolors.to_rgb(base_color))
        white = np.array([1.0, 1.0, 1.0])
        blended_rgb = alpha * rgb + (1.0 - alpha) * white
        
        box = FancyBboxPatch((self.t - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             ec="black", fc=blended_rgb, lw=1.5, zorder=2)
        self.ax.add_patch(box)
        
        if is_meas:
            eff_h = min(height, 1.0)
            theta = np.linspace(np.pi/6, 5*np.pi/6, 50)
            r = eff_h * 0.35
            self.ax.plot(self.t + r*np.cos(theta), y - eff_h*0.1 + r*np.sin(theta), 'k-', lw=1.5, zorder=3)
            self.ax.arrow(self.t, y - eff_h*0.15, r*0.5*np.cos(np.pi/3), r*0.7*np.sin(np.pi/3),
                          head_width=0.08, head_length=0.1, fc='k', ec='k', zorder=3)
        else:
            self.ax.text(self.t, y, text, ha='center', va='center', fontsize=14, zorder=3)

    # --- API Commands ---
    # Note: Default width is now 0.8 (wider than the 0.7 height) to prevent vertical-rectangle single gates.
    
    def pulse(self, mode: str, label: str, color: str = 'green', width: float = 0.8, alpha: Optional[float] = None):
        """
        Draws a single-mode pulse.
        
        Args:
            mode (str): The mode on which to apply the pulse.
            label (str): The label/text for the pulse.
            color (str): The color of the pulse box.
            width (float): The width of the pulse box.
            alpha (float, optional): Alpha transparency override.
        """
        self._draw_box(self.modes[mode], width, 0.7, label, color, alpha=alpha)

    def delay(self, mode: str, label: str, width: float = 1.8, alpha: Optional[float] = None):
        """
        Draws a single-mode delay block.
        
        Args:
            mode (str): The mode on which the delay occurs.
            label (str): The label/text for the delay.
            width (float): The width of the delay block.
            alpha (float, optional): Alpha transparency override.
        """
        self._draw_box(self.modes[mode], width, 0.7, label, 'orange', alpha=alpha)

    def two_mode_gate(self, mode1: str, mode2: str, label: str, color: str = 'purple', width: float = 0.8, alpha: Optional[float] = None):
        """
        Draws a gate that spans across two modes.
        
        Args:
            mode1 (str): The first mode of the gate.
            mode2 (str): The second mode of the gate.
            label (str): The label/text for the gate.
            color (str): The color of the gate box.
            width (float): The width of the gate box.
            alpha (float, optional): Alpha transparency override.
        """
        y1, y2 = self.modes[mode1], self.modes[mode2]
        center_y = (y1 + y2) / 2
        height = abs(y1 - y2) + 0.7
        self._draw_box(center_y, width, height, label, color, alpha=alpha)

    def measure(self, mode: Union[str, List[str]], width: float = 0.8, alpha: Optional[float] = None):
        """
        Draws a measurement block. If multiple modes are provided, the block spans them.
        
        Args:
            mode (Union[str, List[str]]): The mode(s) being measured.
            width (float): The width of the measurement block.
            alpha (float, optional): Alpha transparency override.
        """
        # Always force 'gray' color lookup for the alpha theme on measurements
        if alpha is None:
            alpha = self.alpha_theme.get('gray', 1.0)
            
        if isinstance(mode, (list, tuple)):
            if len(mode) == 1:
                self._draw_box(self.modes[mode[0]], width, 0.7, '', 'gray', is_meas=True, alpha=alpha)
            else:
                y1, y2 = self.modes[mode[0]], self.modes[mode[1]]
                center_y = (y1 + y2) / 2
                height = abs(y1 - y2) + 0.7
                self._draw_box(center_y, width, height, '', 'gray', is_meas=True, alpha=alpha)
        else:
            self._draw_box(self.modes[mode], width, 0.7, '', 'gray', is_meas=True, alpha=alpha)

    def render(self, save_as: Optional[Union[str, Path]] = None):
        """
        Renders the circuit diagram, drawing the mode lines and showing the plot.
        
        Args:
            save_as (Union[str, Path], optional): If provided, saves the figure to this path.
        """
        end_t = self.t + 1.0
        for mode_name, y in self.modes.items():
            self.ax.plot([-0.5, end_t], [y, y], color='black', lw=2, zorder=1)
            self.ax.text(-0.8, y, mode_name, va='center', ha='right', fontsize=14, color='black')
        
        self.ax.set_xlim(-2.0, end_t + 0.5)
        
        self.ax.set_aspect('equal')

        plt.tight_layout()
        if save_as:
            plt.savefig(save_as, bbox_inches='tight', transparent=True)
            print(f"Saved to {save_as}")
        plt.show()
