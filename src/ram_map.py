import numpy as np

"""
Ms. Pac-Man NES RAM map (stable-retro), consolidated.
Includes:
- Pac-Man and ghost X/Y coordinates
- Global timers
- Stable frightened states (red and pink) + approximate (cyan and orange)
"""

PACMAN_X = 0x60  # 2 bytes
PACMAN_Y = 0x62  # 2 bytes
GHOSTS_XY = {
    'red':    (0x64, 0x66),
    'pink':   (0x68, 0x6A),
    'cyan':   (0x6C, 0x6E), 
    'orange': (0x70, 0x72)
}
FRIGHTENED_TIMER = 0xAB
SCATTER_TIMER = 0xBE

# Frightened states (stable flags)
RED_FRIGHTENED   = 0xD3  # reliable
PINK_FRIGHTENED  = 0xD6  # reliable
CYAN_FRIGHTENED  = 0xCA  # not reliable
ORANGE_FRIGHTENED= 0xCD  # not reliable


def get_all_positions(ram):
    """
    Get precise positions for all entities using:
    - Your verified RAM addresses
    - Proper 2-byte coordinate reading
    - Maze-relative conversion
    """
    def read_nes_coords(x_addr, y_addr):
        """Read 2-byte NES coordinates and convert to maze grid"""
        # Read 2-byte coordinates safely
        x = np.int16(ram[x_addr]) + (np.int16(ram[x_addr + 1]) << 8)
        y = np.int16(ram[y_addr]) + (np.int16(ram[y_addr + 1]) << 8)
    
        # Convert to maze coordinates (protected arithmetic)
        with np.errstate(all='ignore'):
            maze_col = (x - np.int16(8)) // 8
            maze_row = (y - np.int16(32)) // 8
        
        # Validate bounds
        if (0 <= maze_col < 21) and (0 <= maze_row < 30):
            return (maze_row, maze_col)
        return (-1, -1)

    return {
        'pacman': read_nes_coords(PACMAN_X, PACMAN_Y),
        'red':    read_nes_coords(*GHOSTS_XY['red']),
        'pink':   read_nes_coords(*GHOSTS_XY['pink']),
        'cyan':   read_nes_coords(*GHOSTS_XY['cyan']),
        'orange': read_nes_coords(*GHOSTS_XY['orange'])
    }

def get_frightened_states(ram):
    """
    Reliable frightened detection for Ms. Pac-Man NES
    - Red/Pink: Use 0xA0 (normal) vs 0x05 (frightened) as primary indicators
    - Cyan/Orange: Combine timer with observed value changes (01/03)
    - Global timer as fallback
    """
    frightened = {
        'red': ram[RED_FRIGHTENED] == 0x05 or ram[FRIGHTENED_TIMER] > 0,
        'pink': ram[PINK_FRIGHTENED] == 0x05 or ram[FRIGHTENED_TIMER] > 0,
        'cyan': ram[FRIGHTENED_TIMER] > 0 and ram[CYAN_FRIGHTENED] in (0x01, 0x03),
        'orange': ram[FRIGHTENED_TIMER] > 0 and ram[ORANGE_FRIGHTENED] in (0x01, 0x03),
        'global_timer': ram[FRIGHTENED_TIMER],
        'scatter_timer': ram[SCATTER_TIMER],
        'flags': {
            'red': f"0x{ram[RED_FRIGHTENED]:02X}",
            'pink': f"0x{ram[PINK_FRIGHTENED]:02X}",
            'cyan': f"0x{ram[CYAN_FRIGHTENED]:02X}",
            'orange': f"0x{ram[ORANGE_FRIGHTENED]:02X}"
        }
    }
    return frightened
def get_scatter_chase_timer(ram):
    """
    Returns the global scatter/chase timer.
    """
    return ram[SCATTER_TIMER]