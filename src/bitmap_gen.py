import numpy as np
import cv2

def initial_bitmap(frame, grid_rows=32, grid_cols=28):
    """
    Generates the initial bitmap of the stage:
    '#' = wall
    '1' = pellet
    'P' = power pellet
    '0' = empty space or letters
    Returns only columns 2 to 22 (playable map).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detection thresholds
    lower_pellet = np.array([10, 150, 150])
    upper_pellet = np.array([25, 255, 255])
    pellet_mask = cv2.inRange(hsv, lower_pellet, upper_pellet)

    lower_wall_gray = 30
    upper_wall_gray = 200
    wall_mask_gray = cv2.inRange(gray, lower_wall_gray, upper_wall_gray)

    cell_height = frame.shape[0] // grid_rows
    cell_width = frame.shape[1] // grid_cols

    bitmap_full = np.full((grid_rows, grid_cols), '0', dtype='<U1')

    # Power pellet positions (typical locations)
    power_pellet_positions = [(3, 1), (3, 26), (23, 1), (23, 26)]

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = row * cell_height
            x1 = col * cell_width
            y2 = y1 + cell_height
            x2 = x1 + cell_width

            cell_pellet = pellet_mask[y1:y2, x1:x2]
            cell_wall = wall_mask_gray[y1:y2, x1:x2]

            if np.mean(cell_wall) > 90:
                bitmap_full[row, col] = '#'
            elif np.mean(cell_pellet) > 10:
                # Check if this is a power pellet position
                if (row, col) in power_pellet_positions:
                    bitmap_full[row, col] = 'P'
                else:
                    bitmap_full[row, col] = '1'

    return bitmap_full[:, 0:21]  # Crop to playable area

def update_bitmap(bitmap, frame, pacman_pos=None, grid_rows=32, grid_cols=28):
    """Update bitmap output with position masking"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_pellet = np.array([10, 150, 150])
    upper_pellet = np.array([25, 255, 255])
    pellet_mask = cv2.inRange(hsv, lower_pellet, upper_pellet)

    cell_height = frame.shape[0] // grid_rows
    cell_width = frame.shape[1] // grid_cols

    # Power pellet positions (adjusted for cropped coordinates)
    power_pellet_positions = [(3, 1), (3, 19), (23, 1), (23, 19)]

    for row in range(bitmap.shape[0]):
        for col in range(bitmap.shape[1]):
            if bitmap[row, col] == '#':
                continue

            # Skip Pac-Man's current position to avoid false pellet detection
            if pacman_pos and (row, col) == pacman_pos:
                bitmap[row, col] = '0'
                continue

            col_orig = col + 2
            y1 = row * cell_height
            x1 = col_orig * cell_width
            y2 = y1 + cell_height
            x2 = x1 + cell_width

            cell_pellet = pellet_mask[y1:y2, x1:x2]
            
            if np.mean(cell_pellet) > 50:
                # Check if this is a power pellet position
                if (row, col) in power_pellet_positions:
                    bitmap[row, col] = 'P'
                else:
                    bitmap[row, col] = '1'
            else:
                bitmap[row, col] = '0'

    return bitmap


def print_bitmap(bitmap):
    """
    Prints the map in a readable way
    """
    for row in bitmap:
        print(" ".join(row))
