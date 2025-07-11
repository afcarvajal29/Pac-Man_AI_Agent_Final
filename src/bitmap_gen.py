import numpy as np
import cv2

# Frame counter for bitmap updates
_frame_counter = 0
_cached_bitmap = None

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

    # Improved detection thresholds
    lower_pellet = np.array([8, 100, 100])
    upper_pellet = np.array([35, 255, 255])
    pellet_mask = cv2.inRange(hsv, lower_pellet, upper_pellet)
    
    # Apply noise reduction
    kernel = np.ones((2, 2), np.uint8)
    pellet_mask = cv2.morphologyEx(pellet_mask, cv2.MORPH_CLOSE, kernel)

    # Enhanced wall detection
    wall_mask_gray = cv2.inRange(gray, 20, 180)
    blue_channel = frame[:, :, 2]
    wall_mask_blue = (blue_channel > 80).astype(np.uint8) * 255
    wall_mask = cv2.bitwise_or(wall_mask_gray, wall_mask_blue)

    cell_height = frame.shape[0] // grid_rows
    cell_width = frame.shape[1] // grid_cols

    bitmap_full = np.full((grid_rows, grid_cols), '0', dtype='<U1')

    # Dynamic power pellet detection
    contours, _ = cv2.findContours(pellet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    power_pellet_positions = set()
    for contour in contours:
        if cv2.contourArea(contour) > 40:
            x, y, w, h = cv2.boundingRect(contour)
            grid_row, grid_col = y // cell_height, x // cell_width
            if 0 <= grid_row < grid_rows and 0 <= grid_col < grid_cols:
                power_pellet_positions.add((grid_row, grid_col))

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1, x1 = row * cell_height, col * cell_width
            y2, x2 = y1 + cell_height, x1 + cell_width

            cell_pellet = pellet_mask[y1:y2, x1:x2]
            cell_wall = wall_mask[y1:y2, x1:x2]

            wall_score = np.mean(cell_wall) + (np.sum(cell_wall > 0) / cell_wall.size) * 50
            pellet_score = np.mean(cell_pellet) + np.max(cell_pellet) * 0.3

            if wall_score > 70:
                bitmap_full[row, col] = '#'
            elif pellet_score > 25:
                if (row, col) in power_pellet_positions:
                    bitmap_full[row, col] = 'P'
                else:
                    bitmap_full[row, col] = '1'

    return bitmap_full[:, 0:21]

def update_bitmap(bitmap, frame, pacman_pos=None, grid_rows=32, grid_cols=28):
    """Update bitmap every 6 frames to save CPU"""
    global _frame_counter, _cached_bitmap
    
    _frame_counter += 1
    
    # Only update bitmap every 6 frames
    if _frame_counter % 6 != 0 and _cached_bitmap is not None:
        # Just update Pac-Man position on cached bitmap
        result = _cached_bitmap.copy()
        if pacman_pos:
            result[pacman_pos[0], pacman_pos[1]] = '0'
        return result
    
    # Full bitmap update every 6 frames
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_pellet = np.array([8, 100, 100])
    upper_pellet = np.array([35, 255, 255])
    pellet_mask = cv2.inRange(hsv, lower_pellet, upper_pellet)
    
    kernel = np.ones((2, 2), np.uint8)
    pellet_mask = cv2.morphologyEx(pellet_mask, cv2.MORPH_CLOSE, kernel)

    cell_height = frame.shape[0] // grid_rows
    cell_width = frame.shape[1] // grid_cols

    # Dynamic power pellet detection
    contours, _ = cv2.findContours(pellet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    power_pellet_positions = set()
    for contour in contours:
        if cv2.contourArea(contour) > 40:
            x, y, w, h = cv2.boundingRect(contour)
            grid_row = y // cell_height
            grid_col = (x // cell_width) - 2
            if 0 <= grid_row < bitmap.shape[0] and 0 <= grid_col < bitmap.shape[1]:
                power_pellet_positions.add((grid_row, grid_col))

    for row in range(bitmap.shape[0]):
        for col in range(bitmap.shape[1]):
            if bitmap[row, col] == '#':
                continue

            if pacman_pos and (row, col) == pacman_pos:
                bitmap[row, col] = '0'
                continue

            col_orig = col + 2
            y1, x1 = row * cell_height, col_orig * cell_width
            y2, x2 = y1 + cell_height, x1 + cell_width

            cell_pellet = pellet_mask[y1:y2, x1:x2]
            pellet_score = np.mean(cell_pellet) + np.max(cell_pellet) * 0.3
            
            if pellet_score > 35:
                if (row, col) in power_pellet_positions:
                    bitmap[row, col] = 'P'
                else:
                    bitmap[row, col] = '1'
            else:
                bitmap[row, col] = '0'

    _cached_bitmap = bitmap.copy()
    return bitmap

def print_bitmap(bitmap):
    """
    Prints the map in a readable way
    """
    for row in bitmap:
        print(" ".join(row))
