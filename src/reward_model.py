import numpy as np

# =============================
# Defaults
# =============================

DEFAULT_REWARDS = {
    "pellet": 10,
    "power_pellet": 50,
    "frightened_ghost": 200
}

DEFAULT_RHO0 = 4            # Risk radius
DEFAULT_LAMBDA_RISK = 0.7   # Risk aversion

# =============================
# Reward function V(s,u)
# =============================


def compute_reward(pacman_pos, target_tile, ghosts, last_reward_state={}):
    """
        Compute V(s,u) — the reward function for the target tile Pac-Man moves to.
    
    Args:
        pacman_pos (tuple): (row, col) of Pac-Man.
        target_tile (str): bitmap character at target tile ('1', 'P', '#', '0').
        ghosts (list of dict): ghost dicts with 'pos' and 'frightened' keys.
        rewards (dict): reward weights.

    Returns:
        float: reward value
    """
    reward = 0
    
    # Position validation
    if pacman_pos == (-1, -1):
        return 0.0, last_reward_state
    
    # Direct pellet reward
    if target_tile == '1':
        reward += 10
        print(f"Pellet at {pacman_pos}, reward: +10")
    elif target_tile == 'P':
        reward += 100
        print(f"Power pellet at {pacman_pos}, reward: +100")
    
    # Ghost collision check
    for i, ghost in enumerate(ghosts):
        if ghost['pos'] == pacman_pos:
            if ghost['frightened']:
                reward += 200
                print(f"Frightened ghost eaten at {pacman_pos}, reward: +200")
            else:
                reward = -1000
                print(f"Death at {pacman_pos}, reward: -1000")
                break
    
    # Small step penalty to encourage efficiency
    reward -= 1
    
    return reward, last_reward_state

# =============================
# Risk function R(s,u)
# =============================

def compute_risk(
    pacman_pos,
    ghosts,
    rho0=DEFAULT_RHO0
):
    """
    Compute R(s,u) — risk based on dangerous ghosts and their proximity.
    
    Args:
        pacman_pos (tuple): (row, col) of Pac-Man.
        ghosts (list of dict): ghost dicts with 'pos' and 'frightened' keys.
        rho0 (int): risk radius.

    Returns:
        float: risk value
    """
    if pacman_pos == (-1, -1):
        return 0.0
        
    closest_ghost_dist = float('inf')
    
    for ghost in ghosts:
        if not ghost['frightened'] and ghost['pos'] != (-1, -1):
            dx, dy = abs(ghost['pos'][0] - pacman_pos[0]), abs(ghost['pos'][1] - pacman_pos[1])
            dist = dx + dy
            closest_ghost_dist = min(dist, closest_ghost_dist)
    
    if 0 < closest_ghost_dist <= rho0:
        return (1/closest_ghost_dist - 1/rho0) ** 2
    return 0.0

# =============================
# Utility function L(s,u)
# =============================

def compute_utility(
    reward,
    risk,
    lambda_risk=DEFAULT_LAMBDA_RISK
):
    """
    Compute L(s,u) — utility as reward minus weighted risk.

    Args:
        reward (float): reward value V(s,u)
        risk (float): risk value R(s,u)
        lambda_risk (float): risk aversion coefficient

    Returns:
        float: utility value
    """
    return reward - lambda_risk * risk

def direct_reward(ram, last_score=0):
    """
    Extract reward directly from RAM using score changes.
    
    Args:
        ram: RAM array from environment
        last_score: Previous score for comparison
        
    Returns:
        tuple: (reward, current_score)
    """
    from ram_map import get_score
    
    current_score = get_score(ram)
    reward = (current_score - last_score) // 10
    
    return reward, current_score
