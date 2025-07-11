import gymnasium as gym
import numpy as np
import retro
from gymnasium import spaces

from bitmap_gen import (
    initial_bitmap,
    update_bitmap,
    print_bitmap as print_bitmap_fn
)
from ram_map import (
    get_all_positions,
    get_frightened_states
)

from reward_model import (
    direct_reward as reward_fn,
    compute_risk as risk_fn,
    compute_utility as utility_fn
)

class PacManEnv(gym.Env):
    """
    Gymnasium wrapper for Ms. Pac-Man NES.
    Observation: bitmap of walls & pellets.
    Game state (RAM) available for advanced agents.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {render_mode}")

        self.render_mode = render_mode
        self.env = retro.make("MsPacMan-Nes", render_mode="rgb_array")
        self.last_score = 0  # Track score for direct reward

        # NES action space: already discrete with all directions
        self.action_space = self.env.action_space

        # Bitmap observation space: 32x21 cropped map
        self.bitmap_shape = (32, 21)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.bitmap_shape, dtype=np.int8
        )

        self.bitmap = None

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset()
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("render() returned None — check render_mode")

        self.bitmap = initial_bitmap(frame)
        self.last_score = 0  # Reset score tracking

        return self._get_obs(), info

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        
        game_state = self.get_game_state()
        pacman_pos = game_state["pacman"]
        
        # Check target tile BEFORE masking
        try:
            target_tile = self.bitmap[pacman_pos[0], pacman_pos[1]]
        except IndexError:
            target_tile = '0'
        
        # Update bitmap with Pac-Man position masking
        self.bitmap = update_bitmap(self.bitmap, frame, pacman_pos)
        
        ghosts = list(game_state["ghosts"].values())

        v, self.last_score = self.compute_reward(self.get_ram(), self.last_score)
        r = self.compute_risk(pacman_pos, ghosts)
        l = self.compute_utility(v, r)
        
        return self._get_obs(), l, terminated, truncated, info



    def render(self):
        """
        Return current RGB frame as numpy array (H, W, 3).
        """
        return self.env.render()
    


    def compute_reward(self, ram, last_score):
        """
        Wraps direct_reward() from reward.py
        """
        return reward_fn(ram, last_score)

    def compute_risk(self, pacman_pos, ghosts):
        """
        Wraps risk_fn() from reward.py
        """
        return risk_fn(pacman_pos, ghosts)

    def compute_utility(self, reward, risk):
        """
        Wraps utility_fn() from reward.py
        """
        return utility_fn(reward, risk)

    def close(self):
        self.env.close()

    def _get_obs(self) -> np.ndarray:
        """
        Return current bitmap observation (30x21 array).
        """
        return self.bitmap.copy()

    def print_bitmap(self):
        """
        Print bitmap as ASCII map.
        """
        print_bitmap_fn(self.bitmap)

    def get_ram(self) -> np.ndarray:
        """
        Return current RAM snapshot (2048 bytes).
        """
        return self.env.get_ram()

    def get_game_state(self):
        ram = self.get_ram()
        positions = get_all_positions(ram)
        frightened = get_frightened_states(ram)

        return {
            "pacman": positions["pacman"],
            "ghosts": {
                "red": {"pos": positions["red"], "frightened": frightened["red"]},
                "pink": {"pos": positions["pink"], "frightened": frightened["pink"]},
                "cyan": {"pos": positions["cyan"], "frightened": frightened["cyan"]},
                "orange": {"pos": positions["orange"], "frightened": frightened["orange"]},
            },
            "frightened_timer": frightened["global_timer"],
            "scatter_timer": frightened["scatter_timer"],  # Global scatter/chase timer
        }
