from pacman_env import PacManEnv
from render import init_pygame, blit_frame, handle_pygame_events

# Constants
SCREEN_WIDTH = 224
SCREEN_HEIGHT = 288
SCALE = 2

# Initialize environment and Pygame window
env = PacManEnv(render_mode="rgb_array")
window = init_pygame(SCREEN_WIDTH, SCREEN_HEIGHT, scale=SCALE)

obs, info = env.reset()
terminated = False
truncated = False

step_count = 0

try:
    while not (terminated or truncated):
        # Render frame and show it
        frame = env.render()
        blit_frame(window, frame)
        handle_pygame_events()

        # Sample random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)



        step_count += 1
        if step_count % 100 == 0:
            print(f"\nStep: {step_count}")

            env.print_bitmap()
            print(env.get_game_state())

finally:
    env.close()
