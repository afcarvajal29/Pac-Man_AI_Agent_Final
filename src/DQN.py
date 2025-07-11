import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import time
from datetime import datetime

from render import init_pygame, blit_frame, handle_pygame_events
from pacman_env import PacManEnv
from agent_policy import CNNPolicy

# ========== HYPERPARAMETERS ==========
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 100_000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 1000
NUM_EPISODES = 5000
MAX_STEPS = 5000
FRAME_SKIP = 6  # Number of frames to repeat each action

# Training optimization parameters
RENDER_EVERY = 1  # Only render every N episodes
DISABLE_RENDERING = True # Set to True to completely disable pygame rendering
PRINT_EVERY = 1    # Print stats every N episodes
SAVE_EVERY = 10   # Save model every N episodes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========== EXPERIENCE REPLAY ==========
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in batch_indices]
        
        # More efficient unpacking and conversion
        states, actions, rewards, next_states, dones = zip(*batch)

        # Direct tensor conversion (more efficient)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Add channel dimension for CNN input
        states = states.unsqueeze(1)  # [batch, 1, 32, 21]
        next_states = next_states.unsqueeze(1)  # [batch, 1, 32, 21]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ========== PREPROCESS ==========
def preprocess(bitmap):
    """
    Convert bitmap (np.array of str) into normalized tensor of shape (32, 21).
    Optimized version with pre-computed mapping.
    """
    # Pre-computed mapping for faster vectorized operations
    mapping = {'0': 0.0, '1': 1.0, 'P': 2.0, '#': -1.0}
    
    # More efficient vectorized conversion
    numeric = np.vectorize(mapping.get, otypes=[float])(bitmap)
    
    # Ensure consistent shape (32, 21)
    if numeric.shape != (32, 21):
        padded = np.zeros((32, 21), dtype=np.float32)
        h = min(numeric.shape[0], 32)
        w = min(numeric.shape[1], 21)
        padded[:h, :w] = numeric[:h, :w]
        numeric = padded
    
    return numeric.astype(np.float32)  # Return numpy array, convert to tensor later

# ========== MODEL SAVING/LOADING ==========
def save_model(policy_net, target_net, optimizer, episode, epsilon, save_dir="models"):
    """Save the trained model and training state."""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dqn_pacman_ep{episode}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
    }, filepath)
    
    print(f"Model saved to {filepath}")
    return filepath

def load_model(filepath, device=None):
    """Simplified model loading that only requires the file path
    Args:
        filepath: Path to the saved model
        device: Device to load the model onto (default: None will use DEVICE constant)
    Returns:
        tuple: (policy_net, episode, epsilon)
    """
    if device is None:
        device = DEVICE
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Initialize new policy network
    policy_net = CNNPolicy(input_shape=(32, 21), num_actions=9).to(device)
    
    # Load state dict
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    
    # Extract training metadata
    episode = checkpoint.get('episode', 0)
    epsilon = checkpoint.get('epsilon', EPSILON_END)
    
    print(f"Model loaded from {filepath}")
    print(f"Training state: Episode {episode}, Epsilon {epsilon:.3f}")
    
    return policy_net, episode, epsilon

def find_latest_model(model_dir="models"):
    import glob
    if not os.path.exists(model_dir):
        return None
    
    model_files = glob.glob(os.path.join(model_dir, "dqn_pacman_*.pth"))
    if not model_files:
        return None
    
    return max(model_files, key=os.path.getctime)

def load_existing_model():
    latest_model = find_latest_model()
    if latest_model:
        policy_net, episode, epsilon = load_model(latest_model)
        return policy_net, episode + 1, epsilon
    return None, 0, EPSILON_START

# ========== MAIN ==========
def train(loaded_policy=None, start_episode=None, epsilon=None):
    env = PacManEnv()
    
    # Initialize pygame but don't create window initially
    window = None
    
    replay_buffer = ReplayBuffer(MEMORY_SIZE, DEVICE)

    # Try to load existing model, otherwise create new
    loaded_policy, start_episode, epsilon = load_existing_model() if loaded_policy is None else (loaded_policy, start_episode, epsilon)

    
    if loaded_policy is not None:
        policy_net = loaded_policy
        print(f"Resuming training from episode {start_episode}")
    else:
        policy_net = CNNPolicy(input_shape=(32, 21), num_actions=env.action_space.n).to(DEVICE)
        start_episode = 0
        print("Starting fresh training")
    
    target_net = CNNPolicy(input_shape=(32, 21), num_actions=env.action_space.n).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    global_step = 0
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print("Starting training...")
    start_time = time.time()

    for episode in range(start_episode, start_episode + NUM_EPISODES):
        obs, _ = env.reset()
        state = preprocess(obs)  # Keep as numpy array
        episode_reward = 0
        episode_loss = 0
        loss_count = 0

        # Only render occasionally to speed up training
        should_render = (episode % RENDER_EVERY == 0) and not DISABLE_RENDERING
        if should_render and window is None:
            window = init_pygame(252, 248, scale=2)

        for t in range(MAX_STEPS):
            global_step += 1

            # Render only occasionally
            if should_render:
                frame = env.render()
                blit_frame(window, frame)
                handle_pygame_events()

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # Convert state to tensor only when needed
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    
                    q_values = policy_net(state_tensor)
                    action_idx = q_values.argmax(1).item()
                    
                    # Convert single action index to multi-discrete format
                    action = np.zeros(9, dtype=int)
                    action[action_idx] = 1
            
            total_reward = 0
            done = False
            for _ in range(FRAME_SKIP):
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    done = True
                    break
            next_state = preprocess(next_obs)  # Keep as numpy array

            # Convert action to single integer for storage
            action_idx = np.argmax(action) if isinstance(action, np.ndarray) else action
            
            replay_buffer.push(
                state,
                action_idx,
                total_reward,
                next_state,
                done
            )

            state = next_state
            episode_reward += total_reward

            # Learn (batched training for efficiency)
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                
                # Forward pass
                q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                    targets = rewards.unsqueeze(1) + GAMMA * next_q_values * (1 - dones.unsqueeze(1))

                loss = mse_loss(q_values, targets)
                episode_loss += loss.item()
                loss_count += 1

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

            # Update target network
            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Update epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(t + 1)
        if loss_count > 0:
            losses.append(episode_loss / loss_count)
        
        # Print progress
        if episode % PRINT_EVERY == 0:
            avg_reward = np.mean(episode_rewards[-PRINT_EVERY:])
            avg_length = np.mean(episode_lengths[-PRINT_EVERY:])
            avg_loss = np.mean(losses[-PRINT_EVERY:]) if losses else 0
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Time: {elapsed_time:.1f}s")
        
        # Save model periodically
        if episode % SAVE_EVERY == 0 and episode > 0:
            save_model(policy_net, target_net, optimizer, episode, epsilon)

        # Close pygame window at the end of each episode if it was opened
        if window is not None:
            try:
                import pygame
                pygame.display.quit()
                window = None
            except Exception:
                pass

    # Final save
    final_path = save_model(policy_net, target_net, optimizer, NUM_EPISODES, epsilon)
    
    # Print final statistics
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final model saved to: {final_path}")
    
    env.close()

if __name__ == "__main__":


    # Load existing model if available, otherwise start latest
    # p, ep, epsilon = load_model("models/dqn_pacman_ep50_20250711_001006.pth")
    # train(p, ep, epsilon)

    train()