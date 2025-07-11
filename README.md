# Pac-Man AI Agent Final

A Deep Q-Network (DQN) reinforcement learning agent specialized for playing the original Ms. Pac-Man NES game. This project implements a CNN-based policy network that learns to play Pac-Man through experience replay and epsilon-greedy exploration.

## Project Overview

This AI agent uses:
- **Deep Q-Network (DQN)** with experience replay
- **Convolutional Neural Network** for processing game states
- **Stable-Retro** environment for NES game emulation
- **PyTorch** for deep learning implementation
- **Custom reward system** optimized for Pac-Man gameplay

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv pacman_ai_env

# Activate virtual environment
# On Linux/Mac:
source pacman_ai_env/bin/activate
# On Windows:
pacman_ai_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install opencv-python stable-retro gymnasium pygame torch numpy
```

### 3. Project Structure

```
Pac-Man_AI_Agent_Final/
├── src/
│   ├── DQN.py              # Main training script
│   ├── agent_policy.py     # CNN policy network
│   ├── pacman_env.py       # Game environment wrapper
│   ├── reward_model.py     # Custom reward system
│   ├── render.py           # Pygame rendering
│   └── test.py             # Testing utilities
├── models/                 # Saved model checkpoints
├── roms/                   # Game ROM files
└── README.md
```

## Training the Agent

### Start Training

```bash
cd src
python DQN.py
```

### Training Features
- **Automatic model saving** every 10 episodes
- **Experience replay** with 100,000 memory buffer
- **Target network updates** every 1000 steps
- **Epsilon-greedy exploration** with decay
- **Frame skipping** for faster training
- **Optional rendering** for monitoring progress

## Loading Different Models

You can load different pre-trained models by modifying the `load_model` parameter in the DQN.py file before calling `train()`:

### Method 1: Load Specific Model

```python
# At the bottom of DQN.py, modify these lines:
if __name__ == "__main__":
    # Load specific model by filename
    p, ep, epsilon = load_model("models/dqn_pacman_ep100_20250711_005732.pth")
    train(p, ep, epsilon)
```

### Method 2: Load Latest Model Automatically

```python
if __name__ == "__main__":
    # Automatically load the most recent model
    p, ep, epsilon = load_existing_model()
    train(p, ep, epsilon)
```

### Method 3: Start Fresh Training

```python
if __name__ == "__main__":
    # Start training from scratch
    train()
```

### Available Models

The project includes several pre-trained models in the `models/` directory:
- `dqn_pacman_ep40_20250711_000209.pth` - 40 episodes trained
- `dqn_pacman_ep100_20250711_005732.pth` - 100 episodes trained
- `dqn_pacman_ep130_20250711_012333.pth` - 130 episodes trained

## Model Loading Parameters

The `load_model()` function accepts:
- `filepath`: Path to the saved model file
- `device`: Optional device specification (CPU/GPU)

Returns:
- `policy_net`: Loaded neural network
- `episode`: Episode number when model was saved
- `epsilon`: Exploration rate when model was saved

## Training Configuration

Key hyperparameters in DQN.py:
- `NUM_EPISODES = 5000` - Total training episodes
- `GAMMA = 0.99` - Discount factor
- `LR = 1e-4` - Learning rate
- `BATCH_SIZE = 64` - Training batch size
- `EPSILON_START = 1.0` - Initial exploration rate
- `EPSILON_END = 0.1` - Final exploration rate

## Requirements

- Python 3.9+ (recommended for optimal compatibility)
- CUDA-compatible GPU (optional, for faster training)
- Ms. Pac-Man NES ROM file

## Usage Tips

1. **Monitor training**: Set `DISABLE_RENDERING = False` to watch the agent play
2. **Speed up training**: Set `DISABLE_RENDERING = True` for faster training
3. **Resume training**: The system automatically saves and can resume from checkpoints
4. **Adjust difficulty**: Modify reward parameters in `reward_model.py`
