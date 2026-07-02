import torch
import numpy as np
import pandas as pd
from Phase4_Environment import TradingEnvironment
from Phase5_network import DQNetwork, DQNAgent
from Phase5_normalizer import StateNormalizer

# ─────────────────────────────────────────
# Option A: Load for inference only
# (just want to use the agent, not train it)
# ─────────────────────────────────────────
def load_agent_for_inference(path="models/dqn_online.pth"):
    agent = DQNAgent(state_size=5, action_size=3)

    # Load saved weights into online network
    agent.online_network.load_state_dict(
        torch.load(path, weights_only=True)
    )

    # Set to evaluation mode — disables any training-only behaviour
    agent.online_network.eval()

    # Force epsilon to 0 — pure exploitation, no random actions
    agent.epsilon = 0.0

    print(f"Model loaded from {path}")
    print("Epsilon set to 0.0 — agent will act greedily")
    return agent


# ─────────────────────────────────────────
# Option B: Load checkpoint to resume training
# ─────────────────────────────────────────
def load_agent_for_training(path="models/dqn_checkpoint.pth"):
    agent = DQNAgent(state_size=5, action_size=3)

    checkpoint = torch.load(path, weights_only=True)

    agent.online_network.load_state_dict(checkpoint['online_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
    agent.epsilon       = checkpoint['epsilon']
    agent.episode_count = checkpoint['episode_count']

    print(f"Checkpoint loaded from {path}")
    print(f"Resuming from episode {agent.episode_count}, epsilon {agent.epsilon:.4f}")
    return agent