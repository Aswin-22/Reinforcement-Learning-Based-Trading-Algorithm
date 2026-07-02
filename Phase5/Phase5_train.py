import torch
import pandas as pd
import numpy as np
import os
from Phase4_Environment import TradingEnvironment
from Phase5_network import DQNetwork, DQNAgent
from Phase5_normalizer import StateNormalizer

# ─────────────────────────────────────────
# 1. Load and prepare data (same as Phase 4)
# ─────────────────────────────────────────
df = pd.read_csv("./src/APPL1Y.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

df['SMA_20'] = df['Close'].rolling(20).mean()

delta = df['Close'].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs       = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

df = df.dropna().reset_index(drop=True)

# ─────────────────────────────────────────
# 2. Initialize all components
# ─────────────────────────────────────────
env        = TradingEnvironment(df)
agent      = DQNAgent(state_size=5, action_size=3)
normalizer = StateNormalizer(
    price_min=df['Close'].min(),
    price_max=df['Close'].max()
)

EPISODES = 50  # Same as Phase 4 — fair comparison

# ─────────────────────────────────────────
# 3. Training loop
# ─────────────────────────────────────────
for episode in range(EPISODES):

    # Reset environment — get raw state, then normalize it
    raw_state = env.reset()          # returns (trend, rsi_zone, position) — discrete
    # Switch to continuous state instead
    raw_state  = env._get_state()    # returns [Close, SMA_20, RSI, position, cash]
    state      = normalizer.normalize(raw_state)

    done         = False
    total_reward = 0

    while not done:

        # 3a. Agent chooses action from normalized state
        action = agent.choose(state)

        # 3b. Environment executes action — returns raw next state
        _, reward, done = env.step(action)
        # Note: env.step() returns discrete next_state — we ignore it (_)
        # and get the continuous version ourselves below

        # 3c. Get continuous next state and normalize it
        raw_next_state = env._get_state()
        next_state     = normalizer.normalize(raw_next_state)

        # 3d. Store experience in replay buffer
        agent.push(state, action, reward, next_state, done)

        # 3e. Train the online network on a random batch
        # (learn() internally checks if buffer has enough experiences)
        agent.learn()

        # 3f. Move to next step
        state         = next_state
        total_reward += reward
        agent.decay_epsilon()

    # ── End of episode ──

    # Decay epsilon — same as Phase 4
    

    # Sync target network every target_update_freq episodes
    agent.episode_count += 1
    if agent.episode_count % agent.target_update_freq == 0:
        agent.update_target_network()
        print(f"  >> Target network synced at episode {agent.episode_count}")

    print(f"Episode {episode+1:3d} | "
          f"Total Reward: {total_reward:8.2f} | "
          f"Portfolio: {env.portfolio_value:8.2f} | "
          f"Epsilon: {agent.epsilon:.3f} | "
          f"Buffer: {len(agent.memory)}")

# ─────────────────────────────────────────
# 4. Final results
# ─────────────────────────────────────────
print("\n" + "="*55)
print("Training complete")
print(f"Final Portfolio Value : {env.portfolio_value:.2f}")
print(f"Total Reward (last ep): {total_reward:.2f}")
print(f"Epsilon (final)       : {agent.epsilon:.4f}")
print(f"Experiences stored    : {len(agent.memory)}")
print("="*55)

# ─────────────────────────────────────────
# 5. Save the trained agent
# ─────────────────────────────────────────

# Create a models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save online network weights — this is all you need
torch.save(agent.online_network.state_dict(), "models/dqn_online.pth")

# Save epsilon so you can resume training from where you left off
# instead of restarting exploration from scratch
torch.save({
    'online_state_dict'  : agent.online_network.state_dict(),
    'target_state_dict'  : agent.target_network.state_dict(),
    'epsilon'            : agent.epsilon,
    'episode_count'      : agent.episode_count,
}, "models/dqn_checkpoint.pth")

print("\nModel saved to models/dqn_online.pth")
print("Checkpoint saved to models/dqn_checkpoint.pth")