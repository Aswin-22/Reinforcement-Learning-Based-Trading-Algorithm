import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        super(DQNetwork, self).__init__()
        
        # This is your "Q-table replacement"
        # Input: 5 market features → Hidden: 64 → Hidden: 64 → Output: 3 Q-values
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # Layer 1: 5 inputs → 64 neurons
            nn.ReLU(),                             # Activation: "fire if positive"
            nn.Linear(hidden_size, hidden_size),  # Layer 2: 64 → 64
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)   # Output: 64 → 3 Q-values (no activation)
        )
    
    def forward(self, x):
        # x is a tensor of shape (batch_size, 5)
        # Returns Q-values of shape (batch_size, 3)
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=5, action_size=3):

        self.state_size = state_size    # [Close, SMA_20, RSI, position, cash]
        self.action_size = action_size  # Hold, Buy, Sell

        # --- Hyperparameters (same philosophy as Phase 4) ---
        self.gamma = 0.95          # Same as Phase 4 — long term reward importance
        self.alpha = 0.001         # Learning rate for optimizer (much smaller than Phase 4's 0.1)
        self.epsilon = 1.0         # Same as Phase 4 — start fully exploratory
        self.epsilon_decay = 0.9995  # Same decay schedule
        self.epsilon_min = 0.01

        self.batch_size = 32
        self.buffer_min = 64       # Wait until buffer has this many before training
        self.target_update_freq = 10  # Sync target network every 10 episodes

        # --- Replay Buffer ---
        self.memory = deque(maxlen=2000)

        # --- Networks ---
        # online_network: trained every step — the "learning" network
        self.online_network = DQNetwork(state_size, 64, action_size)

        # target_network: frozen copy — provides stable TD targets
        self.target_network = DQNetwork(state_size, 64, action_size)

        # Make target_network start as an exact copy of online_network
        self.target_network.load_state_dict(self.online_network.state_dict())

        # Freeze target_network — it will NOT be updated by the optimizer
        # We update it manually every target_update_freq episodes
        for param in self.target_network.parameters():
            param.requires_grad = False

        # --- Optimizer ---
        # Adam adjusts learning rate automatically per weight — better than fixed alpha
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.alpha)

        # --- Loss function ---
        # MSE: measures how far our predicted Q-value is from the TD target
        self.loss_fn = nn.MSELoss()

        # Episode counter — used to know when to sync target network
        self.episode_count = 0
    
    def choose(self, state):
        # Epsilon-greedy — identical logic to Phase 4
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore

        # Convert state to tensor so the network can process it
        # unsqueeze(0) adds a batch dimension: shape (5,) → (1, 5)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Tell PyTorch: don't track gradients here, we're just choosing
        with torch.no_grad():
            q_values = self.online_network(state_tensor)  # shape: (1, 3)

        # Pick action with highest Q-value — same as np.argmax in Phase 4
        return q_values.argmax().item()

    def learn(self):
        # Guard: don't train until buffer has enough variety
        if len(self.memory) < self.buffer_min:
            return

        # Sample a random batch of 32 experiences
        batch = random.sample(self.memory, self.batch_size)

        # Unpack the batch into separate arrays
        # Each is a list of 32 items
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors for PyTorch
        states      = torch.FloatTensor(np.array(states))       # shape: (32, 5)
        actions     = torch.LongTensor(actions)                  # shape: (32,)
        rewards     = torch.FloatTensor(rewards)                 # shape: (32,)
        next_states = torch.FloatTensor(np.array(next_states))  # shape: (32, 5)
        dones       = torch.FloatTensor(dones)                   # shape: (32,)

        # --- Compute what the online network currently predicts ---
        # For each of 32 states, get all 3 Q-values → shape (32, 3)
        all_q_values = self.online_network(states)

        # Pick only the Q-value for the action that was actually taken
        # .gather pulls the specific column (action index) for each row
        predicted_q = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # predicted_q shape: (32,) — one Q-value per experience

        # --- Compute the TD target using the FROZEN target network ---
        with torch.no_grad():
            # Get max Q-value of next state from target network
            next_q_values = self.target_network(next_states).max(1)[0]  # shape: (32,)

            # If done=True, there is no next state — target is just the reward
            # (1 - dones) zeroes out the future term for terminal steps
            td_target = rewards + self.gamma * next_q_values * (1 - dones)

        # --- Loss: how wrong were our predictions? ---
        loss = self.loss_fn(predicted_q, td_target)

        # --- Backpropagation: adjust online_network weights ---
        self.optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()             # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=1.0)  # Clip gradients
        self.optimizer.step()       # Update weights

    def push(self, state, action, reward, next_state, done):
        # Store one experience in the buffer
        self.memory.append((state, action, reward, next_state, done))

    def decay_epsilon(self):
        # Identical to Phase 4
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Manually copy online_network weights → target_network
        # Called every target_update_freq episodes, not every step
        self.target_network.load_state_dict(self.online_network.state_dict())