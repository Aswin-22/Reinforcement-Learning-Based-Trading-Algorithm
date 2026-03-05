import numpy as np
import random


# Tabular Reinforcement Learning
class QLearningAgent:
    def __init__(self):
        # State space size:
        # Trend (2) × RSI Zone (3) × Position (2) = 12 states
        self.q_table = np.zeros((2, 3, 2, 3))  
        # Last dimension = 3 actions (Hold, Buy, Sell)

        self.alpha = 0.1      # Learning rate, 0.1 slow and steady, 0.9 fast and unstable
        self.gamma = 0.95     # Discount factor, 0 - instant profit, 0.95 long term profit

        # Exploration = Try random actions (learn new strategies)
        # Exploitation = Use best known action (make profit)
        self.epsilon = 1.0    # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose(self, state):
        trend, rsi, position = state

        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action
        else:
            return np.argmax(self.q_table[trend, rsi, position])

    def learn(self, state, action, reward, next_state):
        t, r, p = state
        nt, nr, np_ = next_state

        best_next_action = np.argmax(self.q_table[nt, nr, np_])

        td_target = reward + self.gamma * self.q_table[nt, nr, np_, best_next_action]
        td_error = td_target - self.q_table[t, r, p, action]

        self.q_table[t, r, p, action] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay