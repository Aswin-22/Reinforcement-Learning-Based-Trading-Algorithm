from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=2000):
        # deque automatically drops oldest item when full
        # 2000 is generous for ~200 rows of stock data
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store one experience tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        # Randomly pick batch_size experiences from memory
        # This is what breaks the correlation
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        # Lets you check how full the buffer is
        # e.g. len(replay_buffer) >= 64 before starting training
        return len(self.buffer)