import pandas as pd
from Phase3_Environment import TradingEnvironment
import random

# Load your cleaned Phase 1 dataframe
df = pd.read_csv("./src/APPL1Y.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Indicators (same as Phase 1)
df['SMA_20'] = df['Close'].rolling(20).mean()

delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

df = df.dropna()
df = df.reset_index(drop=True)

# Create environment
env = TradingEnvironment(df)

state = env.reset()
done = False
total_reward = 0

while not done:
    action = random.choice([0, 1, 2])  # Random agent
    next_state, reward, done = env.step(action)
    total_reward += reward

print("Simulation finished")
print("Final Portfolio Value:", env.portfolio_value)
print("Total Reward:", total_reward)