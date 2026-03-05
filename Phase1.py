import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./src/APPL1Y.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df = df.sort_index()

df['SMA_20'] = df['Close'].rolling(window=20).mean() # Rolling average of last 20 Days

# plt.figure()
# plt.plot(df.index, df['Close'], label='Close')
# plt.plot(df.index, df['SMA_20'], label='SMA 20')
# plt.legend()
# plt.show()

delta = df['Close'].diff()
gain = delta.clip(lower=0) # Capture only positive values and use 0 for negative ones
loss = -delta.clip(upper=0) # Same as abovee vice versa

avg_gain = gain.rolling(14).mean() # 14 is the standard amount of days used for RSI
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df['RSI'] =100 - (100 / (1 + rs)) # Scale relative strength to a degree of 0 - 100.
df = df.dropna()

print(df.isna().sum())
print(df.head())
print(df.index.is_monotonic_increasing)


# plt.figure()
# plt.plot(df.index, df['RSI'])
# plt.axhline(70)
# plt.axhline(30)
# plt.title("RSI")
# plt.show()