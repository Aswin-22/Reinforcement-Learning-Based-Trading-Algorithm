# ---------------------- Phase 2: RSI Rule-Based Backtest ----------------------

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Load Data ----------------------

df = pd.read_csv("./src/APPL1Y.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# ---------------------- RSI Calculation ----------------------

delta = df['Close'].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))


# ---------------------- Initial Setup ----------------------

initial_cash = 100000
cash = initial_cash

position = 0        # 0 = no stock, 1 = holding stock
buy_price = 0

# Performance tracking
portfolio_value = []
positions = []
trades = []

# Trading thresholds
BUY_RSI = 30
SELL_RSI = 70


# ---------------------- Backtesting Loop ----------------------

for date, row in df.iterrows():

    price = row['Close']
    rsi = row['RSI']

    # ---- Buy Logic (Oversold) ----
    if rsi < BUY_RSI and position == 0:
        position = 1
        buy_price = price
        cash -= price
        trades.append((date, 'BUY', price))

    # ---- Sell Logic (Overbought) ----
    elif rsi > SELL_RSI and position == 1:
        position = 0
        cash += price
        profit = price - buy_price
        trades.append((date, 'SELL', price, profit))

    # ---- Portfolio Value Calculation ----
    current_value = cash + (price if position == 1 else 0)

    portfolio_value.append(current_value)
    positions.append(position)


# ---------------------- Store Results ----------------------

df['Portfolio_Value'] = portfolio_value
df['Position'] = positions


# ---------------------- Performance Metrics ----------------------

# Total Return
total_return = (df['Portfolio_Value'].iloc[-1] - initial_cash) / initial_cash
print(f"Total Return: {total_return:.2%}")

# Number of completed trades (Buy + Sell)
num_trades = len(trades) // 2
print("Number of trades:", num_trades)

# Rolling maximum portfolio value
rolling_max = df['Portfolio_Value'].cummax()

# Drawdown series
drawdown = (df['Portfolio_Value'] - rolling_max) / rolling_max

# Maximum drawdown
max_drawdown = drawdown.min()
print(f"Max Drawdown: {max_drawdown:.2%}")

print(trades)


# ---------------------- Equity Curve ----------------------

plt.figure()
plt.plot(df.index, df['Portfolio_Value'])
plt.title("RSI Rule-Based Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.show()


# Data source example:
# https://www.marketwatch.com/investing/stock/aapl/download-data?startDate=1/1/2020&endDate=1/3/2022