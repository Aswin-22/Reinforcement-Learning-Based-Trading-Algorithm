import numpy as np

class StateNormalizer:
    def __init__(self, price_min, price_max, initial_cash=10000):
        # You provide the price range from your dataset
        # Compute these once before training:
        # price_min = df['Close'].min()
        # price_max = df['Close'].max()
        self.price_min = price_min
        self.price_max = price_max
        self.initial_cash = initial_cash

    def normalize(self, state):
        close, sma, rsi, position, cash = state

        norm_close    = (close - self.price_min) / (self.price_max - self.price_min)
        norm_sma      = (sma   - self.price_min) / (self.price_max - self.price_min)
        norm_rsi      = rsi / 100.0                        # RSI is always 0–100
        norm_position = float(position)                    # already 0 or 1
        norm_cash     = cash / self.initial_cash           # cash as fraction of starting cash

        return np.array([norm_close, norm_sma, norm_rsi, norm_position, norm_cash],
                        dtype=np.float32)