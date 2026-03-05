import numpy as np

class TradingEnvironment:
    def __init__(self, df, initial_cash=10000):
        self.df = df
        self.initial_cash = initial_cash
        self.n_steps = len(df)

        # Portfolio state variables
        self.cash = initial_cash
        self.position = 0  # 0 = no stock, 1 = holding stock
        self.portfolio_value = initial_cash

        # Step tracker
        self.current_step = 0
    
    def get_discrete_state(self):
        price = self.df['Close'].iloc[self.current_step]
        sma = self.df['SMA_20'].iloc[self.current_step]
        rsi = self.df['RSI'].iloc[self.current_step]

        # Trend: 1 = Uptrend, 0 = Downtrend
        trend = 1 if price > sma else 0

        # RSI Zones
        if rsi < 30:
            rsi_zone = 0  # Oversold
        elif rsi > 70:
            rsi_zone = 2  # Overbought
        else:
            rsi_zone = 1  # Neutral

        position = self.position  # 0 or 1

        return (trend, rsi_zone, position)

    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.position = 0
        self.portfolio_value = self.cash
        self.prev_portfolio_value = self.cash

        return self.get_discrete_state()

    def _get_state(self):
        """
        Returns the current state observed by the agent.
        """
        row = self.df.loc[self.current_step]

        state = np.array([
            row['Close'],
            row['SMA_20'],
            row['RSI'],
            self.position,
            self.cash
        ], dtype=np.float32)

        return state

    def step(self, action):
        """
        Executes one timestep in the environment.
        Action:
        0 = Hold
        1 = Buy
        2 = Sell
        """
        done = False

        # Current price
        price = self.df.loc[self.current_step, 'Close']

        # Store previous portfolio value (for reward calculation)
        prev_portfolio_value = self.portfolio_value

        # === Execute Action ===
        if action == 1:  # Buy
            if self.position == 0 and self.cash >= price:
                self.position = 1
                self.cash -= price

        elif action == 2:  # Sell
            if self.position == 1:
                self.position = 0
                self.cash += price

        # Move to next timestep
        self.current_step += 1

        # Check if episode finished
        if self.current_step >= self.n_steps - 1:
            done = True

        # Calculate new portfolio value
        current_price = self.df.loc[self.current_step, 'Close']
        self.portfolio_value = self.cash + (self.position * current_price)

        # Reward = change in portfolio value
        reward = self.portfolio_value - prev_portfolio_value

        # Get next state
        next_state = self.get_discrete_state()
        return next_state, reward, done