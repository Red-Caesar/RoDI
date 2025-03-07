import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple


class StockEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, tickers: List[str], initial_budget: float = 10000.0):
        """
        Initialize the environment

        Args:
            data: DataFrame with stock data from yfinance
            tickers: List of stock tickers
            initial_budget: Initial portfolio budget
        """
        super(StockEnv, self).__init__()
        processed_data = []

        for column in data.columns:
            ticker = column[1]  # yfinance creates columns like ('Open', 'AAPL')
            feature = column[0]

            if feature in ["Open", "High", "Low", "Close", "Volume"]:
                series = data[column].copy()
                series.name = feature
                df_ticker = pd.DataFrame(series)
                df_ticker["Ticker"] = ticker
                df_ticker.reset_index(inplace=True)
                df_ticker.rename(columns={"index": "Date"}, inplace=True)
                processed_data.append(df_ticker)

        self.data = pd.concat(processed_data)
        self.data = self.data.pivot_table(
            values=["Open", "High", "Low", "Close", "Volume"], index=["Date", "Ticker"]
        )

        self.tickers = tickers
        self.initial_budget = initial_budget
        self.current_step = 0
        self.max_steps = len(self.data.index.get_level_values(0).unique()) - 1

        # Actions: portfolio weights for each asset (must sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(tickers),), dtype=np.float32)

        # State space: price, volume, holdings for each asset + cash
        # For each ticker: [open, high, low, close, volume, holdings]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(tickers) * 6 + 1,), dtype=np.float32
        )

        self.portfolio = {ticker: 0 for ticker in tickers}
        self.cash = initial_budget
        self.portfolio_value_history = []

    def _get_observation(self) -> np.ndarray:
        dates = self.data.index.get_level_values(0).unique()
        current_date = dates[self.current_step]
        current_data = self.data.xs(current_date, level=0)

        obs = []
        for ticker in self.tickers:
            if ticker in current_data.index:
                ticker_data = current_data.loc[ticker]
                obs.extend(
                    [
                        ticker_data["Open"],
                        ticker_data["High"],
                        ticker_data["Low"],
                        ticker_data["Close"],
                        ticker_data["Volume"],
                        self.portfolio[ticker],
                    ]
                )
            else:
                # If ticker data is missing, use zeros
                obs.extend([0, 0, 0, 0, 0, self.portfolio[ticker]])

        obs.append(self.cash)

        return np.array(obs, dtype=np.float32)

    def _get_portfolio_value(self) -> float:

        dates = self.data.index.get_level_values(0).unique()
        current_date = dates[self.current_step]
        current_data = self.data.xs(current_date, level=0)

        portfolio_value = self.cash
        for ticker in self.tickers:
            if ticker in current_data.index:
                portfolio_value += self.portfolio[ticker] * current_data.loc[ticker]["Close"]

        return portfolio_value

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio = {ticker: 0 for ticker in self.tickers}
        self.cash = self.initial_budget
        self.portfolio_value_history = [self.initial_budget]

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment

        Args:
            action: Portfolio weights for each asset (must sum to 1)
        """
        if self.current_step >= self.max_steps:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {"portfolio_value": self._get_portfolio_value(), "return": 0},
            )

        # Ensure action sums to 1
        action = action / np.sum(action) if np.sum(action) > 0 else action

        dates = self.data.index.get_level_values(0).unique()
        current_date = dates[self.current_step]

        current_data = self.data.xs(current_date, level=0)
        prev_portfolio_value = self._get_portfolio_value()

        self.cash = prev_portfolio_value
        self.portfolio = {ticker: 0 for ticker in self.tickers}

        for i, ticker in enumerate(self.tickers):
            if ticker in current_data.index and action[i] > 0:
                amount_to_invest = prev_portfolio_value * action[i]
                price = current_data.loc[ticker]["Close"]
                shares = amount_to_invest / price

                self.portfolio[ticker] = shares
                self.cash -= shares * price

        self.current_step += 1
        done = self.current_step >= self.max_steps

        new_portfolio_value = self._get_portfolio_value()
        self.portfolio_value_history.append(new_portfolio_value)
        reward = (new_portfolio_value / prev_portfolio_value) - 1  # Return

        obs = self._get_observation()

        info = {"portfolio_value": new_portfolio_value, "return": reward}

        return obs, reward, done, False, info
