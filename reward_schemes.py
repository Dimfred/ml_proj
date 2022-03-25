import numpy as np
import pandas as pd

from tensortrade.env.generic import RewardScheme
from tensortrade.feed.core import Stream, DataFeed


class RiskAdjustedReturns(RewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth,
    while penalizing more volatile strategies.

    Parameters
    ----------
    return_algorithm : {'sharpe', 'sortino'}, Default 'sharpe'.
        The risk-adjusted return metric to use.
    risk_free_rate : float, Default 0.
        The risk free rate of returns to use for calculating metrics.
    target_returns : float, Default 0
        The target returns per period for use in calculating the sortino ratio.
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(
        self,
        return_algorithm: str = "sharpe",
        risk_free_rate: float = 0.0,
        target_returns: float = 0.0,
        window_size: int = 1,
    ) -> None:
        algorithm = self.default("return_algorithm", return_algorithm)

        assert algorithm in ["sharpe", "sortino"]

        if algorithm == "sharpe":
            return_algorithm = self._sharpe_ratio
        elif algorithm == "sortino":
            return_algorithm = self._sortino_ratio

        self._return_algorithm = return_algorithm
        self._risk_free_rate = self.default("risk_free_rate", risk_free_rate)
        self._target_returns = self.default("target_returns", target_returns)
        self._window_size = self.default("window_size", window_size)

    def reward(self, env: "TradingEnv") -> float:
        return self.get_reward(env.action_scheme.portfolio)

    def _sharpe_ratio(self, returns: "pd.Series") -> float:
        """Computes the sharpe ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sharpe ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1e-9) / (
            np.std(returns) + 1e-9
        )

    def _sortino_ratio(self, returns: "pd.Series") -> float:
        """Computes the sortino ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sortino ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()
        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - self._risk_free_rate + 1e-9) / (downside_std + 1e-9)

    def get_reward(self, portfolio: "Portfolio") -> float:
        """Computes the reward corresponding to the selected risk-adjusted return metric.

        Parameters
        ----------
        portfolio : `Portfolio`
            The current portfolio being used by the environment.

        Returns
        -------
        float
            The reward corresponding to the selected risk-adjusted return metric.
        """
        net_worths = [nw["net_worth"] for nw in portfolio.performance.values()][
            -(self._window_size + 1) :
        ]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)

        return risk_adjusted_return


class WinningRateBasedReturn(RewardScheme):
    """A reward scheme that rewards the agent based on the winning rate of trades

    Parameters
    ----------
    """

    def __init__(
        self,
    ) -> None:
        pass

    def reward(self, env: "TradingEnv") -> float:
        return self.get_reward(env.action_scheme.portfolio)

    def get_reward(self, portfolio: "Portfolio") -> float:
        """Computes the reward corresponding to the selected risk-adjusted return metric.

        Parameters
        ----------
        portfolio : `Portfolio`
            The current portfolio being used by the environment.

        Returns
        -------
        float
            The reward corresponding to the selected risk-adjusted return metric.
        """
        net_worths = [nw["net_worth"] for nw in portfolio.performance.values()][
            -(self._window_size + 1) :
        ]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)

        return risk_adjusted_return
