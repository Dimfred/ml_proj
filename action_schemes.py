import logging
from abc import abstractmethod
from itertools import product
from typing import Union, List, Any

from gym.spaces import Space, Discrete

from tensortrade.core import Clock
from tensortrade.env.default.actions import TensorTradeActionScheme
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.orders import (
    Broker,
    Order,
    OrderListener,
    OrderSpec,
    proportion_order,
    risk_managed_order,
    TradeSide,
    TradeType,
)
from tensortrade.oms.wallets import Portfolio


class ManagedRiskOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.

    Parameters
    ----------
    stop : List[float]
        A list of possible stop loss percentages for each order.
    take : List[float]
        A list of possible take profit percentages for each order.
    trade_sizes : List[float]
        A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
        '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
    durations : List[int]
        A list of durations to select from when submitting an order.
    trade_type : `TradeType`
        A type of trade to make.
    order_listener : OrderListener
        A callback class to use for listening to steps of the order process.
    min_order_pct : float
        The minimum value when placing an order, calculated in percent over net_worth.
    min_order_abs : float
        The minimum value when placing an order, calculated in absolute order value.
    """

    def __init__(
        self,
        stop: "List[float]" = [0.02, 0.04, 0.06],
        take: "List[float]" = [0.01, 0.02, 0.03],
        trade_sizes: "Union[List[float], int]" = 10,
        durations: "Union[List[int], int]" = None,
        trade_type: "TradeType" = TradeType.MARKET,
        order_listener: "OrderListener" = None,
        min_order_pct: float = 0.02,
        min_order_abs: float = 0.00,
    ):
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        self.stop = self.default("stop", stop)
        self.take = self.default("take", take)

        trade_sizes = self.default("trade_sizes", trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default("durations", durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default("trade_type", trade_type)
        self._order_listener = self.default("order_listener", order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> "Space":
        if not self._action_space:
            self.actions = product(
                self.stop,
                self.take,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL],
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: "Portfolio") -> "List[Order]":
        if action == 0:
            return []

        (ep, (stop, take, proportion, duration, side)) = self.actions[action]

        side = TradeSide(side)

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = balance * proportion
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if (
            size < 10 ** -instrument.precision
            or size < self.min_order_pct * portfolio.net_worth
            or size < self.min_order_abs
        ):
            return []

        params = {
            "side": side,
            "exchange_pair": ep,
            "price": ep.price,
            "quantity": quantity,
            "down_percent": stop,
            "up_percent": take,
            "portfolio": portfolio,
            "trade_type": self._trade_type,
            "end": self.clock.step + duration if duration else None,
        }

        order = risk_managed_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]
