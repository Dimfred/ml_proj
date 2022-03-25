# Project Summary

## CLI

```
./cli --help

# create the data for an experiment
./cli data <experiment_number>

# train an experiment
./cli train <experiment_number>

# test an experiment
./cli test <experiment_number>

# view charts, opens browser in the experiment directory
# navigate to <number>_experiment/<train/test> click chart
./charts.sh

```

## Framework

During the project, various frameworks have been investigated, which could suite the aim of this project.
The selection was based on the following criteria:
- financial training environment (exchange, market orders, stop loss, take profit, trade size, etc.)
- ability to simply modify the data
- easy integration of other models
- ability for live trading

The decision was made to use [TensorTrade](https://github.com/tensortrade-org), because it appears to be the most mature framework out there and meets all the described criteria.
The framework sadly does not support the trading of futures, therefore shorts are not an option.

## Changes made to default components of the framework

- refactor models and move train logic into a trainer
- add ability to split chart rendering (a lot of datapoints are rendered very slow in js/html, which increases the training time)
- some scripts to easily inspect charts in the browser (`./charts.sh`)
- add a config to define an experiment and a cli for easy training reproducibility
- add a feature generation pipeline, where features can be build from a string in the config

## Reinforcement Learning (RL)

RL in general can be structured into the following core parts:

1. An env in which an agent interacts
1. An agent who picks actions, which are then applied to the env, where the state of the env changes based on the action
1. An action space, where the agent can pick actions from
1. A reward function, which is used during training, which is calculated based on the env and the action taken by the agent

## RL in TensorTrade

The environment consists of the trading chart.
Trading chart normally means candlesticks aka. the Open, High, Low, Close and Volume (OHLCV) in a certain timeframe.
Indicators are functions, which generate features from the candlestick data, and are also part of the environment.
Changes in the environment occur on every timestep, where new sequential data is added to the sequence of available data.
Part of the environment is also the available equity in the wallet of the agent.
This changes based on trading decisions taken by the agent.
Normally an action is the submission of an order performed by the agent.
In this project an order can be specified by the following variables:

- Side (buy or sell)
- Stoploss (where the order should be closed with a loss)
- Takeprofit (where the order should be closed with a profit)
- Tradesize (size of the trade based on a percentage of the total available equity)

Those variables are selected by the agent during trading time and then an order is submitted.
Additionally, a reward function is used.
In this project, the reward function is based on the equity gain over time.
Therefore, the more the money the agent made in a certain lookback window the more he gets rewarded.

## Models

Two models are presented in this project.
The DeepQNetwork (DQN) and the Branch Dueling DeepQNetwork (BDQN).
The latter has been ported from tf1 to tf2, but has not been used in the experiments, since the underlying problems presented in the experiments don't appear to be solveable by another architecture.
Tho, this is just speculation and should be taken with a grain of salt.

### [DQN](https://arxiv.org/abs/1312.5602)

The DQN's input consists of a T x X matrix, where T is the length of the input window and X the number of features.
In most experiements a T of 60 was used (60 * 5 minute candles).
X varies depending on the features used in an experiment.
In the original version of the DQN the input is processed by an Conv / MLP network.
The network looks as follows:

Type      | Filters  | KernelSize / Neurons | Activation | Padding
--        | --       | --                   | --         | --
Input     |          |                      |            |
Conv1D    | 64       | 6                    | tanh       | same
MaxPool1D |          | 2                    |            |
Conv1D    | 32       | 3                    | tanh       | same
MaxPool1D |          | 2                    |            |
Flatten   |          |                      |            |
Dense     |          | NActions             | sig        |
Dense     |          | NActions             | softmax    |

In one experiment additionally two LSTM layers have been added after the Input layer, which could benefit the agent in better understanding the underlying sequential data.

The idea behind the DQN is to find a policy for which the reward function is optimal.
In previous RL models a action-state table was used, where for an observed state an optimal action has been observed.
In complex environments, this can rapidly lead to huge tables.
Further, not all entries can be filled during training, because not all state have been observed during training.
DQN mitigates that through its policy network.
It has been shown that the network is also able to generalize well over unseen states.

### [BDQN](https://arxiv.org/abs/1711.08946)

The BDQN is an extension of the Dueling-DeepQNetwork ([DDQN](https://arxiv.org/abs/1511.06581)).
A dueling architecture separates the state value and the advantage of an action into two separate networks (streams).
This is reused in the BDQN.
In the RL in Finance section the possible values have been introduced with which an order is parameterized.
To create a suiteable action space all possible permutations of the variables are created.
With a high dimensionality per variable this can lead to a huge amount of possible actions.
To tackle this, the BDQN introduces separate action branches per variable, where each branch is a separate network.
It has been shown, that this architecture can output pseudo continues actions through very small discretization.

It was planned to use this architecture to give the agent a high space of actions to chose from, but since the base models couldn't perform well it has never been tested.

## Data

For performed experiments data from the binance cryptocurrency exchange has been collected.
The data is in the form of OHLCV on a 1 minute timeframe ranging from 12.09.2020 - 27.05.2021.
To reduce the noise on the data, which is heavy in a 1 minute timeframe, the data has been resampled into a 5 minute timeframe.
Data was used offline.
Live trading can be enabled in the framework through [CCXT](https://github.com/ccxt/ccxt), tho this was untested.

The raw price and volume data is fed into the chart renderer, when used for training and testing it is further processed.
For stationary data like the one produced by the RSI indicator data is normalized based on the maximum value.
Price data on the other hand is made stationary by applying the fractional difference on the data.
The fractional difference can be calculated as follows:

```python
close'[i] = (close[i] / close[i - 1]) - 1

This formula is derived from the derivation, which is defined as:

close[i]' = (close[i] - close[i-1]) / i

Since in candlesticks i and i-1 will alwasys be one, since there is always a one candle difference, this is equal to:

close[i]' = close[i] - close[i-1]

This tho contains an absolute difference of the price. To make it relative it is divided by close[i-1]:

close'[i] = (close[i] - close[i-1]) / close[i - 1] = (close[i] / close[i - 1]) - 1

```

Afterwards, min-max normalization is applied:

```python
close = max( close.max(), close.min().abs() )
```

So the data is scaled based on the biggest value, or the absolute of the lowest value.
The corresponding scaling variables are stored and the test dataset is processed with the scaling variables from the training dataset.

Data is split with 90% for training and 10% for testing.
During training no separate dataset is used for evaluation.

## Experiments

The numbers behind each experiment indicate the number of the experiment in the folder.
Final Equity is defined as the equity, which is hold by the agent at the end of the episode.
In general during the training period the price increases int about 600%, therefore a training can be considered successful, when the agent beats those 600% (beats the market).
During the testing period the price drops about 30%, therefore it can be considered successful when the agent manages to be profitable at all.

### Learning Rate (2 - 7)

First some initial learning rates were tested.

LR_train     | 0.0001 (2) | 0.0005 (3) | 0.001 (4) | 0.005 (5) | 0.01 (6) | 0.05 (7)
--           | --         | --         | --        | --        | --       | --
final_equity | 2536       | 1963       | 1692      | 1410      | 1819     | 3073

LR_test      | 0.0001     | 0.0005     | 0.001     | 0.005     | 0.01     | 0.05
--           | --         | --         | --        | --        | --       | --
final_equity | 726        | 746        | 742       | 783       | 771      | 723

Best learning rate is applied to the following experiments.
The large learning rate hasn't done any trades with the weights of the first episode.
It is probably too high and is disregarded.
All other learning rates seem to perform more or less equally.
The further trainings are continued with a learning rate of 0.005

The trading decisions are hard to interpret on the chart.
It seems that the model is performing a lot of trades in general.
Sometimes peaks are hit very precisely, but this could just be luck.
Hard to tell.

### Raw Data Experiments (8 - 11)

In this experiment I wanted too see how the model performs with only raw data, aka. no indicators applied to the data.
In the default setting I used `OHLCV`.
Here I also want to see how the model performs on `OC`, `C`, `OCV`.

Data_train   | C        | OC   | OCV  | CV
--           | --       | --   | --   | --
final_equity | 2005     | 2230 | 1583 | 1660

Data_test    | C        | OC  | OCV   | CV
--           | --       | --  | --    | --
final_equity | 744      | 729 | 772   | 749

`OCV` is selected.

### Adding SMA Indicator (12 - 16)

Using here `OCV` + indicator(s).
From here on the initial equity is increased to 10k, since some trades can't be taken because the position size is too small, due to the precision of the asset.

This experiments should show whether the model learns better when the `SMA` indicator is added as a feature.
`SMA` should remove noise from the raw samples, and in with higher windows also should give a good estimate of the current trend.

SMA25 | SMA50 | SMA100  | SMA200 | SMA500 | train | test
--    | --    | --      | --     | --     | --    | --
x     | _     | _       | _      | _      | 1676  | 6589
_     | x     | _       | _      | _      | 2114  | 6860
_     | _     | x       | _      | _      | 1697  | 5804
_     | _     | _       | x      | _      | 1645  | 5764
_     | _     | _       | _      | x      | 1887  | 6340


The losses here are equivalent to the losses of the market.
Market dumped to ~63% of the initial price in that time.
Sometimes the losses are actually worse.
It looks like indicators didn't really help here.
From here maybe it should just be considered to switch the model, or change the reward function to be incorporate the time.
The initial intention behind this project is to train a daytraiding model.
Right now it looks like the model didn't really learn anything.

<!-- ### ding RSI Indicator -->
<!-- The SI` is a momentum indicator. -->
<!-- It is used in various strategies to capture overbought and oversold regions. -->

### Update the policy network to use LSTM (17)

We add two LSTM layers on top of the policy network as has been done by Chen et. al in "Deep Q-Learning with Recurrent Neural Networks".
This could maybe improve the understanding of the chart for the model.
The features are OCV + SMA(25, 50, 100, 200, 500)

train | test
--    | --
11617 | -

No trades have been taken during the testing period.

### LSTM with 40 episodes (18)

Maybe the model is not able to generalize in such a short period of time, therefore try some more episodes.

train | test
--    | --
      |

## Future Experiments, Ideas, Implementations

- it would be very interesting to see how the agent performs in a futures exchange environment (short trades), tho it is likely that the outcome will be the same
- adapt the reward function to not use the gain of the portfolio over a certain timeframe, but instead trim the agent on a positive outcome of a trade, aka maximize the winrate
- maybe play more with indicators, could be that they just don't represent the data well enough
- play with the model, better understanding of the data could lead to better results


