from pydantic import BaseModel
from enum import Enum

from typing import List, Union

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers


class Aggregator(Enum):
    naive = 0
    reduce_local_max = 1
    reduce_local_mean = 2
    reduce_global_mean = 3
    reduce_global_max = 4

    @staticmethod
    def from_str(val):
        return getattr(Aggregator, val)

    def __str__(self):
        return self.name

    def create(self):
        pass


class TargetVersion(Enum):
    mean = 0
    independent = 1
    max = 2


class ActivationFunction(Enum):
    ReLU = 0

    @staticmethod
    def from_str(val):
        return getattr(ActivationFunction, val)

    def __str__(self):
        return self.name

    def create(self, *args, **kwargs):
        ActivationFunction = getattr(layers, str(self))
        return ActivationFunction(*args, **kwargs)


TARGET_VERSION = ["mean", "independent", "max", "mean"]
LOSSES_VERSION = [1, 2, 3, 4, 5]
N_ACTIONS_PAD = 33  # sub actions per action space to pad to


class BDQConfig(BaseModel):
    n_outputs: int
    # list of sizes of hidden layers in the shared network module -- if this is an empty
    # list, then the learners across the branches are considered 'independent'
    shared_dim: List[int] = [512, 256]
    # list of sizes of hidden layers in the action-value/advantage branches -- currently
    # assumed the same across all such branches
    actions_dim: List[int] = [128]
    # list of sizes of hidden layers for the state-value branch
    state_dim: List[int] = [128]
    # the layer activation function
    activation: ActivationFunction = ActivationFunction.ReLU
    # number of branches
    n_action_branches: int
    # per branch
    n_actions: int
    aggregator: Aggregator = Aggregator.reduce_local_mean


class BDQBuilder:
    @staticmethod
    def create_shared_network(config: BDQConfig) -> Sequential:
        return BDQBuilder._create_nn(config.shared_dim, config.activation)

    @staticmethod
    def create_action_value_network(config: BDQConfig) -> Sequential:
        total_action_scores = []
        for action_stream in range(config.n_action_branches):
            action_branch = BDQBuilder._create_nn(config.actions_dim, config.activation)
            action_branch.add(
                layers.Dense(config.n_actions // config.n_action_branches)
            )

            # TODO
            # if (
            #     config.aggregator == Aggregator.reduce_local_mean
            #     or config.aggregator == Aggregator.reduce_local_max
            # ):
            #     action_branch.add(config.aggregator.create())

            total_action_scores.append(action_branch)

        return total_action_scores

    @staticmethod
    def create_state_value_network(config: BDQConfig) -> Sequential:
        state_branch = BDQBuilder._create_nn(config.state_dim, config.activation)
        state_branch.add(layers.Dense(1))

        return state_branch

    @staticmethod
    def _create_nn(hidden_neurons, activation):
        model = Sequential()
        for n_hidden in hidden_neurons:
            model.add(BDQBuilder._create_dense(n_hidden, activation))

        return model

    @staticmethod
    def _create_dense(n_hidden, activation):
        activation = activation.create()
        return layers.Dense(n_hidden, activation=activation)


class BDQNetwork(K.Model):
    def __init__(self, config: BDQConfig):
        super(BDQNetwork, self).__init__(name="BDQN")
        self.shared_net = BDQBuilder.create_shared_network(config)
        self.action_value_net = BDQBuilder.create_action_value_network(config)
        self.state_value_net = BDQBuilder.create_state_value_network(config)

    def call(self, x):
        shared = self.shared_net(x)
        action_values = self.action_value_net(shared)
        state_value = self.state_value_net(shared)

        dueling = [state_value + action_value for action_value in action_values]

        return dueling
