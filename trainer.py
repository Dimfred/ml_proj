from dataclasses import dataclass

from typing import Callable, Any
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensortrade.agents import ReplayMemory
from tensortrade.env.generic.environment import TradingEnv

from tabulate import tabulate
from collections import namedtuple
import time

from dimfred import Stopwatch


@dataclass
class Trainer:
    env: TradingEnv
    # create_env: Callable[[], None] #: TradingEnv

    weights_path: Path

    # maximum number of episodes to run
    n_episodes: int = 1
    batch_size: int = 128
    discount_factor: float = 0.9999
    learning_rate: float = 0.0001

    # starting point for the epsilon gready approach for action picking
    eps_start: float = 0.9
    # end point for the epsilon gready approach for action picking
    eps_end: float = 0.05
    # decay of the value
    eps_decay_steps: int = 200

    # after how much steps the network should be updated
    update_target_every: int = 1000
    memory_capacity: int = 1000

    render_interval: int = 500
    on_render: Callable[[TradingEnv], None] = lambda env: env.renderer.fig.write_html(
        file="charts/chart.html", include_plotlyjs="cdn"
    )

    save_every: int = 0
    save_path: str = ""

    TransitionType: namedtuple = None

    def __post_init__(self):
        self.memory = ReplayMemory(
            self.memory_capacity, transition_type=self.TransitionType
        )

    def train_step(self, model):
        decay = np.exp(-self.total_steps / self.eps_decay_steps)
        # calculates the action picking threshold
        # aka. with which probability an action is taken at random
        # (space exploration)
        threshold = self.eps_end + (self.eps_start - self.eps_end) * decay
        action = model.get_action(self.state, threshold=threshold)
        # execute an action, retrieve reward for that step and the next step
        next_state, reward, done, _ = self.env.step(action)
        # store the values in memory
        self.memory.push(self.state, action, reward, next_state, done)

        return reward, next_state, threshold

    def test_step(self, model):
        action = model.get_action(self.state, threshold=0)
        next_state, reward, done, _ = self.env.step(action)

        return reward, next_state

    def train(
        self,
        model,
        n_steps: int = 1,
    ):
        start_time = time.perf_counter()
        self.total_steps, self.total_reward = -1, 0
        for episode in range(self.n_episodes):
            # reset the env at each new episode
            self.state = self.env.reset()
            self.env.renderer.min_step = 0

            render_sw = Stopwatch()
            for step in range(n_steps):
                self.total_steps += 1
                # make a step in the environment
                reward, next_state, threshold = self.train_step(model)
                self.state = next_state
                self.total_reward += reward

                # first fill the memory before we do any updates
                if len(self.memory) < self.batch_size:
                    continue

                model._apply_gradient_descent(
                    self.memory,
                    self.batch_size,
                    self.learning_rate,
                    self.discount_factor,
                )

                # render the env
                if step % self.render_interval == 0:
                    self.env.render(
                        episode=episode, max_episodes=self.n_episodes, max_steps=n_steps
                    )
                    self.on_render(episode, step, self.env)
                    # fmt: off
                    pretty = [
                        ["Episode", "Step", "TotalSteps", "Rendering", "Time", "Thresh"],
                        [episode, f"{step} / {n_steps}", self.total_steps, render_sw(), f"{(time.perf_counter() - start_time) / 3600:.2f}h", threshold],
                    ]
                    # fmt: on
                    print(tabulate(pretty))

                # update the model
                if self.total_steps % self.update_target_every == 0:
                    model.target_network = tf.keras.models.clone_model(
                        model.policy_network
                    )
                    model.target_network.trainable = False

            self.on_render(episode, step, self.env)
            model.save(self.weights_path, episode=episode)

        # calculate the mean reward
        mean_reward = self.total_reward / self.total_steps

        return self.total_reward, mean_reward

    def test(
        self,
        model,
        n_steps: int = 1,
    ):
        model.target_network.trainable = False

        start_time = time.perf_counter()
        self.total_steps, self.total_reward = -1, 0
        self.state = self.env.reset()
        self.env.renderer.min_step = 0

        render_sw = Stopwatch()
        for step in range(n_steps):
            self.total_steps += 1
            reward, next_state = self.test_step(model)
            self.state = next_state
            self.total_reward += reward

            if step % self.render_interval == 0:
                self.env.render(episode=1, max_episodes=1, max_steps=n_steps)
                self.on_render(1, step, self.env)
                # fmt: off
                pretty = [
                    ["Episode", "Step", "TotalSteps", "Rendering", "Time"],
                    [1, f"{step} / {n_steps}", self.total_steps, render_sw(), f"{(time.perf_counter() - start_time) / 3600:.2f}h"],
                ]
                # fmt: on
                print(tabulate(pretty))

        self.on_render(1, step, self.env)
        mean_reward = self.total_reward / self.total_steps

        return self.total_reward, mean_reward
