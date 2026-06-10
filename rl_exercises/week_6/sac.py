"""Soft Actor-Critic for Week 6 Level 3.

The implementation follows the main SAC ingredients from Haarnoja et al.:
an entropy-regularized stochastic actor, two Q-functions, target Q-networks,
experience replay, and automatic entropy-temperature tuning.
"""

from __future__ import annotations

from typing import Any

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_5.ddpg import ContinuousReplayBuffer
from rl_exercises.week_5.policy_gradient import set_seed

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    """Gaussian policy with tanh squashing into the environment action bounds."""

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.state_shape = state_space.shape
        self.action_shape = action_space.shape
        self.state_dim = int(np.prod(self.state_shape))
        self.action_dim = int(np.prod(self.action_shape))

        action_low = np.asarray(action_space.low, dtype=np.float32).reshape(1, -1)
        action_high = np.asarray(action_space.high, dtype=np.float32).reshape(1, -1)
        if not np.all(np.isfinite(action_low)) or not np.all(np.isfinite(action_high)):
            raise ValueError("SAC needs finite action bounds")

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, self.action_dim)
        self.log_std = nn.Linear(hidden_size, self.action_dim)
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def _as_batch(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(
                state, dtype=torch.float32, device=self.action_scale.device
            )
        else:
            state = state.to(device=self.action_scale.device, dtype=torch.float32)

        if state.dim() == len(self.state_shape):
            state = state.reshape(1, self.state_dim)
        else:
            state = state.reshape(state.shape[0], self.state_dim)
        return state

    def forward(
        self, state: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian mean and log standard deviation before squashing."""
        features = self.net(self._as_batch(state))
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a bounded action and return its log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = normal.log_prob(pre_tanh)
        correction = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)

        deterministic = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic


class QNetwork(nn.Module):
    """Q-network over state-action pairs."""

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.state_shape = state_space.shape
        self.action_shape = action_space.shape
        self.state_dim = int(np.prod(self.state_shape))
        self.action_dim = int(np.prod(self.action_shape))
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def _as_state_batch(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device=device, dtype=torch.float32)

        if state.dim() == len(self.state_shape):
            state = state.reshape(1, self.state_dim)
        else:
            state = state.reshape(state.shape[0], self.state_dim)
        return state

    def _as_action_batch(self, action: torch.Tensor | np.ndarray) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.float32, device=device)
        else:
            action = action.to(device=device, dtype=torch.float32)

        if action.dim() == len(self.action_shape):
            action = action.reshape(1, self.action_dim)
        else:
            action = action.reshape(action.shape[0], self.action_dim)
        return action

    def forward(
        self, state: torch.Tensor | np.ndarray, action: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """Estimate Q(s, a)."""
        state = self._as_state_batch(state)
        action = self._as_action_batch(action)
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent(AbstractAgent):
    """Soft Actor-Critic agent for continuous-control environments."""

    def __init__(
        self,
        env: gym.Env,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        learning_starts: int = 1000,
        hidden_size: int = 256,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("SAC needs a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("SAC needs a continuous Box action space")

        set_seed(env, seed)
        self.env = env
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.actor = SquashedGaussianActor(
            env.observation_space, env.action_space, hidden_size
        ).to(self.device)
        self.q1 = QNetwork(env.observation_space, env.action_space, hidden_size).to(
            self.device
        )
        self.q2 = QNetwork(env.observation_space, env.action_space, hidden_size).to(
            self.device
        )
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(np.prod(env.action_space.shape))

        self.buffer = ContinuousReplayBuffer(buffer_capacity)
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32)
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy-temperature parameter."""
        return self.log_alpha.exp()

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(
                target.parameters(), source.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def predict_action(
        self,
        state: np.ndarray,
        info: dict[str, Any] | None = None,
        evaluate: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select a stochastic action for training or the mean action for eval."""
        del info
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            action, _, deterministic = self.actor.sample(state)
            selected = deterministic if evaluate else action
        if was_training:
            self.actor.train()
        clipped = np.clip(selected.cpu().numpy()[0], self.action_low, self.action_high)
        return clipped.astype(np.float32), {}

    def update_agent(
        self,
        training_batch: list[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict[str, Any]]
        ],
    ) -> dict[str, float]:
        """Update actor, critics, target critics, and entropy temperature."""
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        states_t = torch.as_tensor(
            np.asarray(states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions_t = torch.as_tensor(
            np.asarray(actions, dtype=np.float32).reshape(len(training_batch), -1),
            dtype=torch.float32,
            device=self.device,
        )
        rewards_t = torch.as_tensor(
            np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        next_states_t = torch.as_tensor(
            np.asarray(next_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dones_t = torch.as_tensor(
            np.asarray(dones, dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states_t)
            next_q = torch.min(
                self.q1_target(next_states_t, next_actions),
                self.q2_target(next_states_t, next_actions),
            )
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * (
                next_q - self.alpha.detach() * next_log_probs
            )

        q1_loss = F.mse_loss(self.q1(states_t, actions_t), target_q)
        q2_loss = F.mse_loss(self.q2(states_t, actions_t), target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        sampled_actions, log_probs, _ = self.actor.sample(states_t)
        min_q = torch.min(
            self.q1(states_t, sampled_actions),
            self.q2(states_t, sampled_actions),
        )
        actor_loss = (self.alpha.detach() * log_probs - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)

        return {
            "actor_loss": float(actor_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
        }

    def train(
        self,
        num_frames: int,
        eval_env: gym.Env | None = None,
        eval_interval: int = 1000,
        eval_episodes: int = 5,
    ) -> dict[str, list[float]]:
        """Train for a fixed number of environment steps and return a history."""
        eval_env = eval_env or gym.make(self.env.spec.id)
        state, _ = self.env.reset()
        done = False
        episode_return = 0.0
        recent_returns: list[float] = []
        history: dict[str, list[float]] = {
            "frames": [],
            "mean_reward_10": [],
            "eval_mean": [],
            "eval_std": [],
            "actor_loss": [],
            "q1_loss": [],
            "q2_loss": [],
            "alpha": [],
        }
        latest_losses = {
            "actor_loss": np.nan,
            "q1_loss": np.nan,
            "q2_loss": np.nan,
            "alpha": float(self.alpha.detach().item()),
        }

        for frame in range(1, num_frames + 1):
            if frame <= self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.buffer.add(state, action, float(reward), next_state, done, info)
            episode_return += float(reward)
            state = next_state
            self.total_steps += 1

            if len(self.buffer) >= self.batch_size and frame > self.learning_starts:
                batch = self.buffer.sample(self.batch_size)
                losses = self.update_agent(batch)
                latest_losses = {
                    "actor_loss": losses["actor_loss"],
                    "q1_loss": losses["q1_loss"],
                    "q2_loss": losses["q2_loss"],
                    "alpha": losses["alpha"],
                }

            if done:
                recent_returns.append(episode_return)
                state, _ = self.env.reset()
                episode_return = 0.0
                done = False

            if frame % eval_interval == 0:
                eval_mean, eval_std = self.evaluate(eval_env, eval_episodes)
                history["frames"].append(frame)
                history["mean_reward_10"].append(
                    float(np.mean(recent_returns[-10:])) if recent_returns else np.nan
                )
                history["eval_mean"].append(eval_mean)
                history["eval_std"].append(eval_std)
                history["actor_loss"].append(float(latest_losses["actor_loss"]))
                history["q1_loss"].append(float(latest_losses["q1_loss"]))
                history["q2_loss"].append(float(latest_losses["q2_loss"]))
                history["alpha"].append(float(latest_losses["alpha"]))

        return history

    def evaluate(self, eval_env: gym.Env, num_episodes: int = 5) -> tuple[float, float]:
        """Evaluate the deterministic mean action."""
        returns = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            total_return = 0.0
            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_return += float(reward)
            returns.append(total_return)
        return float(np.mean(returns)), float(np.std(returns))
