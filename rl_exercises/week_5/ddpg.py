"""Deep Deterministic Policy Gradient for Week 5 Level 3.

This follows the main stability tricks from Lillicrap et al.:
experience replay, target actor/critic networks, soft target updates, and
temporally correlated exploration noise.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_5.policy_gradient import set_seed


class ContinuousReplayBuffer:
    """Uniform replay buffer that stores continuous actions."""

    def __init__(self, capacity: int) -> None:
        assert capacity > 0, "Replay buffer capacity must be positive"
        self.capacity = int(capacity)
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[bool] = []
        self.infos: list[dict[str, Any]] = []

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self.states)

    def _pop_oldest(self) -> None:
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.next_states.pop(0)
        self.dones.pop(0)
        self.infos.pop(0)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray | list[float] | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Store one transition."""
        if len(self.states) >= self.capacity:
            self._pop_oldest()

        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(bool(done))
        self.infos.append(dict(info or {}))

    def sample(
        self, batch_size: int
    ) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict[str, Any]]]:
        """Sample a minibatch uniformly without replacement."""
        if batch_size > len(self):
            raise ValueError("Cannot sample more transitions than are stored")

        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        return [
            (
                self.states[int(index)],
                self.actions[int(index)],
                self.rewards[int(index)],
                self.next_states[int(index)],
                self.dones[int(index)],
                self.infos[int(index)],
            )
            for index in indices
        ]


class Actor(nn.Module):
    """Deterministic policy network mapping states to bounded actions."""

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
            raise ValueError("DDPG needs finite action bounds")

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Return an action in the environment's Box bounds."""
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

        return self.net(state) * self.action_scale + self.action_bias


class Critic(nn.Module):
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

    def forward(
        self,
        state: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Estimate Q(s, a)."""
        device = next(self.parameters()).device
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device=device, dtype=torch.float32)
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.float32, device=device)
        else:
            action = action.to(device=device, dtype=torch.float32)

        if state.dim() == len(self.state_shape):
            state = state.reshape(1, self.state_dim)
        else:
            state = state.reshape(state.shape[0], self.state_dim)

        if action.dim() == len(self.action_shape):
            action = action.reshape(1, self.action_dim)
        else:
            action = action.reshape(action.shape[0], self.action_dim)

        return self.net(torch.cat([state, action], dim=-1))


@dataclass
class OrnsteinUhlenbeckNoise:
    """Temporally correlated action noise used in the DDPG paper."""

    size: int
    mu: float = 0.0
    theta: float = 0.15
    sigma: float = 0.2
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        """Reset the process at the start of an episode."""
        self.x_prev = np.full(self.size, self.mu, dtype=np.float32)

    def sample(self) -> np.ndarray:
        """Draw one noise vector."""
        random_term = self.rng.normal(size=self.size).astype(np.float32)
        dx = (
            self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * random_term
        )
        self.x_prev = self.x_prev + dx
        return self.x_prev.copy()


class DDPGAgent(AbstractAgent):
    """Off-policy actor-critic agent for continuous control."""

    def __init__(
        self,
        env: gym.Env,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        learning_starts: int = 1000,
        hidden_size: int = 256,
        noise_sigma: float = 0.2,
        noise_theta: float = 0.15,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("DDPG needs a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("DDPG needs a continuous Box action space")

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

        self.actor = Actor(env.observation_space, env.action_space, hidden_size).to(
            self.device
        )
        self.actor_target = Actor(
            env.observation_space, env.action_space, hidden_size
        ).to(self.device)
        self.critic = Critic(env.observation_space, env.action_space, hidden_size).to(
            self.device
        )
        self.critic_target = Critic(
            env.observation_space, env.action_space, hidden_size
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = ContinuousReplayBuffer(buffer_capacity)

        self.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32)
        self.action_shape = env.action_space.shape
        self.action_dim = int(np.prod(self.action_shape))
        self.noise = OrnsteinUhlenbeckNoise(
            self.action_dim, theta=noise_theta, sigma=noise_sigma, seed=seed
        )

        self.total_steps = 0
        self.update_steps = 0

    def reset_noise(self) -> None:
        """Reset exploration noise for a new episode."""
        self.noise.reset()

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Move target parameters slowly towards source parameters."""
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
        """Select a deterministic action, with OU noise during training."""
        del info
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        if was_training:
            self.actor.train()

        if not evaluate:
            action = action + self.noise.sample()

        clipped = np.clip(
            action.reshape(self.action_shape), self.action_low, self.action_high
        )
        return clipped.astype(np.float32), {}

    def update_agent(
        self,
        training_batch: list[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict[str, Any]]
        ],
    ) -> dict[str, float]:
        """Update critic and actor from a replay minibatch."""
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
            np.asarray(rewards, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        next_states_t = torch.as_tensor(
            np.asarray(next_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dones_t = torch.as_tensor(
            np.asarray(dones, dtype=np.float32), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            next_q = self.critic_target(next_states_t, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        current_q = self.critic(states_t, actions_t).squeeze(1)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)
        self.update_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def evaluate(self, eval_env: gym.Env, num_episodes: int = 5) -> tuple[float, float]:
        """Evaluate the deterministic actor."""
        returns = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            total_return = 0.0
            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_return += float(reward)
                state = next_state
            returns.append(total_return)

        mean = float(np.mean(returns)) if returns else 0.0
        std = float(np.std(returns)) if returns else 0.0
        return mean, std

    def save(self, path: str) -> None:
        """Save model and optimizer state."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_steps": self.total_steps,
                "update_steps": self.update_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.update_steps = int(checkpoint.get("update_steps", 0))

    def train(
        self,
        num_frames: int,
        eval_env: gym.Env | None = None,
        eval_interval: int = 1000,
        eval_episodes: int = 5,
    ) -> dict[str, list[float]]:
        """Train for a fixed number of environment frames."""
        state, _ = self.env.reset()
        self.reset_noise()
        episode_return = 0.0
        recent_returns: list[float] = []
        history: dict[str, list[float]] = {
            "frames": [],
            "mean_reward_10": [],
            "eval_mean": [],
            "eval_std": [],
            "critic_loss": [],
            "actor_loss": [],
        }
        latest_losses = {"critic_loss": np.nan, "actor_loss": np.nan}

        for frame in range(1, num_frames + 1):
            if self.total_steps < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.buffer.add(state, action, float(reward), next_state, done, info)
            self.total_steps += 1
            episode_return += float(reward)
            state = next_state

            if (
                len(self.buffer) >= self.batch_size
                and self.total_steps >= self.learning_starts
            ):
                latest_losses = self.update_agent(self.buffer.sample(self.batch_size))

            if done:
                recent_returns.append(episode_return)
                episode_return = 0.0
                state, _ = self.env.reset()
                self.reset_noise()

            if eval_interval > 0 and frame % eval_interval == 0:
                history["frames"].append(float(frame))
                mean_10 = (
                    float(np.mean(recent_returns[-10:])) if recent_returns else 0.0
                )
                history["mean_reward_10"].append(mean_10)
                history["critic_loss"].append(float(latest_losses["critic_loss"]))
                history["actor_loss"].append(float(latest_losses["actor_loss"]))
                if eval_env is not None:
                    eval_mean, eval_std = self.evaluate(eval_env, eval_episodes)
                else:
                    eval_mean, eval_std = mean_10, 0.0
                history["eval_mean"].append(eval_mean)
                history["eval_std"].append(eval_std)

        return history
