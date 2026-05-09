"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import csv
import sys
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import PrioritizedReplayBuffer, ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q‐Learning agent with ε‐greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        double_dqn: bool = False,
        prioritized_replay: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 10000,
        per_epsilon: float = 1e-6,
        device: str | None = None,
    ) -> None:
        """
        Initialize replay buffer, Q‐networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini‐batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target‐network syncs.
        seed : int
            RNG seed.
        hidden_dim : int
            Width of hidden layers in the Q-network.
        hidden_layers : int
            Number of hidden layers in the Q-network.
        double_dqn : bool
            If True, use online-network action selection and target-network
            evaluation for the bootstrap target.
        prioritized_replay : bool
            If True, use prioritized experience replay.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        assert hasattr(env.observation_space, "shape"), "DQN needs vector observations"
        assert hasattr(env.action_space, "n"), "DQN needs a discrete action space"

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q‐network and frozen target
        self.q = QNetwork(obs_dim, n_actions, hidden_dim, hidden_layers).to(self.device)
        self.target_q = QNetwork(obs_dim, n_actions, hidden_dim, hidden_layers).to(
            self.device
        )
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        if prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                buffer_capacity,
                alpha=per_alpha,
                beta=per_beta_start,
                priority_epsilon=per_epsilon,
            )
        else:
            self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.per_epsilon = per_epsilon

        self.total_steps = 0  # environment frames, used for epsilon decay
        self.update_steps = 0  # gradient updates, used for target sync
        self.training_history: Dict[str, List[float]] = {
            "frames": [],
            "mean_reward_10": [],
            "episode_rewards": [],
            "losses": [],
        }

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        if self.epsilon_decay <= 0:
            return self.epsilon_final
        return float(
            self.epsilon_final
            + (self.epsilon_start - self.epsilon_final)
            * np.exp(-self.total_steps / self.epsilon_decay)
        )

    def _per_beta(self) -> float:
        """Linearly anneal PER beta towards 1.0."""
        if self.per_beta_frames <= 0:
            return 1.0
        progress = min(1.0, self.total_steps / self.per_beta_frames)
        return float(self.per_beta_start + progress * (1.0 - self.per_beta_start))

    def _greedy_action(self, state: np.ndarray) -> int:
        """Select the action with the largest predicted Q-value."""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def predict_action(
        self,
        state: np.ndarray,
        info: Dict[str, Any] | None = None,
        evaluate: bool = False,
    ) -> int:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        """
        if evaluate or self.rng.random() >= self.epsilon():
            return self._greedy_action(state)
        return int(self.env.action_space.sample())

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "target_parameters": self.target_q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "update_steps": self.update_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q.load_state_dict(checkpoint["parameters"])
        self.target_q.load_state_dict(
            checkpoint.get("target_parameters", checkpoint["parameters"])
        )
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.update_steps = int(checkpoint.get("update_steps", 0))

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        infos = [transition[5] for transition in training_batch]
        s = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        a = a.unsqueeze(1)
        r = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        s_next = torch.as_tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        done_mask = torch.as_tensor(
            np.array(dones), dtype=torch.float32, device=self.device
        )
        sampling_weights = torch.as_tensor(
            [info.get("sampling_weight", 1.0) for info in infos],
            dtype=torch.float32,
            device=self.device,
        )

        # current Q estimates for taken actions
        pred = self.q(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q(s_next).argmax(dim=1, keepdim=True)
                next_q = self.target_q(s_next).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_q(s_next).max(dim=1).values
            target = r + self.gamma * next_q * (1.0 - done_mask)

        td_errors = target - pred
        loss = (sampling_weights * td_errors.pow(2)).mean()

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        if hasattr(self.buffer, "update_priorities"):
            indices = [info["buffer_index"] for info in infos if "buffer_index" in info]
            if indices:
                priorities = td_errors.detach().abs().cpu().numpy()
                self.buffer.update_priorities(  # type: ignore[attr-defined]
                    indices, priorities
                )

        return float(loss.item())

    def train(
        self,
        num_frames: int,
        eval_interval: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        verbose : bool
            If True, print training progress.
        """
        state, _ = self.env.reset(seed=None)
        ep_reward = 0.0
        recent_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            self.total_steps += 1

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                if isinstance(self.buffer, PrioritizedReplayBuffer):
                    self.buffer.beta = self._per_beta()
                batch = self.buffer.sample(self.batch_size)
                loss = self.update_agent(batch)
                self.training_history["losses"].append(loss)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                self.training_history["episode_rewards"].append(ep_reward)
                ep_reward = 0.0
                # logging
                if verbose and len(recent_rewards) % 10 == 0:
                    avg = float(np.mean(recent_rewards[-10:]))
                    msg = (
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, "
                        f"ε={self.epsilon():.3f}"
                    )
                    print(msg)

            if eval_interval > 0 and frame % eval_interval == 0:
                avg = float(np.mean(recent_rewards[-10:])) if recent_rewards else 0.0
                self.training_history["frames"].append(float(frame))
                self.training_history["mean_reward_10"].append(avg)

        if verbose:
            print("Training complete.")
        return self.training_history


def save_training_history(history: Dict[str, List[float]], path: str | Path) -> None:
    """Save frame-based training statistics to a CSV file."""
    path = Path(path)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["frames", "mean_reward_10"])
        writer.writeheader()
        for frame, reward in zip(history["frames"], history["mean_reward_10"]):
            writer.writerow({"frames": int(frame), "mean_reward_10": reward})


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 2) map config to agent kwargs
    agent_kwargs = dict(
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
        hidden_dim=cfg.agent.get("hidden_dim", 64),
        hidden_layers=cfg.agent.get("hidden_layers", 2),
        double_dqn=cfg.agent.get("double_dqn", False),
        prioritized_replay=cfg.agent.get("prioritized_replay", False),
        per_alpha=cfg.agent.get("per_alpha", 0.6),
        per_beta_start=cfg.agent.get("per_beta_start", 0.4),
        per_beta_frames=cfg.agent.get("per_beta_frames", 10000),
        per_epsilon=cfg.agent.get("per_epsilon", 1e-6),
    )

    # 3) instantiate and train
    agent = DQNAgent(env, **agent_kwargs)
    history = agent.train(cfg.train.num_frames, cfg.train.eval_interval)
    save_training_history(history, Path.cwd() / "train_rewards.csv")
    agent.save(str(Path.cwd() / "dqn_model.pt"))


if __name__ == "__main__":
    main()
