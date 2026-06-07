"""Experiment helpers for Week 5 REINFORCE tasks.

Examples
--------
Run the Level 2 runs used for the observations:

    python -m rl_exercises.week_5.experiments --mode l2

Run the Level 3 DDPG comparison:

    python -m rl_exercises.week_5.experiments --mode l3
"""

from __future__ import annotations

from typing import Any

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rl_exercises.week_5.ddpg import DDPGAgent
from rl_exercises.week_5.policy_gradient import REINFORCEAgent, set_seed

L2_CONFIGS: dict[str, dict[str, Any]] = {
    "cartpole_cap50": {
        "env_name": "CartPole-v1",
        "max_episode_steps": 50,
        "seeds": [0, 1, 2],
        "episodes": 250,
        "eval_every": 50,
        "eval_episodes": 5,
        "lr": 1e-2,
        "hidden_size": 64,
    },
    "cartpole_cap200": {
        "env_name": "CartPole-v1",
        "max_episode_steps": 200,
        "seeds": [0, 1, 2],
        "episodes": 250,
        "eval_every": 50,
        "eval_episodes": 5,
        "lr": 1e-2,
        "hidden_size": 64,
    },
    "cartpole_cap500": {
        "env_name": "CartPole-v1",
        "max_episode_steps": 500,
        "seeds": [0, 1, 2],
        "episodes": 250,
        "eval_every": 50,
        "eval_episodes": 5,
        "lr": 1e-2,
        "hidden_size": 64,
    },
    "lunar_lander_v3": {
        "env_name": "LunarLander-v3",
        "max_episode_steps": None,
        "seeds": [0],
        "episodes": 120,
        "eval_every": 30,
        "eval_episodes": 3,
        "lr": 1e-3,
        "hidden_size": 128,
    },
}

FIELDNAMES = [
    "config",
    "seed",
    "episode",
    "train_mean_last_10",
    "eval_mean",
    "eval_std",
    "loss",
]

L3_DDPG_CONFIG: dict[str, Any] = {
    "env_name": "Pendulum-v1",
    "max_episode_steps": 200,
    "seeds": [0],
    "frames": 12000,
    "eval_every": 3000,
    "eval_episodes": 3,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_capacity": 50000,
    "batch_size": 64,
    "learning_starts": 256,
    "hidden_size": 128,
    "noise_sigma": 0.2,
    "noise_theta": 0.15,
}

L3_REINFORCE_CONFIG: dict[str, Any] = {
    "env_name": "Pendulum-v1",
    "max_episode_steps": 200,
    "seeds": [0],
    "episodes": 60,
    "eval_every": 15,
    "eval_episodes": 3,
    "lr": 1e-3,
    "hidden_size": 128,
    "action_bins": 9,
}

L3_FIELDNAMES = [
    "agent",
    "config",
    "seed",
    "step",
    "episode",
    "train_mean_last_10",
    "eval_mean",
    "eval_std",
    "loss",
    "critic_loss",
    "actor_loss",
]


def make_env(env_name: str, seed: int, max_episode_steps: int | None) -> gym.Env:
    """Create and seed a Gymnasium environment."""
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_name, **kwargs)
    set_seed(env, seed)
    return env


class DiscretizeOneDimensionalAction(gym.ActionWrapper):
    """Expose a one-dimensional Box action space as evenly spaced actions."""

    def __init__(self, env: gym.Env, bins: int = 9) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("Can only discretize a Box action space")
        if int(np.prod(env.action_space.shape)) != 1:
            raise ValueError("This wrapper only supports one-dimensional actions")
        if bins < 2:
            raise ValueError("Need at least two action bins")

        self.actions = np.linspace(
            float(env.action_space.low.reshape(-1)[0]),
            float(env.action_space.high.reshape(-1)[0]),
            bins,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(bins)

    def action(self, action: int) -> np.ndarray:
        """Map a discrete action index to the wrapped continuous action."""
        return np.asarray([self.actions[int(action)]], dtype=np.float32)


def make_discrete_action_env(
    env_name: str,
    seed: int,
    max_episode_steps: int | None,
    action_bins: int,
) -> gym.Env:
    """Create Pendulum with a discrete action wrapper for REINFORCE."""
    env = make_env(env_name, seed, max_episode_steps)
    wrapped = DiscretizeOneDimensionalAction(env, action_bins)
    set_seed(wrapped, seed)
    return wrapped


def rollout_episode(agent: REINFORCEAgent, env: gym.Env) -> tuple[float, float]:
    """Run one training episode and update the policy from that trajectory."""
    state, _ = env.reset()
    done = False
    batch = []
    total_return = 0.0

    while not done:
        action, info = agent.predict_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        batch.append((state, action, float(reward), next_state, done, info))
        total_return += float(reward)
        state = next_state

    loss = agent.update_agent(batch)
    return total_return, loss


def train_config(config_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Train one experiment configuration and collect evaluation rows."""
    rows = []
    for seed in config["seeds"]:
        print(f"Running {config_name}, seed={seed}")
        env = make_env(config["env_name"], seed, config["max_episode_steps"])
        eval_env = make_env(
            config["env_name"],
            seed + 1000,
            config["max_episode_steps"],
        )
        agent = REINFORCEAgent(
            env,
            lr=config["lr"],
            gamma=0.99,
            seed=seed,
            hidden_size=config["hidden_size"],
        )
        recent_returns = []

        for episode in range(1, config["episodes"] + 1):
            total_return, loss = rollout_episode(agent, env)
            recent_returns.append(total_return)

            if episode % config["eval_every"] == 0:
                eval_mean, eval_std = agent.evaluate(
                    eval_env,
                    num_episodes=config["eval_episodes"],
                )
                rows.append(
                    {
                        "config": config_name,
                        "seed": seed,
                        "episode": episode,
                        "train_mean_last_10": float(np.mean(recent_returns[-10:])),
                        "eval_mean": eval_mean,
                        "eval_std": eval_std,
                        "loss": float(loss),
                    }
                )

        env.close()
        eval_env.close()

    return rows


def save_rows(
    rows: list[dict[str, Any]], path: Path, fieldnames: list[str] = FIELDNAMES
) -> None:
    """Save experiment rows as CSV."""
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def plot_l2(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot mean evaluation return per configuration."""
    configs = sorted({row["config"] for row in summary_rows})
    plt.figure(figsize=(10, 6))

    for config in configs:
        rows = [row for row in summary_rows if row["config"] == config]
        episodes = sorted({int(row["episode"]) for row in rows})
        means = []
        stds = []
        for episode in episodes:
            values = [
                float(row["eval_mean"])
                for row in rows
                if int(row["episode"]) == episode
            ]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))

        means_arr = np.asarray(means)
        stds_arr = np.asarray(stds)
        plt.plot(episodes, means_arr, label=config)
        plt.fill_between(
            episodes, means_arr - stds_arr, means_arr + stds_arr, alpha=0.18
        )

    plt.xlabel("Episode")
    plt.ylabel("Evaluation return")
    plt.title("Week 5 Level 2 REINFORCE runs")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def run_l2(output_dir: Path) -> None:
    """Run all Level 2 experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for config_name, config in L2_CONFIGS.items():
        rows = train_config(config_name, config)
        save_rows(rows, output_dir / f"{config_name}_training.csv")
        summary_rows.extend(rows)

    save_rows(summary_rows, output_dir / "summary.csv")
    plot_l2(summary_rows, output_dir / "l2_training_curves.png")


def train_ddpg_l3(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Train DDPG on the continuous Pendulum action space."""
    rows = []
    for seed in config["seeds"]:
        print(f"Running ddpg_pendulum, seed={seed}")
        env = make_env(config["env_name"], seed, config["max_episode_steps"])
        eval_env = make_env(
            config["env_name"],
            seed + 1000,
            config["max_episode_steps"],
        )
        agent = DDPGAgent(
            env,
            actor_lr=config["actor_lr"],
            critic_lr=config["critic_lr"],
            gamma=config["gamma"],
            tau=config["tau"],
            buffer_capacity=config["buffer_capacity"],
            batch_size=config["batch_size"],
            learning_starts=config["learning_starts"],
            hidden_size=config["hidden_size"],
            noise_sigma=config["noise_sigma"],
            noise_theta=config["noise_theta"],
            seed=seed,
        )
        history = agent.train(
            num_frames=config["frames"],
            eval_env=eval_env,
            eval_interval=config["eval_every"],
            eval_episodes=config["eval_episodes"],
        )

        for index, step in enumerate(history["frames"]):
            rows.append(
                {
                    "agent": "DDPG",
                    "config": "ddpg_pendulum",
                    "seed": seed,
                    "step": int(step),
                    "episode": "",
                    "train_mean_last_10": history["mean_reward_10"][index],
                    "eval_mean": history["eval_mean"][index],
                    "eval_std": history["eval_std"][index],
                    "loss": np.nan,
                    "critic_loss": history["critic_loss"][index],
                    "actor_loss": history["actor_loss"][index],
                }
            )

        env.close()
        eval_env.close()

    return rows


def train_reinforce_l3(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Train a discretized-action REINFORCE baseline on Pendulum."""
    rows = []
    for seed in config["seeds"]:
        print(f"Running reinforce_discrete_pendulum, seed={seed}")
        env = make_discrete_action_env(
            config["env_name"],
            seed,
            config["max_episode_steps"],
            config["action_bins"],
        )
        eval_env = make_discrete_action_env(
            config["env_name"],
            seed + 1000,
            config["max_episode_steps"],
            config["action_bins"],
        )
        agent = REINFORCEAgent(
            env,
            lr=config["lr"],
            gamma=0.99,
            seed=seed,
            hidden_size=config["hidden_size"],
        )
        recent_returns = []
        total_steps = 0

        for episode in range(1, config["episodes"] + 1):
            state, _ = env.reset()
            done = False
            batch = []
            episode_return = 0.0

            while not done:
                action, info = agent.predict_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                batch.append((state, action, float(reward), next_state, done, info))
                episode_return += float(reward)
                total_steps += 1
                state = next_state

            loss = agent.update_agent(batch)
            recent_returns.append(episode_return)

            if episode % config["eval_every"] == 0:
                eval_mean, eval_std = agent.evaluate(
                    eval_env,
                    num_episodes=config["eval_episodes"],
                )
                rows.append(
                    {
                        "agent": "REINFORCE-discrete",
                        "config": "reinforce_discrete_pendulum",
                        "seed": seed,
                        "step": total_steps,
                        "episode": episode,
                        "train_mean_last_10": float(np.mean(recent_returns[-10:])),
                        "eval_mean": eval_mean,
                        "eval_std": eval_std,
                        "loss": float(loss),
                        "critic_loss": np.nan,
                        "actor_loss": np.nan,
                    }
                )

        env.close()
        eval_env.close()

    return rows


def plot_l3(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot the Level 3 DDPG and REINFORCE comparison."""
    agents = sorted({row["agent"] for row in summary_rows})
    plt.figure(figsize=(10, 6))

    for agent_name in agents:
        rows = [row for row in summary_rows if row["agent"] == agent_name]
        steps = sorted({int(row["step"]) for row in rows})
        means = []
        stds = []
        for step in steps:
            values = [
                float(row["eval_mean"]) for row in rows if int(row["step"]) == step
            ]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))

        means_arr = np.asarray(means)
        stds_arr = np.asarray(stds)
        plt.plot(steps, means_arr, label=agent_name)
        plt.fill_between(steps, means_arr - stds_arr, means_arr + stds_arr, alpha=0.18)

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("Week 5 Level 3: DDPG vs REINFORCE on Pendulum-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def _latest_eval(summary_rows: list[dict[str, Any]], agent_name: str) -> float | None:
    rows = [row for row in summary_rows if row["agent"] == agent_name]
    if not rows:
        return None
    latest_step = max(int(row["step"]) for row in rows)
    latest_values = [
        float(row["eval_mean"]) for row in rows if int(row["step"]) == latest_step
    ]
    return float(np.mean(latest_values))


def write_l3_observations(summary_rows: list[dict[str, Any]], path: Path) -> None:
    """Write short Level 3 notes connected to the DDPG paper."""
    ddpg_final = _latest_eval(summary_rows, "DDPG")
    reinforce_final = _latest_eval(summary_rows, "REINFORCE-discrete")
    ddpg_text = "n/a" if ddpg_final is None else f"{ddpg_final:.1f}"
    reinforce_text = "n/a" if reinforce_final is None else f"{reinforce_final:.1f}"

    text = f"""Week 5 - Level 3 observations
=============================

Setup
-----
We compared DDPG and REINFORCE on Pendulum-v1.  Pendulum has a continuous
one-dimensional torque action in [-2, 2], so DDPG can act in the native action
space.  The existing REINFORCE implementation is discrete, so the baseline uses
a wrapper with 9 evenly spaced torque actions.

The run is intentionally small so it can be reproduced quickly:
- DDPG: {L3_DDPG_CONFIG["frames"]} environment steps, actor_lr={L3_DDPG_CONFIG["actor_lr"]}, critic_lr={L3_DDPG_CONFIG["critic_lr"]}, tau={L3_DDPG_CONFIG["tau"]}
- REINFORCE: {L3_REINFORCE_CONFIG["episodes"]} episodes with the same episode cap and 9 action bins

Because Pendulum-v1 uses 200-step episodes here, both agents are compared over
12000 environment steps.

Paper context
-------------
Lillicrap et al. describe DDPG as a deterministic actor-critic method that
brings the DQN stability tricks into continuous control.  The important
changes compared with a naive neural DPG implementation are:
- a replay buffer, which breaks up correlations between consecutive samples and
  allows off-policy minibatch updates;
- target actor and target critic networks, which make Bellman targets change
  slowly;
- soft target updates, theta_target <- tau theta + (1 - tau) theta_target,
  instead of hard copies;
- exploration by adding a noise process to the deterministic actor.  We use
  Ornstein-Uhlenbeck noise here, as in the paper's physical-control setup.

The paper also uses batch normalization to make learning less sensitive to
different observation scales across physical-control tasks.  This small
exercise implementation leaves that part out, but it keeps the main DQN-style
stability changes: replay and slowly moving target networks.

Comparison
----------
Final evaluation mean in this small run:
- DDPG: {ddpg_text}
- REINFORCE with discretized actions: {reinforce_text}

The comparison is not a claim that the short DDPG run has solved Pendulum.  It
is mainly a structural comparison.  REINFORCE uses full-episode Monte Carlo
returns and throws away each trajectory after one update, so the gradient is
high variance.  DDPG uses replayed transitions and bootstrapped critic targets,
which is closer to the DQN intuition from Week 4.

The action space also matters.  Discretizing Pendulum lets the old REINFORCE
agent run, but it removes the continuous structure that DDPG is designed to
exploit.  This connects directly to the paper's motivation: discretizing
continuous controls scales badly and can discard useful action geometry.

To reproduce the plot and CSV files:

.venv/bin/python -m rl_exercises.week_5.experiments --mode l3
"""
    path.write_text(text)
    print(f"Saved {path}")


def run_l3(output_dir: Path) -> None:
    """Run the Level 3 DDPG comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ddpg_rows = train_ddpg_l3(L3_DDPG_CONFIG)
    reinforce_rows = train_reinforce_l3(L3_REINFORCE_CONFIG)
    summary_rows = ddpg_rows + reinforce_rows

    save_rows(ddpg_rows, output_dir / "ddpg_pendulum_training.csv", L3_FIELDNAMES)
    save_rows(
        reinforce_rows,
        output_dir / "reinforce_discrete_pendulum_training.csv",
        L3_FIELDNAMES,
    )
    save_rows(summary_rows, output_dir / "summary.csv", L3_FIELDNAMES)
    plot_l3(summary_rows, output_dir / "l3_training_curves.png")
    write_l3_observations(
        summary_rows,
        Path(__file__).resolve().parent / "observations_l3.txt",
    )


def main() -> None:
    """Run Week 5 experiment groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["l2", "l3", "all"], default="l2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    if args.mode == "l2":
        run_l2(args.output_dir / "l2")
    elif args.mode == "l3":
        run_l3(args.output_dir / "l3")
    else:
        run_l2(args.output_dir / "l2")
        run_l3(args.output_dir / "l3")


if __name__ == "__main__":
    main()
