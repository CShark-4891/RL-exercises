"""Experiment helpers for Week 5 REINFORCE tasks.

Examples
--------
Run the Level 2 runs used for the observations:

    python -m rl_exercises.week_5.experiments --mode l2
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


def make_env(env_name: str, seed: int, max_episode_steps: int | None) -> gym.Env:
    """Create and seed a Gymnasium environment."""
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_name, **kwargs)
    set_seed(env, seed)
    return env


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


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Save experiment rows as CSV."""
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
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


def main() -> None:
    """Run Week 5 experiment groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["l2"], default="l2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    if args.mode == "l2":
        run_l2(args.output_dir / "l2")


if __name__ == "__main__":
    main()
