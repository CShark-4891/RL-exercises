"""Experiment helpers for Week 6 actor-critic tasks.

Examples
--------
Run the complete Week 6 submission experiments:

    python -m rl_exercises.week_6.experiments --mode all

Run only the PPO comparison:

    python -m rl_exercises.week_6.experiments --mode l2
"""

from __future__ import annotations

from typing import Any

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rl_exercises.week_5.policy_gradient import set_seed  # noqa: E402
from rl_exercises.week_6.actor_critic import ActorCriticAgent  # noqa: E402
from rl_exercises.week_6.ppo import PPOAgent  # noqa: E402
from rl_exercises.week_6.sac import SACAgent  # noqa: E402

BASE_FIELDNAMES = [
    "level",
    "agent",
    "config",
    "env",
    "seed",
    "step",
    "episode",
    "train_return",
    "train_mean_last_10",
    "eval_mean",
    "eval_std",
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "actor_loss",
    "q1_loss",
    "q2_loss",
    "alpha",
]

L1_BASELINES = ["none", "avg", "value", "gae"]
L1_ENVS = ["CartPole-v1", "LunarLander-v3"]
L2_CONFIGS = {
    "actor_critic_gae": {"agent": "actor_critic", "baseline_type": "gae"},
    "ppo_vanilla": {"agent": "ppo", "value_clip": False, "target_kl": None},
    "ppo_enhanced": {"agent": "ppo", "value_clip": True, "target_kl": 0.03},
}


class DiscretizeOneDimensionalAction(gym.ActionWrapper):
    """Expose a one-dimensional Box action space as a small Discrete space."""

    def __init__(self, env: gym.Env, bins: int = 9) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("Can only discretize a Box action space")
        if int(np.prod(env.action_space.shape)) != 1:
            raise ValueError("Only one-dimensional continuous actions are supported")
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
        return np.asarray([self.actions[int(action)]], dtype=np.float32)


def parse_seeds(seed_text: str) -> list[int]:
    """Parse comma-separated seeds."""
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def slug(text: str) -> str:
    """Make a short file-safe identifier."""
    return text.lower().replace("-", "_").replace("/", "_")


def make_env(
    env_name: str,
    seed: int,
    max_episode_steps: int | None = None,
    discrete_action_bins: int | None = None,
) -> gym.Env:
    """Create and seed an environment."""
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_name, **kwargs)
    if discrete_action_bins is not None:
        env = DiscretizeOneDimensionalAction(env, discrete_action_bins)
    set_seed(env, seed)
    return env


def empty_row(level: str, agent: str, config: str, env_name: str, seed: int) -> dict:
    """Create a row with all known CSV fields."""
    return {
        "level": level,
        "agent": agent,
        "config": config,
        "env": env_name,
        "seed": seed,
        "step": "",
        "episode": "",
        "train_return": "",
        "train_mean_last_10": "",
        "eval_mean": "",
        "eval_std": "",
        "policy_loss": "",
        "value_loss": "",
        "entropy_loss": "",
        "actor_loss": "",
        "q1_loss": "",
        "q2_loss": "",
        "alpha": "",
    }


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Save experiment rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BASE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def evaluate_ppo(
    agent: PPOAgent, eval_env: gym.Env, num_episodes: int
) -> tuple[float, float]:
    """Evaluate PPO with deterministic argmax actions."""
    returns = []
    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        done = False
        total_return = 0.0
        while not done:
            action, _, _, _ = agent.predict(state, evaluate=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_return += float(reward)
        returns.append(total_return)
    return float(np.mean(returns)), float(np.std(returns))


def maybe_add_eval_row(
    rows: list[dict[str, Any]],
    level: str,
    agent_name: str,
    config_name: str,
    env_name: str,
    seed: int,
    step: int,
    episode: int,
    recent_returns: list[float],
    eval_mean: float,
    eval_std: float,
    losses: dict[str, float],
) -> None:
    """Append one evaluation row."""
    row = empty_row(level, agent_name, config_name, env_name, seed)
    row.update(
        {
            "step": step,
            "episode": episode,
            "train_return": recent_returns[-1] if recent_returns else "",
            "train_mean_last_10": (
                float(np.mean(recent_returns[-10:])) if recent_returns else ""
            ),
            "eval_mean": eval_mean,
            "eval_std": eval_std,
        }
    )
    for key, value in losses.items():
        row[key] = value
    rows.append(row)


def train_actor_critic_config(
    env_name: str,
    baseline_type: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    level: str = "l1",
    config_name: str | None = None,
) -> list[dict[str, Any]]:
    """Train one actor-critic configuration and collect eval rows."""
    config_name = config_name or baseline_type
    env = make_env(env_name, seed)
    eval_env = make_env(env_name, seed + 1000)
    agent = ActorCriticAgent(
        env,
        lr_actor=5e-4 if env_name == "CartPole-v1" else 1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        hidden_size=64,
        baseline_type=baseline_type,
        baseline_decay=0.9,
    )

    rows: list[dict[str, Any]] = []
    recent_returns: list[float] = []
    latest_losses = {"policy_loss": np.nan, "value_loss": np.nan}
    step = 0
    episode = 0

    while step < total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []
        episode_return = 0.0
        episode += 1

        while not done and step < total_steps:
            action, logp = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, float(reward), next_state, done, logp))
            episode_return += float(reward)
            state = next_state
            step += 1

            if step % eval_interval == 0:
                eval_mean, eval_std = agent.evaluate(eval_env, eval_episodes)
                maybe_add_eval_row(
                    rows,
                    level,
                    "ActorCritic",
                    config_name,
                    env_name,
                    seed,
                    step,
                    episode,
                    recent_returns,
                    eval_mean,
                    eval_std,
                    latest_losses,
                )

        if trajectory:
            policy_loss, value_loss = agent.update_agent(trajectory)
            latest_losses = {"policy_loss": policy_loss, "value_loss": value_loss}
        recent_returns.append(episode_return)

    env.close()
    eval_env.close()
    return rows


def train_ppo_config(
    env_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    config_name: str,
    value_clip: bool,
    target_kl: float | None,
    max_episode_steps: int | None = None,
    discrete_action_bins: int | None = None,
    level: str = "l2",
) -> list[dict[str, Any]]:
    """Train one PPO configuration and collect eval rows."""
    env = make_env(env_name, seed, max_episode_steps, discrete_action_bins)
    eval_env = make_env(env_name, seed + 1000, max_episode_steps, discrete_action_bins)
    agent = PPOAgent(
        env,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        value_clip=value_clip,
        target_kl=target_kl,
        seed=seed,
        hidden_size=64,
    )

    rows: list[dict[str, Any]] = []
    recent_returns: list[float] = []
    latest_losses = {
        "policy_loss": np.nan,
        "value_loss": np.nan,
        "entropy_loss": np.nan,
    }
    step = 0
    episode = 0

    while step < total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []
        episode_return = 0.0
        episode += 1

        while not done and step < total_steps:
            action, logp, entropy, _ = agent.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append(
                (state, action, logp, entropy, float(reward), float(done), next_state)
            )
            episode_return += float(reward)
            state = next_state
            step += 1

            if step % eval_interval == 0:
                eval_mean, eval_std = evaluate_ppo(agent, eval_env, eval_episodes)
                maybe_add_eval_row(
                    rows,
                    level,
                    "PPO",
                    config_name,
                    env_name,
                    seed,
                    step,
                    episode,
                    recent_returns,
                    eval_mean,
                    eval_std,
                    latest_losses,
                )

        if trajectory:
            policy_loss, value_loss, entropy_loss = agent.update(trajectory)
            latest_losses = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
            }
        recent_returns.append(episode_return)

    env.close()
    eval_env.close()
    return rows


def train_sac_config(
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
) -> list[dict[str, Any]]:
    """Train SAC on native continuous Pendulum-v1."""
    env_name = "Pendulum-v1"
    env = make_env(env_name, seed, max_episode_steps=200)
    eval_env = make_env(env_name, seed + 1000, max_episode_steps=200)
    agent = SACAgent(
        env,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=50000,
        batch_size=64,
        learning_starts=256,
        hidden_size=64,
        seed=seed,
    )
    history = agent.train(
        num_frames=total_steps,
        eval_env=eval_env,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )

    rows: list[dict[str, Any]] = []
    for index, step in enumerate(history["frames"]):
        row = empty_row("l3", "SAC", "sac_continuous", env_name, seed)
        row.update(
            {
                "step": int(step),
                "episode": "",
                "train_mean_last_10": history["mean_reward_10"][index],
                "eval_mean": history["eval_mean"][index],
                "eval_std": history["eval_std"][index],
                "actor_loss": history["actor_loss"][index],
                "q1_loss": history["q1_loss"][index],
                "q2_loss": history["q2_loss"][index],
                "alpha": history["alpha"][index],
            }
        )
        rows.append(row)

    env.close()
    eval_env.close()
    return rows


def rows_to_score_dict(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert rows into config -> seeds x eval_steps score arrays."""
    configs = sorted({str(row["config"]) for row in rows})
    steps = np.asarray(sorted({int(row["step"]) for row in rows}), dtype=int)
    score_dict: dict[str, np.ndarray] = {}

    for config in configs:
        config_rows = [row for row in rows if row["config"] == config]
        seeds = sorted({int(row["seed"]) for row in config_rows})
        seed_scores = []
        for seed in seeds:
            seed_rows = [row for row in config_rows if int(row["seed"]) == seed]
            by_step = {int(row["step"]): float(row["eval_mean"]) for row in seed_rows}
            seed_scores.append([by_step[int(step)] for step in steps])
        score_dict[config] = np.asarray(seed_scores, dtype=float)

    return steps, score_dict


def plot_mean_training_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    """Plot mean evaluation return with one standard deviation band."""
    steps, score_dict = rows_to_score_dict(rows)
    plt.figure(figsize=(10, 6))
    for config, scores in score_dict.items():
        mean = scores.mean(axis=0)
        std = scores.std(axis=0)
        plt.plot(steps, mean, label=config)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.18)
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title(title)
    plt.legend(fontsize="small")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_rliable_mean_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    reps: int = 500,
) -> None:
    """Plot RLiable bootstrap intervals for mean evaluation return."""
    from rliable import metrics
    from rliable.library import get_interval_estimates
    from rliable.plot_utils import plot_sample_efficiency_curve

    steps, score_dict = rows_to_score_dict(rows)

    def aggregate(scores: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                metrics.aggregate_mean(scores[:, eval_idx : eval_idx + 1])
                for eval_idx in range(scores.shape[-1])
            ],
            dtype=float,
        )

    point_estimates, interval_estimates = get_interval_estimates(
        score_dict, aggregate, reps=reps
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_sample_efficiency_curve(
        steps,
        point_estimates,
        interval_estimates,
        algorithms=list(score_dict.keys()),
        xlabel="Environment steps",
        ylabel="Evaluation return",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def run_l1(
    output_dir: Path,
    seeds: list[int],
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    rliable_reps: int,
) -> None:
    """Run Level 1 actor-critic baseline comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    for env_name in L1_ENVS:
        env_rows: list[dict[str, Any]] = []
        for baseline in L1_BASELINES:
            config_rows = []
            for seed in seeds:
                print(f"Running L1 {env_name} baseline={baseline}, seed={seed}")
                config_rows.extend(
                    train_actor_critic_config(
                        env_name,
                        baseline,
                        seed,
                        total_steps,
                        eval_interval,
                        eval_episodes,
                    )
                )
            save_rows(
                config_rows,
                output_dir / f"actor_critic_{slug(env_name)}_{baseline}.csv",
            )
            env_rows.extend(config_rows)
            all_rows.extend(config_rows)

        plot_mean_training_curves(
            env_rows,
            output_dir / f"l1_{slug(env_name)}_mean_curves.png",
            f"Week 6 L1 actor-critic baselines on {env_name}",
        )
        plot_rliable_mean_curves(
            env_rows,
            output_dir / f"l1_{slug(env_name)}_rliable_curves.png",
            f"Week 6 L1 RLiable mean curve on {env_name}",
            reps=rliable_reps,
        )

    save_rows(all_rows, output_dir / "summary.csv")


def run_l2(
    output_dir: Path,
    seeds: list[int],
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    rliable_reps: int,
) -> None:
    """Run Level 2 PPO comparisons on LunarLander-v3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    env_name = "LunarLander-v3"

    for config_name, config in L2_CONFIGS.items():
        config_rows = []
        for seed in seeds:
            print(f"Running L2 {config_name}, seed={seed}")
            if config["agent"] == "actor_critic":
                rows = train_actor_critic_config(
                    env_name,
                    config["baseline_type"],
                    seed,
                    total_steps,
                    eval_interval,
                    eval_episodes,
                    level="l2",
                    config_name=config_name,
                )
            else:
                rows = train_ppo_config(
                    env_name,
                    seed,
                    total_steps,
                    eval_interval,
                    eval_episodes,
                    config_name=config_name,
                    value_clip=bool(config["value_clip"]),
                    target_kl=config["target_kl"],
                    level="l2",
                )
            config_rows.extend(rows)
        save_rows(config_rows, output_dir / f"{config_name}.csv")
        all_rows.extend(config_rows)

    save_rows(all_rows, output_dir / "summary.csv")
    plot_mean_training_curves(
        all_rows,
        output_dir / "l2_mean_curves.png",
        "Week 6 L2 PPO vs actor-critic on LunarLander-v3",
    )
    plot_rliable_mean_curves(
        all_rows,
        output_dir / "l2_rliable_curves.png",
        "Week 6 L2 RLiable mean curve on LunarLander-v3",
        reps=rliable_reps,
    )


def run_l3(
    output_dir: Path,
    seeds: list[int],
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    rliable_reps: int,
) -> None:
    """Run Level 3 SAC comparison on Pendulum-v1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    ppo_rows = []
    sac_rows = []
    for seed in seeds:
        print(f"Running L3 PPO-discrete Pendulum, seed={seed}")
        ppo_rows.extend(
            train_ppo_config(
                "Pendulum-v1",
                seed,
                total_steps,
                eval_interval,
                eval_episodes,
                config_name="ppo_discrete",
                value_clip=True,
                target_kl=0.03,
                max_episode_steps=200,
                discrete_action_bins=9,
                level="l3",
            )
        )

        print(f"Running L3 SAC Pendulum, seed={seed}")
        sac_rows.extend(
            train_sac_config(seed, total_steps, eval_interval, eval_episodes)
        )

    save_rows(ppo_rows, output_dir / "ppo_discrete_pendulum.csv")
    save_rows(sac_rows, output_dir / "sac_pendulum.csv")
    all_rows.extend(ppo_rows)
    all_rows.extend(sac_rows)
    save_rows(all_rows, output_dir / "summary.csv")
    plot_mean_training_curves(
        all_rows,
        output_dir / "l3_mean_curves.png",
        "Week 6 L3 SAC vs discrete PPO on Pendulum-v1",
    )
    plot_rliable_mean_curves(
        all_rows,
        output_dir / "l3_rliable_curves.png",
        "Week 6 L3 RLiable mean curve on Pendulum-v1",
        reps=rliable_reps,
    )


def main() -> None:
    """Run Week 6 experiment groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "l1", "l2", "l3"], default="all")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--l1-steps", type=int, default=3000)
    parser.add_argument("--l2-steps", type=int, default=3000)
    parser.add_argument("--l3-steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--rliable-reps", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    torch.set_num_threads(1)

    if args.mode in {"all", "l1"}:
        run_l1(
            args.output_dir / "l1",
            seeds,
            args.l1_steps,
            args.eval_interval,
            args.eval_episodes,
            args.rliable_reps,
        )
    if args.mode in {"all", "l2"}:
        run_l2(
            args.output_dir / "l2",
            seeds,
            args.l2_steps,
            args.eval_interval,
            args.eval_episodes,
            args.rliable_reps,
        )
    if args.mode in {"all", "l3"}:
        run_l3(
            args.output_dir / "l3",
            seeds,
            args.l3_steps,
            args.eval_interval,
            args.eval_episodes,
            args.rliable_reps,
        )


if __name__ == "__main__":
    main()
