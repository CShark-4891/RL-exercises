"""Experiment helpers for Week 4 DQN tasks.

Examples
--------
Run the five-seed RLiable plot for Level 2:

    python -m rl_exercises.week_4.experiments --mode l2

Run the Rainbow-style ablation for Level 3:

    python -m rl_exercises.week_4.experiments --mode l3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rl_exercises.week_4.dqn import DQNAgent

L1_CONFIGS = {
    "base": {
        "hidden_dim": 64,
        "hidden_layers": 2,
        "buffer_capacity": 10000,
        "batch_size": 32,
    },
    "wide": {
        "hidden_dim": 128,
        "hidden_layers": 2,
        "buffer_capacity": 10000,
        "batch_size": 32,
    },
    "deep": {
        "hidden_dim": 64,
        "hidden_layers": 3,
        "buffer_capacity": 10000,
        "batch_size": 32,
    },
    "small_buffer": {
        "hidden_dim": 64,
        "hidden_layers": 2,
        "buffer_capacity": 2000,
        "batch_size": 32,
    },
    "large_batch": {
        "hidden_dim": 64,
        "hidden_layers": 2,
        "buffer_capacity": 10000,
        "batch_size": 64,
    },
}

L1_LABELS = {
    "base": "base (64x2, buffer=10000, batch=32)",
    "wide": "wide (128x2, buffer=10000, batch=32)",
    "deep": "deep (64x3, buffer=10000, batch=32)",
    "small_buffer": "small_buffer (64x2, buffer=2000, batch=32)",
    "large_batch": "large_batch (64x2, buffer=10000, batch=64)",
}

L3_CONFIGS = {
    "base_dqn": {},
    "prioritized_replay": {"prioritized_replay": True},
    "double_dqn": {"double_dqn": True},
    "prioritized_replay_double_dqn": {
        "prioritized_replay": True,
        "double_dqn": True,
    },
}

DEFAULT_AGENT_KWARGS = {
    "buffer_capacity": 10000,
    "batch_size": 32,
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_final": 0.01,
    "epsilon_decay": 500,
    "target_update_freq": 1000,
    "hidden_dim": 64,
    "hidden_layers": 2,
}


def parse_seeds(seed_text: str) -> list[int]:
    """Parse comma-separated seed values."""
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def run_single_seed(
    env_name: str,
    seed: int,
    num_frames: int,
    eval_interval: int,
    agent_kwargs: dict,
    config_name: str,
) -> pd.DataFrame:
    """Train one DQN agent and return fixed-frame reward statistics."""
    hidden_dim = agent_kwargs.get("hidden_dim", DEFAULT_AGENT_KWARGS["hidden_dim"])
    hidden_layers = agent_kwargs.get(
        "hidden_layers", DEFAULT_AGENT_KWARGS["hidden_layers"]
    )
    buffer_capacity = agent_kwargs.get(
        "buffer_capacity", DEFAULT_AGENT_KWARGS["buffer_capacity"]
    )
    batch_size = agent_kwargs.get("batch_size", DEFAULT_AGENT_KWARGS["batch_size"])
    details = (
        f"hidden_dim={hidden_dim}, hidden_layers={hidden_layers}, "
        f"buffer_capacity={buffer_capacity}, batch_size={batch_size}"
    )
    print(f"Running {config_name}, seed={seed} ({details})")
    env = gym.make(env_name)
    kwargs = dict(DEFAULT_AGENT_KWARGS)
    kwargs.update(agent_kwargs)
    kwargs["seed"] = seed

    agent = DQNAgent(env, **kwargs)
    history = agent.train(
        num_frames=num_frames,
        eval_interval=eval_interval,
        verbose=False,
    )
    env.close()

    return pd.DataFrame(
        {
            "seed": seed,
            "frames": np.asarray(history["frames"], dtype=int),
            "mean_reward_10": np.asarray(history["mean_reward_10"], dtype=float),
        }
    )


def collect_results(
    configs: dict[str, dict],
    env_name: str,
    seeds: list[int],
    num_frames: int,
    eval_interval: int,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run all configs and return scores shaped as seed x time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dict: dict[str, np.ndarray] = {}
    steps: np.ndarray | None = None

    for name, config in configs.items():
        frames = []
        seed_scores = []
        dfs = []
        for seed in seeds:
            df = run_single_seed(
                env_name,
                seed,
                num_frames,
                eval_interval,
                config,
                name,
            )
            df["config"] = name
            dfs.append(df)
            frames.append(df["frames"].to_numpy())
            seed_scores.append(df["mean_reward_10"].to_numpy())

        csv_path = output_dir / f"{name}_training.csv"
        pd.concat(dfs, ignore_index=True).to_csv(csv_path, index=False)
        print(f"Saved training data to {csv_path}")
        if steps is None:
            steps = frames[0]
        score_dict[name] = np.vstack(seed_scores)

    assert steps is not None
    return steps, score_dict


def plot_mean_training_curves(
    steps: np.ndarray,
    score_dict: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    labels: dict[str, str] | None = None,
) -> None:
    """Plot mean reward curves with one standard deviation."""
    plt.figure(figsize=(11, 6))
    for name, scores in score_dict.items():
        mean = scores.mean(axis=0)
        std = scores.std(axis=0)
        label = labels.get(name, name) if labels is not None else name
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.18)
    plt.xlabel("Number of frames")
    plt.ylabel("Mean reward over last 10 episodes")
    plt.title(title)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved mean training curve to {output_path}")


def plot_rliable_curves(
    steps: np.ndarray,
    score_dict: dict[str, np.ndarray],
    output_path: Path,
    optimal_score: float = 500.0,
    reps: int = 2000,
) -> None:
    """Plot IQM, mean, median, and optimality gap with RLiable intervals."""
    from rliable import metrics
    from rliable.library import get_interval_estimates
    from rliable.plot_utils import plot_sample_efficiency_curve

    normalized_scores = {
        name: np.clip(scores / optimal_score, 0.0, 1.0)
        for name, scores in score_dict.items()
    }
    algorithms = list(normalized_scores.keys())
    metric_fns = {
        "IQM": metrics.aggregate_iqm,
        "Mean": metrics.aggregate_mean,
        "Median": metrics.aggregate_median,
        "Optimality gap": metrics.aggregate_optimality_gap,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric_name, metric_fn) in zip(axes.ravel(), metric_fns.items()):

        def aggregate(scores: np.ndarray, fn=metric_fn) -> np.ndarray:
            return np.array(
                [
                    fn(scores[:, eval_idx : eval_idx + 1])
                    for eval_idx in range(scores.shape[-1])
                ]
            )

        point_estimates, interval_estimates = get_interval_estimates(
            normalized_scores,
            aggregate,
            reps=reps,
        )
        plot_sample_efficiency_curve(
            steps,
            point_estimates,
            interval_estimates,
            algorithms=algorithms,
            xlabel="Number of frames",
            ylabel=metric_name,
            ax=ax,
        )
        ax.set_title(metric_name)

    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=min(len(legend_labels), 4),
        fontsize="medium",
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved RLiable curves to {output_path}")


def main() -> None:
    """Run Week 4 experiment groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["l1", "l2", "l3"], default="l2")
    parser.add_argument("--env-name", default="CartPole-v1")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--num-frames", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--rliable-reps", type=int, default=2000)
    args = parser.parse_args()

    if args.mode == "l1":
        configs = L1_CONFIGS
        labels = L1_LABELS
        default_seeds = [0]
    elif args.mode == "l2":
        configs = {"base_dqn": {}}
        labels = None
        default_seeds = [0, 1, 2, 3, 4]
    else:
        configs = L3_CONFIGS
        labels = None
        default_seeds = [0, 1, 2, 3, 4]

    seeds = parse_seeds(args.seeds) if args.seeds is not None else default_seeds

    output_dir = args.output_dir / args.mode
    steps, scores = collect_results(
        configs,
        args.env_name,
        seeds,
        args.num_frames,
        args.eval_interval,
        output_dir,
    )

    plot_mean_training_curves(
        steps,
        scores,
        output_dir / f"{args.mode}_mean_training_curves.png",
        title=f"Week 4 {args.mode.upper()} training curves",
        labels=labels,
    )
    if args.mode in {"l2", "l3"}:
        plot_rliable_curves(
            steps,
            scores,
            output_dir / f"{args.mode}_rliable_curves.png",
            reps=args.rliable_reps,
        )


if __name__ == "__main__":
    main()
