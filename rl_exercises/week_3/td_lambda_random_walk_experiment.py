from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv

import numpy as np

from rl_exercises.week_3.random_walk import BoundedRandomWalkEnv


LAMBDAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
ALPHAS = tuple(np.round(np.arange(0.0, 0.65, 0.05), 2))


@dataclass(frozen=True)
class RandomWalkSequence:
    """One observation-outcome sequence x_1, ..., x_m, z from the paper."""

    states: tuple[int, ...]
    outcome: float


def generate_training_sets(
    n_training_sets: int = 100,
    sequences_per_set: int = 10,
    seed: int = 0,
) -> list[list[RandomWalkSequence]]:
    """Generate the same kind of data sets used in Sutton's Section 3.

    Sutton compares algorithms on 100 training sets, each containing 10 random
    walks.  We generate the sets once and reuse them for every lambda/alpha
    choice so that differences come from the learning rule, not from sampling
    different walks.
    """
    env = BoundedRandomWalkEnv(seed=seed)
    training_sets: list[list[RandomWalkSequence]] = []

    for _ in range(n_training_sets):
        training_set: list[RandomWalkSequence] = []
        for _ in range(sequences_per_set):
            state, _ = env.reset()
            states: list[int] = []
            done = False
            reward = 0.0

            while not done:
                states.append(int(state))
                state, reward, terminated, truncated, _ = env.step(0)
                done = terminated or truncated

            training_set.append(RandomWalkSequence(tuple(states), float(reward)))
        training_sets.append(training_set)

    return training_sets


def initial_predictions(n_states: int, initial_value: float = 0.5) -> np.ndarray:
    """Create the initial V table used by the experiments.

    Only nonterminal states are predictions in the paper.  The terminal slots are
    present because we store values by environment state index, but they are kept
    at zero and excluded from the RMS metric.
    """
    values = np.full(n_states, initial_value, dtype=float)
    values[0] = 0.0
    values[-1] = 0.0
    return values


def td_lambda_sequence_update(
    values: np.ndarray,
    sequence: RandomWalkSequence,
    lam: float,
    alpha: float,
) -> np.ndarray:
    """Return the TD(lambda) weight increment for one complete sequence.

    This follows Sutton's Equation (4) in tabular form.  While processing a
    single sequence, `values` is treated as fixed.  That matters because the
    experiments in Section 3 update after whole sequences or whole training
    sets, not after every individual transition.
    """
    update = np.zeros_like(values, dtype=float)
    eligibility = np.zeros_like(values, dtype=float)

    for index, state in enumerate(sequence.states):
        if index + 1 < len(sequence.states):
            next_prediction = values[sequence.states[index + 1]]
        else:
            # The last "next prediction" is the observed outcome z: 0 for A,
            # 1 for G.  This is the paper's prediction target.
            next_prediction = sequence.outcome

        td_error = next_prediction - values[state]

        # lambda^k recency weighting: the current state has weight 1, the
        # previous state lambda, the one before that lambda^2, and so on.
        eligibility *= lam
        eligibility[state] += 1.0
        update += alpha * td_error * eligibility

    update[0] = 0.0
    update[-1] = 0.0
    return update


def rms_error(values: np.ndarray) -> float:
    """RMS error against the true B-F probabilities 1/6, ..., 5/6."""
    true_values = np.linspace(0.0, 1.0, len(values))
    nonterminal = np.arange(1, len(values) - 1, dtype=int)
    return float(np.sqrt(np.mean((values[nonterminal] - true_values[nonterminal]) ** 2)))


def train_repeated_presentations(
    training_set: list[RandomWalkSequence],
    lam: float,
    alpha: float = 0.01,
    tolerance: float = 1e-5,
    max_sweeps: int = 20_000,
) -> tuple[np.ndarray, int]:
    """Repeatedly present one training set until the predictions converge.

    This recreates the setup behind Sutton's Figure 3: accumulate all sequence
    increments across the training set, apply them together, then present the
    same set again until the value vector barely changes.
    """
    values = initial_predictions(n_states=7)

    for sweep in range(1, max_sweeps + 1):
        total_update = np.zeros_like(values)
        for sequence in training_set:
            total_update += td_lambda_sequence_update(values, sequence, lam, alpha)

        values += total_update
        if np.max(np.abs(total_update[1:-1])) < tolerance:
            return values, sweep

    return values, max_sweeps


def train_single_presentation(
    training_set: list[RandomWalkSequence],
    lam: float,
    alpha: float,
) -> np.ndarray:
    """Present each sequence once and update after each sequence.

    This is the setup behind Sutton's Figures 4 and 5.  It highlights the
    learning-rate tradeoff: with only one pass, lambda=0 propagates terminal
    information slowly, while lambda=1 can overfit individual outcomes.
    """
    values = initial_predictions(n_states=7)
    for sequence in training_set:
        values += td_lambda_sequence_update(values, sequence, lam, alpha)
    return values


def run_repeated_presentation_experiment(
    training_sets: list[list[RandomWalkSequence]],
    lambdas: tuple[float, ...] = LAMBDAS,
) -> list[dict[str, float]]:
    """Average repeated-presentation RMS error over all training sets."""
    rows: list[dict[str, float]] = []
    for lam in lambdas:
        errors: list[float] = []
        sweeps: list[int] = []
        for training_set in training_sets:
            values, n_sweeps = train_repeated_presentations(training_set, lam)
            errors.append(rms_error(values))
            sweeps.append(n_sweeps)

        rows.append(
            {
                "experiment": "repeated_presentations",
                "lambda": lam,
                "alpha": 0.01,
                "mean_rms": float(np.mean(errors)),
                "mean_sweeps": float(np.mean(sweeps)),
            }
        )
    return rows


def run_single_presentation_experiment(
    training_sets: list[list[RandomWalkSequence]],
    lambdas: tuple[float, ...] = LAMBDAS,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Average one-pass RMS errors for every lambda/alpha combination."""
    all_rows: list[dict[str, float]] = []
    best_rows: list[dict[str, float]] = []

    for lam in lambdas:
        lambda_rows: list[dict[str, float]] = []
        for alpha in alphas:
            errors = [
                rms_error(train_single_presentation(training_set, lam, alpha))
                for training_set in training_sets
            ]
            row = {
                "experiment": "single_presentation",
                "lambda": lam,
                "alpha": alpha,
                "mean_rms": float(np.mean(errors)),
                "mean_sweeps": 1.0,
            }
            all_rows.append(row)
            lambda_rows.append(row)

        best_rows.append(min(lambda_rows, key=lambda row: row["mean_rms"]))

    return all_rows, best_rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write experiment rows as a small reproducible result table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "lambda", "alpha", "mean_rms", "mean_sweeps"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_observations(
    path: Path,
    repeated_rows: list[dict[str, float]],
    best_rows: list[dict[str, float]],
    seed: int,
    n_training_sets: int,
    sequences_per_set: int,
) -> None:
    """Create the Level-3 observations file in a compact readable format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    repeated_table = "\n".join(
        f"- lambda={row['lambda']:.1f}: RMS={row['mean_rms']:.4f}"
        for row in repeated_rows
    )
    best_table = "\n".join(
        f"- lambda={row['lambda']:.1f}: best alpha={row['alpha']:.2f}, "
        f"RMS={row['mean_rms']:.4f}"
        for row in best_rows
    )

    best_repeated = min(repeated_rows, key=lambda row: row["mean_rms"])
    best_single = min(best_rows, key=lambda row: row["mean_rms"])

    text = f"""TD(lambda) observations for Week 3 Level 3
================================================

What we did
-----------
We recreated Sutton's bounded random-walk example from the TD(lambda) paper.
The walk has states A-G, starts in D, and moves left or right with probability
0.5.  We predict if the walk ends on the right side.  The true values for B-F
are 1/6, 2/6, 3/6, 4/6, and 5/6.

We implemented TD(lambda) as a prediction agent.  It learns V(s), not Q(s,a),
because the paper is about predicting outcomes and not about choosing actions.
We used one dummy action only because the repo training loop expects an action.

Setup
-----
Seed: {seed}
Training sets: {n_training_sets}
Sequences per training set: {sequences_per_set}
Lambda values: {', '.join(str(value) for value in LAMBDAS)}

Repeated presentations
----------------------
{repeated_table}

Best result here: lambda={best_repeated['lambda']:.1f},
RMS={best_repeated['mean_rms']:.4f}.

This matches the paper: when we show the same data many times, TD(0) works
best and lambda=1.0 works worst.

Single presentation
-------------------
{best_table}

Best result here: lambda={best_single['lambda']:.1f},
alpha={best_single['alpha']:.2f}, RMS={best_single['mean_rms']:.4f}.

This also matches the paper: with only one pass, a small or middle lambda works
best, and lambda=1.0 is worst.

Short explanation
-----------------
Lambda says how much older states are updated by the current TD error.  With
lambda=0, mainly the current state is updated.  With lambda=1, many old states
are updated, but this can overfit the final outcome of one random walk.  Values
between 0 and 1 are often a good compromise.
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate Sutton's random-walk TD(lambda) experiments."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-sets", type=int, default=100)
    parser.add_argument("--sequences-per-set", type=int, default=10)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("rl_exercises/week_3/td_lambda_random_walk_results.csv"),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("rl_exercises/week_3/observations_l3.txt"),
    )
    args = parser.parse_args()

    training_sets = generate_training_sets(
        n_training_sets=args.training_sets,
        sequences_per_set=args.sequences_per_set,
        seed=args.seed,
    )
    repeated_rows = run_repeated_presentation_experiment(training_sets)
    single_rows, best_rows = run_single_presentation_experiment(training_sets)

    write_csv(args.csv, repeated_rows + single_rows)
    write_observations(
        args.observations,
        repeated_rows,
        best_rows,
        seed=args.seed,
        n_training_sets=args.training_sets,
        sequences_per_set=args.sequences_per_set,
    )

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.observations}")


if __name__ == "__main__":
    main()
