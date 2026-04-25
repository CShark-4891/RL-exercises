from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from rich import print as printr

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rl_exercises.environments import ContextualMarsRover
from rl_exercises.week_2 import PolicyIteration, ValueIteration


TRAIN_CONTEXTS = [0, 1]
VALIDATION_CONTEXTS = [2]
TEST_CONTEXTS = [3]
ALL_CONTEXTS = TRAIN_CONTEXTS + VALIDATION_CONTEXTS + TEST_CONTEXTS


def build_env(
    *,
    include_context: bool,
    active_contexts: list[int],
    seed: int = 333,
) -> ContextualMarsRover:
    """Create a contextual MarsRover with the requested split."""
    return ContextualMarsRover(
        include_context=include_context,
        active_contexts=active_contexts,
        context_schedule="round_robin",
        seed=seed,
    )


def evaluate_agent(
    env: ContextualMarsRover,
    agent: PolicyIteration | ValueIteration,
    episodes: int = 20,
    seed: int = 333,
) -> float:
    """Evaluate an agent by averaging returns over several episodes."""
    total_return = 0.0
    for episode_idx in range(episodes):
        observation, info = env.reset(seed=seed + episode_idx)
        done = False
        while not done:
            action, _ = agent.predict_action(observation, info, evaluate=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            done = bool(terminated or truncated)
    return total_return / episodes


def run_experiment(
    agent_class: type[PolicyIteration] | type[ValueIteration],
    *,
    include_context: bool,
    planning_contexts: list[int],
    representation_name: str,
    seed: int = 333,
) -> list[dict[str, float | str]]:
    """Fit one agent and evaluate it on train, validation, and test splits."""
    planning_env = build_env(
        include_context=include_context,
        active_contexts=planning_contexts,
        seed=seed,
    )
    agent = agent_class(
        env=planning_env, seed=seed, filename="/tmp/contextual_policy.npy"
    )
    agent.update_agent()

    split_definitions = {
        "train": TRAIN_CONTEXTS,
        "validation": VALIDATION_CONTEXTS,
        "test": TEST_CONTEXTS,
    }
    results: list[dict[str, float | str]] = []
    for split_name, split_contexts in split_definitions.items():
        eval_env = build_env(
            include_context=include_context,
            active_contexts=split_contexts,
            seed=seed,
        )
        mean_return = evaluate_agent(eval_env, agent, seed=seed)
        results.append(
            {
                "algorithm": agent_class.__name__,
                "representation": representation_name,
                "split": split_name,
                "mean_return": mean_return,
            }
        )
    return results


def main() -> None:
    """Run the Level 3 contextual MarsRover comparisons."""
    experiments = [
        {
            "agent_class": PolicyIteration,
            "include_context": False,
            "planning_contexts": TRAIN_CONTEXTS,
            "representation_name": "hidden_context_average_train",
        },
        {
            "agent_class": ValueIteration,
            "include_context": False,
            "planning_contexts": TRAIN_CONTEXTS,
            "representation_name": "hidden_context_average_train",
        },
        {
            "agent_class": PolicyIteration,
            "include_context": True,
            "planning_contexts": ALL_CONTEXTS,
            "representation_name": "context_aware_oracle_all_contexts",
        },
        {
            "agent_class": ValueIteration,
            "include_context": True,
            "planning_contexts": ALL_CONTEXTS,
            "representation_name": "context_aware_oracle_all_contexts",
        },
    ]

    results: list[dict[str, float | str]] = []
    for experiment in experiments:
        results.extend(run_experiment(**experiment))

    results_df = pd.DataFrame(results)
    output_path = Path(__file__).with_name("contextual_results.csv")
    results_df.to_csv(output_path, index=False)

    printr(results_df.to_string(index=False))
    printr(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
