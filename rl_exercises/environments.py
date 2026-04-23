"""GridCore Env taken from https://github.com/automl/TabularTempoRL/"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class MarsRover(gym.Env):
    """
    Simple Environment for a Mars Rover that can move in a 1D Space.

    The rover starts at position 2 and moves left or right based on discrete actions.
    The environment is stochastic: with a probability defined by a transition matrix,
    the action may be flipped. Each cell has an associated reward.

    Actions
    -------
    Discrete(2):
    - 0: go left
    - 1: go right

    Observations
    ------------
    Discrete(n): The current position of the rover (int).

    Reward
    ------
    Depends on the resulting cell after action is taken.

    Start/Reset State
    -----------------
    Always starts at position 2.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        transition_probabilities: np.ndarray = np.ones((5, 2)),
        rewards: list[float] = [1, 0, 0, 0, 10],
        horizon: int = 10,
        seed: int | None = None,
    ):
        """
        Initialize the Mars Rover environment.

        Parameters
        ----------
        transition_probabilities : np.ndarray, optional
            A (num_states, 2) array specifying the probability of actions being followed.
        rewards : list of float, optional
            Rewards assigned to each position, by default [1, 0, 0, 0, 10].
        horizon : int, optional
            Maximum number of steps per episode, by default 10.
        seed : int or None, optional
            Random seed for reproducibility, by default None.
        """
        self.rng = np.random.default_rng(seed)

        self.rewards = list(rewards)
        self.P = np.asarray(transition_probabilities, dtype=float)
        self.horizon = int(horizon)
        self.current_steps = 0
        self.position = 2  # start at middle

        # spaces
        n = self.P.shape[0]
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(2)

        # helpers
        self.states = np.arange(n)
        self.actions = np.arange(2)

        # transition matrix
        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Seed for environment reset (unused).
        options : dict, optional
            Additional reset options (unused).

        Returns
        -------
        state : int
            Initial state (always 2).
        info : dict
            An empty info dictionary.
        """
        self.current_steps = 0
        self.position = 2
        return self.position, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0: left, 1: right).

        Returns
        -------
        next_state : int
            The resulting position of the rover.
        reward : float
            The reward at the new position.
        terminated : bool
            Whether the episode ended due to task success (always False here).
        truncated : bool
            Whether the episode ended due to reaching the time limit.
        info : dict
            An empty dictionary.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        # stochastic flip with prob 1 - P[pos, action]
        p = float(self.P[self.position, action])
        follow = self.rng.random() < p
        a_used = action if follow else 1 - action

        delta = -1 if a_used == 0 else 1
        self.position = max(0, min(self.states[-1], self.position + delta))

        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """
        Return the reward function R[s, a] for each (state, action) pair.

        R[s, a] is the expected immediate reward after choosing action a in state s.

        Returns
        -------
        R : np.ndarray
            A (num_states, num_actions) array of rewards.
        """
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            for a in range(nA):
                follow_state = self.get_next_state(s, a)
                flipped_state = self.get_next_state(s, 1 - a)
                follow_prob = float(self.P[s, a])

                # When the action is stochastic, planning needs the expected
                # one-step reward of the action, not just the reward of the
                # intended successor state.
                R[s, a] = follow_prob * float(self.rewards[follow_state]) + (
                    1.0 - follow_prob
                ) * float(self.rewards[flipped_state])
        return R

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct the transition matrix T[s, a, s'].

        Parameters
        ----------
        S : np.ndarray, optional
            Array of states. Uses internal states if None.
        A : np.ndarray, optional
            Array of actions. Uses internal actions if None.
        P : np.ndarray, optional
            Action success probabilities. Uses internal P if None.
            P[s, a] is the probability that action a is followed. With
            probability 1 - P[s, a], the opposite action is executed.

        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor where
            T[s, a, s'] = probability of transitioning to s' from s via a.
        """
        if S is None or A is None or P is None:
            S, A, P = self.states, self.actions, self.P

        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in S:
            for a in A:
                follow_state = self.get_next_state(int(s), int(a))
                flipped_state = self.get_next_state(int(s), 1 - int(a))
                follow_prob = float(P[s, a])

                T[s, a, follow_state] += follow_prob
                T[s, a, flipped_state] += 1.0 - follow_prob

        # Every state-action pair must define a proper probability distribution.
        assert np.allclose(T.sum(axis=2), 1.0)
        return T

    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str
            Render mode (only "human" is supported).
        """
        print(f"[MarsRover] pos={self.position}, steps={self.current_steps}")

    def get_next_state(self, state: int, action: int) -> int:
        """Return the clipped successor state for a deterministic action."""
        if action == 0:
            return max(0, state - 1)
        return min(len(self.states) - 1, state + 1)


class ContextualMarsRover(gym.Env):
    """
    MarsRover environment with a finite set of contexts.

    Each context changes two environment properties:
    - `follow_prob`: probability that the chosen action is executed
    - `goal_reward`: reward of the right-most cell

    The environment can either expose the context as part of the observation
    (`include_context=True`) or hide it and average planning quantities across
    the active contexts (`include_context=False`).
    """

    metadata = {"render_modes": ["human"]}

    DEFAULT_CONTEXTS = (
        {
            "name": "stable_low_goal",
            "follow_prob": 1.0,
            "goal_reward": 8.0,
        },
        {
            "name": "mild_slip_medium_goal",
            "follow_prob": 0.85,
            "goal_reward": 10.0,
        },
        {
            "name": "fragile_low_goal",
            "follow_prob": 0.55,
            "goal_reward": 4.0,
        },
        {
            "name": "hostile_slip_low_goal",
            "follow_prob": 0.4,
            "goal_reward": 6.0,
        },
    )

    def __init__(
        self,
        contexts: list[dict[str, float | str]] | None = None,
        active_contexts: list[int] | None = None,
        include_context: bool = False,
        context_schedule: str = "round_robin",
        horizon: int = 10,
        seed: int | None = None,
    ):
        """
        Initialize the contextual MarsRover environment.

        Parameters
        ----------
        contexts : list of dict, optional
            Context definitions. Each context must provide `follow_prob`
            and `goal_reward`.
        active_contexts : list of int, optional
            Indices of contexts that should be sampled during reset.
        include_context : bool, optional
            Whether the observation should encode the context explicitly.
        context_schedule : str, optional
            How contexts are selected on reset. Supported values are
            `round_robin` and `random`.
        horizon : int, optional
            Maximum number of steps per episode.
        seed : int or None, optional
            Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.contexts = [
            {
                "name": str(context["name"]),
                "follow_prob": float(context["follow_prob"]),
                "goal_reward": float(context["goal_reward"]),
            }
            for context in (contexts or list(self.DEFAULT_CONTEXTS))
        ]
        self.active_contexts = (
            [int(index) for index in active_contexts]
            if active_contexts is not None
            else list(range(len(self.contexts)))
        )
        self.include_context = bool(include_context)
        self.context_schedule = context_schedule
        self.horizon = int(horizon)
        self.current_steps = 0
        self.n_positions = 5
        self.start_position = 2
        self.position = self.start_position
        self.actions = np.arange(2)
        self._context_pointer = 0

        if not self.active_contexts:
            raise ValueError("active_contexts must contain at least one context index.")
        if self.context_schedule not in {"round_robin", "random"}:
            raise ValueError(
                "context_schedule must be either 'round_robin' or 'random'."
            )

        for context_index in self.active_contexts:
            if context_index < 0 or context_index >= len(self.contexts):
                raise ValueError(
                    f"context index {context_index} is outside the available range."
                )

        self.current_context_id = self.active_contexts[0]
        self.rewards = self._context_rewards(self.current_context_id)

        n_observations = (
            len(self.contexts) * self.n_positions
            if self.include_context
            else self.n_positions
        )
        self.observation_space = gym.spaces.Discrete(n_observations)
        self.action_space = gym.spaces.Discrete(2)
        self.states = np.arange(n_observations)
        self.transition_matrix = self.T = self.get_transition_matrix()

    def _context_rewards(self, context_id: int) -> list[float]:
        """Return the reward vector induced by a given context."""
        return [1.0, 0.0, 0.0, 0.0, self._goal_reward(context_id)]

    def _follow_prob(self, context_id: int) -> float:
        """Return the action-follow probability for the context."""
        return float(self.contexts[context_id]["follow_prob"])

    def _goal_reward(self, context_id: int) -> float:
        """Return the goal reward for the context."""
        return float(self.contexts[context_id]["goal_reward"])

    def _reward_for_position(self, position: int, context_id: int) -> float:
        """Return the reward of a position under the current context."""
        return float(self._context_rewards(context_id)[position])

    def _base_next_position(self, position: int, action: int) -> int:
        """Move left or right while clipping the rover to valid positions."""
        if action == 0:
            return max(0, position - 1)
        return min(self.n_positions - 1, position + 1)

    def _base_transition(
        self, position: int, action: int, context_id: int
    ) -> np.ndarray:
        """Return the position-level transition distribution for one context."""
        next_distribution = np.zeros(self.n_positions, dtype=float)
        follow_state = self._base_next_position(position, action)
        flipped_state = self._base_next_position(position, 1 - action)
        follow_prob = self._follow_prob(context_id)

        next_distribution[follow_state] += follow_prob
        next_distribution[flipped_state] += 1.0 - follow_prob
        return next_distribution

    def _encode_state(self, position: int, context_id: int) -> int:
        """Encode a position and context as one discrete observation."""
        if not self.include_context:
            return position
        return context_id * self.n_positions + position

    def _select_next_context(self) -> int:
        """Choose the context used for the next episode."""
        if self.context_schedule == "round_robin":
            context_id = self.active_contexts[self._context_pointer]
            self._context_pointer = (self._context_pointer + 1) % len(
                self.active_contexts
            )
            return int(context_id)
        return int(self.rng.choice(self.active_contexts))

    def _build_info(self) -> dict[str, Any]:
        """Expose the active context in the info dictionary."""
        return {
            "context_id": self.current_context_id,
            "context": dict(self.contexts[self.current_context_id]),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment and select the next context.

        Parameters
        ----------
        seed : int, optional
            Seed for environment reset.
        options : dict, optional
            Additional reset options (unused).

        Returns
        -------
        observation : int
            Initial observation.
        info : dict
            Information about the sampled context.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_steps = 0
        self.position = self.start_position
        self.current_context_id = self._select_next_context()
        self.rewards = self._context_rewards(self.current_context_id)
        observation = self._encode_state(self.position, self.current_context_id)
        return observation, self._build_info()

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take one step under the currently active context.

        Parameters
        ----------
        action : int
            Action to take (0: left, 1: right).

        Returns
        -------
        observation : int
            Next observation.
        reward : float
            Immediate reward.
        terminated : bool
            Always False for this environment.
        truncated : bool
            Whether the horizon has been reached.
        info : dict
            Information about the active context.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        follow = self.rng.random() < self._follow_prob(self.current_context_id)
        executed_action = action if follow else 1 - action
        self.position = self._base_next_position(self.position, executed_action)

        reward = self._reward_for_position(self.position, self.current_context_id)
        observation = self._encode_state(self.position, self.current_context_id)
        terminated = False
        truncated = self.current_steps >= self.horizon

        return observation, reward, terminated, truncated, self._build_info()

    def get_transition_matrix(self) -> np.ndarray:
        """
        Return the transition matrix induced by the selected representation.

        If the context is observed, the transition matrix keeps context blocks
        separate. If the context is hidden, the transition matrix is averaged
        across the active contexts.
        """
        nS = self.observation_space.n
        nA = self.action_space.n
        T = np.zeros((nS, nA, nS), dtype=float)

        if self.include_context:
            for context_id in range(len(self.contexts)):
                for position in range(self.n_positions):
                    state = self._encode_state(position, context_id)
                    for action in self.actions:
                        base_distribution = self._base_transition(
                            position,
                            int(action),
                            context_id,
                        )
                        for next_position, probability in enumerate(base_distribution):
                            next_state = self._encode_state(next_position, context_id)
                            T[state, action, next_state] += probability
        else:
            context_weight = 1.0 / len(self.active_contexts)
            for context_id in self.active_contexts:
                for position in range(self.n_positions):
                    for action in self.actions:
                        T[position, action, :] += (
                            context_weight
                            * self._base_transition(
                                position,
                                int(action),
                                context_id,
                            )
                        )

        assert np.allclose(T.sum(axis=2), 1.0)
        return T

    def get_reward_per_action(self) -> np.ndarray:
        """
        Return expected one-step rewards for each state-action pair.

        The hidden-context case averages expected rewards across the active
        contexts, matching the averaged transition matrix used for planning.
        """
        nS = self.observation_space.n
        nA = self.action_space.n
        R = np.zeros((nS, nA), dtype=float)

        if self.include_context:
            for context_id in range(len(self.contexts)):
                for position in range(self.n_positions):
                    state = self._encode_state(position, context_id)
                    for action in self.actions:
                        base_distribution = self._base_transition(
                            position,
                            int(action),
                            context_id,
                        )
                        R[state, action] = sum(
                            probability
                            * self._reward_for_position(next_position, context_id)
                            for next_position, probability in enumerate(
                                base_distribution
                            )
                        )
        else:
            context_weight = 1.0 / len(self.active_contexts)
            for context_id in self.active_contexts:
                for position in range(self.n_positions):
                    for action in self.actions:
                        base_distribution = self._base_transition(
                            position,
                            int(action),
                            context_id,
                        )
                        R[position, action] += context_weight * sum(
                            probability
                            * self._reward_for_position(next_position, context_id)
                            for next_position, probability in enumerate(
                                base_distribution
                            )
                        )

        return R

    def render(self, mode: str = "human"):
        """Render the current state and active context."""
        context_name = self.contexts[self.current_context_id]["name"]
        print(
            "[ContextualMarsRover] "
            f"context={context_name}, pos={self.position}, steps={self.current_steps}"
        )


class MarsRoverPartialObsWrapper(gym.Wrapper):
    """
    Partially-observable wrapper for the MarsRover environment.

    This wrapper injects observation noise to simulate partial observability.
    With a specified probability, the true state (position) is replaced by a randomly
    selected incorrect position in the state space.

    Parameters
    ----------
    env : MarsRover
        The fully observable MarsRover environment to wrap.
    noise : float, default=0.1
        Probability in [0, 1] of returning a random incorrect position.
    seed : int or None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: MarsRover, noise: float = 0.1, seed: int | None = None):
        """
        Initialize the partial observability wrapper.

        Parameters
        ----------
        env : MarsRover
            The environment to wrap.
        noise : float, optional
            Probability of observing an incorrect state, by default 0.1.
        seed : int or None, optional
            Random seed for reproducibility, by default None.
        """
        super().__init__(env)
        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the base environment and return a noisy observation.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the reset, by default None.
        options : dict or None, optional
            Additional reset options, by default None.

        Returns
        -------
        obs : int
            The (possibly noisy) initial observation.
        info : dict
            Additional info returned by the environment.
        """
        true_obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy_obs(true_obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment and return a noisy observation.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        obs : int
            The (possibly noisy) resulting observation.
        reward : float
            The reward received.
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode was truncated due to time limit.
        info : dict
            Additional information from the base environment.
        """
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info

    def _noisy_obs(self, true_obs: int) -> int:
        """
        Return a possibly noisy version of the true observation.

        With probability `noise`, replaces the true observation with
        a randomly selected incorrect state.

        Parameters
        ----------
        true_obs : int
            The true observation/state index.

        Returns
        -------
        obs : int
            A noisy (or true) observation.
        """
        if self.rng.random() < self.noise:
            n = self.observation_space.n
            others = [s for s in range(n) if s != true_obs]
            return int(self.rng.choice(others))
        else:
            return int(true_obs)

    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str, optional
            Render mode, by default "human".

        Returns
        -------
        Any
            Rendered output from the base environment.
        """
        return self.env.render(mode=mode)
