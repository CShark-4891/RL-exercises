from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class BoundedRandomWalkEnv(gym.Env):
    """Bounded random walk from Sutton's TD(lambda) paper.

    Sutton's Section 3.2 uses the states A, B, C, D, E, F, G.  A and G are
    absorbing terminal states; every episode starts in D; from B-F the process
    moves one step left or right with probability 0.5 each.  The learning task is
    prediction, not control: estimate the probability that a walk terminates on
    the right side in G.

    To fit the Gymnasium training loop used in this repository, the terminal
    outcome z from the paper is encoded as a reward on the transition into a
    terminal state:

    - entering A gives reward 0.0
    - entering G gives reward 1.0
    - all nonterminal transitions give reward 0.0

    With gamma=1 and no bootstrapping from terminal states, this is equivalent to
    the paper's final outcome z in the TD error.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_nonterminal_states: int = 5,
        start_state: int | None = None,
        p_right: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Create the bounded random walk Markov process.

        Parameters
        ----------
        n_nonterminal_states : int, optional
            Number of learnable states between the two terminals.  Sutton's
            experiment uses 5, giving A-G in total.
        start_state : int or None, optional
            Integer state index to start from.  If None, the middle state is
            used; for Sutton's A-G walk this is D, index 3.
        p_right : float, optional
            Probability of moving to the right from any nonterminal state.
            Sutton's example uses 0.5.
        seed : int or None, optional
            Seed for the environment's private random generator.
        """
        assert n_nonterminal_states >= 1, "Need at least one nonterminal state"
        assert 0 <= p_right <= 1, "p_right must be in [0, 1]"

        self.n_nonterminal_states = int(n_nonterminal_states)
        self.n_states = self.n_nonterminal_states + 2
        self.left_terminal = 0
        self.right_terminal = self.n_states - 1
        self.start_state = (
            self.n_states // 2 if start_state is None else int(start_state)
        )
        if not 0 < self.start_state < self.right_terminal:
            raise ValueError("start_state must be one of the nonterminal states")

        self.p_right = float(p_right)
        self.rng = np.random.default_rng(seed)
        self.state = self.start_state

        # There is no decision to make in the paper's process.  The single
        # dummy action keeps the class compatible with the repository's generic
        # agent/environment loop.
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(self.n_states)

    @property
    def nonterminal_states(self) -> np.ndarray:
        """State indices whose predictions are learned."""
        return np.arange(1, self.right_terminal, dtype=int)

    @property
    def true_values(self) -> np.ndarray:
        """Exact probability of right-side termination for every state.

        For an unbiased bounded random walk, the probability of hitting the
        right boundary before the left boundary is linear in the position.  In
        Sutton's A-G walk this gives B-F = 1/6, 2/6, ..., 5/6.
        """
        return np.linspace(0.0, 1.0, self.n_states)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset the walk to the middle state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start_state
        return self.state, self._info()

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance the Markov process by one random left/right move."""
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not valid; use the dummy action 0")
        if self.state in {self.left_terminal, self.right_terminal}:
            raise RuntimeError("Cannot call step() after the episode has terminated")

        direction = 1 if self.rng.random() < self.p_right else -1
        self.state += direction

        terminated = self.state in {self.left_terminal, self.right_terminal}
        truncated = False
        reward = 1.0 if self.state == self.right_terminal else 0.0
        return self.state, reward, terminated, truncated, self._info()

    def render(self, mode: str = "human") -> None:
        """Print the current walk state."""
        print(f"[BoundedRandomWalk] state={self.state_label(self.state)}")

    def state_label(self, state: int) -> str:
        """Return Sutton's A-G labels for the default seven-state walk."""
        state = int(state)
        if self.n_states <= 26:
            return chr(ord("A") + state)
        return str(state)

    def observation_vector(self, state: int) -> np.ndarray:
        """One-hot observation vector x_i used in Sutton's linear example.

        The paper represents B-F as five unit basis vectors.  Terminals are not
        prediction states, so asking for their observation vector is a modeling
        error and should be caught early.
        """
        state = int(state)
        if state not in self.nonterminal_states:
            raise ValueError("Only nonterminal states have observation vectors")
        vector = np.zeros(self.n_nonterminal_states, dtype=float)
        vector[state - 1] = 1.0
        return vector

    def _info(self) -> dict[str, Any]:
        """Small diagnostics that make trajectories easier to inspect."""
        return {
            "state_label": self.state_label(self.state),
            "is_terminal": self.state in {self.left_terminal, self.right_terminal},
            "right_terminal_probability": float(self.true_values[self.state]),
        }
