from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent


class TDLambdaPredictionAgent(AbstractAgent):
    """Tabular TD(lambda) prediction agent with accumulating traces.

    The SARSA and Q-learning agent in this week learns Q(s, a), because those
    algorithms solve control tasks.  Sutton's 1988 paper first introduces
    TD(lambda) as a prediction method: given a sequence of observations, learn
    the value V(s), i.e. the expected final outcome from each state.

    This class therefore keeps a one-dimensional value table V instead of a
    Q-table.  The key extra object compared with TD(0) is `eligibility`: it
    remembers which earlier states deserve credit for the latest temporal
    difference error.  Lambda controls how quickly that credit fades:

    - lambda = 0.0 updates only the most recent state, matching TD(0)
    - lambda = 1.0 keeps all states in the episode eligible, matching the
      Widrow-Hoff/supervised-learning endpoint discussed in the paper
    - values between 0 and 1 interpolate between those two extremes
    """

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.3,
        initial_value: float = 0.5,
    ) -> None:
        """Initialize the prediction learner.

        Parameters
        ----------
        env : gym.Env
            Environment with a discrete observation space.  The implementation
            is intentionally tabular to mirror Sutton's random-walk example.
        alpha : float, optional
            Learning rate.
        gamma : float, optional
            Discount factor.  The paper's random-walk prediction task uses 1.0.
        lam : float, optional
            Eligibility-trace decay parameter lambda.
        initial_value : float, optional
            Initial prediction for all states.  Sutton initializes the
            nonterminal random-walk predictions to 0.5 in the single-presentation
            experiment so that the learner is not biased toward either terminal.
        """
        assert hasattr(env.observation_space, "n"), "TD(lambda) needs tabular states"
        assert hasattr(env.action_space, "n"), "Environment needs a discrete action"
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert 0 <= lam <= 1, "Lambda should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"

        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.initial_value = float(initial_value)

        self.n_states = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)
        self.V = np.full(self.n_states, self.initial_value, dtype=float)
        self.eligibility = np.zeros(self.n_states, dtype=float)

        # In the random-walk encoding, terminal outcomes arrive as rewards on
        # the final transition.  Terminal table entries are placeholders, so we
        # keep them at 0 and never rely on them for bootstrapping.
        for terminal_attr in ("left_terminal", "right_terminal"):
            if hasattr(env, terminal_attr):
                self.V[int(getattr(env, terminal_attr))] = 0.0

    def predict_action(
        self, state: Any, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """Return the only meaningful action in a passive prediction problem."""
        return 0, {} if info is None else info

    def save(self, path: str) -> Any:  # type: ignore[override]
        """Save the value table."""
        np.save(path, self.V)

    def load(self, path: str) -> Any:  # type: ignore[override]
        """Load a value table and reset old eligibility traces."""
        self.V = np.load(path, allow_pickle=False)
        self.eligibility = np.zeros_like(self.V, dtype=float)

    def update_agent(self, batch: list) -> float | None:  # type: ignore[override]
        """Update V from the latest transition in the repository's buffer."""
        if not batch or batch[0] is None:
            return None

        state, _, reward, next_state, done, _ = batch[0]
        return self.TD_lambda(state, reward, next_state, done)

    def TD_lambda(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Perform one online TD(lambda) value update.

        The temporal-difference error is

            delta = reward + gamma * V(next_state) - V(state)

        except that terminal states do not bootstrap.  After computing delta,
        the current state's eligibility is increased and all value entries are
        moved by alpha * delta * eligibility.
        """
        state = int(state)
        next_state = int(next_state)
        reward = float(reward)

        next_prediction = 0.0 if done else self.V[next_state]
        td_error = reward + self.gamma * next_prediction - self.V[state]

        # Accumulating trace from Sutton's Equation (4): recent states get the
        # largest update, earlier states still receive credit scaled by lambda.
        self.eligibility *= self.gamma * self.lam
        self.eligibility[state] += 1.0

        self.V += self.alpha * td_error * self.eligibility

        # Episodes are independent sequences in the paper.  Carrying traces
        # across terminal boundaries would assign credit to states from a
        # previous walk, so the trace memory is cleared after each outcome.
        if done:
            self.reset_traces()

        return float(self.V[state])

    def reset_traces(self) -> None:
        """Forget eligibility from the previous episode."""
        self.eligibility.fill(0.0)

    def rms_error(
        self, true_values: np.ndarray, states: np.ndarray | None = None
    ) -> float:
        """Compute RMS prediction error on the nonterminal states."""
        true_values = np.asarray(true_values, dtype=float)
        if states is None and hasattr(self.env, "nonterminal_states"):
            states = np.asarray(getattr(self.env, "nonterminal_states"), dtype=int)
        elif states is None:
            states = np.arange(self.n_states, dtype=int)
        return float(np.sqrt(np.mean((self.V[states] - true_values[states]) ** 2)))
