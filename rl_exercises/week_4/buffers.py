from typing import Any, Dict, List, Tuple

import numpy as np
from rl_exercises.agent import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    Simple FIFO replay buffer.

    Stores tuples of (state, action, reward, next_state, done, info),
    and evicts the oldest when capacity is exceeded.
    """

    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        """
        super().__init__()
        assert capacity > 0, "Replay buffer capacity must be positive"
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

    def _pop_oldest(self) -> None:
        """Remove the oldest transition from all storage lists."""
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.next_states.pop(0)
        self.dones.pop(0)
        self.infos.pop(0)

    def _transition_at(self, index: int) -> Tuple[Any, Any, float, Any, bool, Dict]:
        """Return one stored transition by index."""
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index],
            self.infos[index],
        )

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """
        Add a single transition to the buffer.

        If the buffer is full, the oldest transition is removed.

        Parameters
        ----------
        state : np.ndarray
            Observation before action.
        action : int or float
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Observation after action.
        done : bool
            Whether episode terminated/truncated.
        info : dict
            Gym info dict (can store extras).
        """
        if len(self.states) >= self.capacity:
            self._pop_oldest()

        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(bool(done))
        self.infos.append(dict(info))

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        """
        Uniformly sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        List of transitions as (state, action, reward, next_state, done, info).
        """
        if batch_size > len(self):
            raise ValueError("Cannot sample more transitions than are stored")
        idxs = np.random.choice(len(self.states), size=batch_size, replace=False)
        return [self._transition_at(int(i)) for i in idxs]

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with proportional prioritized experience replay.

    The buffer keeps the same public sample format as ReplayBuffer.  For sampled
    transitions, the returned info dict additionally contains:

    - ``buffer_index``: position of the transition in the buffer
    - ``sampling_weight``: importance-sampling weight for the loss
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        priority_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(capacity)
        assert alpha >= 0, "alpha must be non-negative"
        assert beta >= 0, "beta must be non-negative"
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.priority_epsilon = float(priority_epsilon)
        self.priorities: List[float] = []

    def _pop_oldest(self) -> None:
        """Remove the oldest transition and its priority."""
        super()._pop_oldest()
        self.priorities.pop(0)

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """Add a transition with maximum current priority."""
        max_priority = max(self.priorities, default=1.0)
        super().add(state, action, reward, next_state, done, info)
        self.priorities.append(float(max_priority))

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        """Sample transitions according to their priorities."""
        if batch_size > len(self):
            raise ValueError("Cannot sample more transitions than are stored")

        priorities = np.asarray(self.priorities, dtype=np.float64)
        scaled_priorities = priorities**self.alpha
        total_priority = scaled_priorities.sum()
        if total_priority <= 0:
            probabilities = np.full(len(self), 1.0 / len(self), dtype=np.float64)
        else:
            probabilities = scaled_priorities / total_priority

        idxs = np.random.choice(
            len(self.states), size=batch_size, replace=False, p=probabilities
        )
        weights = (len(self) * probabilities[idxs]) ** (-self.beta)
        weights /= weights.max()

        batch = []
        for idx, weight in zip(idxs, weights):
            transition = self._transition_at(int(idx))
            state, action, reward, next_state, done, info = transition
            enriched_info = dict(info)
            enriched_info["buffer_index"] = int(idx)
            enriched_info["sampling_weight"] = float(weight)
            batch.append((state, action, reward, next_state, done, enriched_info))
        return batch

    def update_priorities(
        self, indices: List[int] | np.ndarray, priorities: List[float] | np.ndarray
    ) -> None:
        """Update priorities after a learning step."""
        for index, priority in zip(indices, priorities):
            self.priorities[int(index)] = float(abs(priority) + self.priority_epsilon)
