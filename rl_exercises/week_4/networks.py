from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple MLP mapping states to Q-values for each action.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size.
        hidden_layers : int
            Number of hidden Linear/ReLU blocks.  The default keeps the
            original two-hidden-layer architecture.
        """
        super().__init__()
        assert hidden_layers >= 0, "hidden_layers must be non-negative"

        layers = OrderedDict()
        in_dim = obs_dim
        for layer_idx in range(hidden_layers):
            layers[f"fc{layer_idx + 1}"] = nn.Linear(in_dim, hidden_dim)
            layers[f"relu{layer_idx + 1}"] = nn.ReLU()
            in_dim = hidden_dim
        layers["out"] = nn.Linear(in_dim, n_actions)

        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)
