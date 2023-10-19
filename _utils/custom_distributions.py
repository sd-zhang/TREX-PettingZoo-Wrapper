"""Probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution, TanhBijector

SelfSquashedDiagGaussianDistribution = TypeVar(
    "SelfSquashedDiagGaussianDistribution", bound="SquashedDiagGaussianDistribution"
)

class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6, std_bias: float = -3.0,):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None
        self.std_bias = std_bias

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        nn.init.uniform_(log_std.weight, a=-1e-2, b=1e-2)
        # nn.init.uniform_(mean_actions.bias, a=-1e-3, b=-1e-3)
        # TODO: allow action dependent std
        log_std = nn.Linear(latent_dim, self.action_dim, bias=True)
        nn.init.uniform_(log_std.weight, a=-1e-2, b=1e-2)
        nn.init.uniform_(log_std.weight, a=-1e-2 + self.std_bias, b=1e-2 + self.std_bias)

        return mean_actions, log_std

    def proba_distribution(
        self: SelfSquashedDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfSquashedDiagGaussianDistribution:
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob