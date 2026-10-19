"""
Skeleton SARSA agent for students to implement.

This file intentionally provides method signatures, docstrings, and
lightweight hints. Students should implement the core algorithms as part
of the assignment.

Tasks for students:
- Implement linear-function SARSA updates in `updateQ`.
- Implement epsilon-greedy action selection in `act`.
- Implement a training loop that uses step-based training in `train`.
- Fill in save/load helpers to persist model weights and feature extractor config.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import gymnasium as gym
import pickle
from feature_extractors import FeatureExtractor


class LinearSarsaAgent:
    """Student skeleton for a SARSA agent using linear function approximation.

    The class provides the public API expected by the rest of the assignment
    (methods and attributes). Students implement the algorithmic details.
    """

    def __init__(
        self,
        env: gym.Env,
        feature_extractor: FeatureExtractor,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        discount_factor: float = 1.0,
    ):
        self.env = env
        self.n_actions = env.action_space.n
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor

        # Students should initialize weights to zeros with shape
        # (n_actions, feature_extractor.n_features)
        self.weights = [] #TODO: Initialize properly

        # Track training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def Q(self, state: np.ndarray, action: int) -> float:
        """Compute linear Q-value for (state,action).

        compute the dot product between weights and features.
        """
        raise NotImplementedError()

    def V(self, state: np.ndarray) -> float:
        """Return the maximum Q over actions for a state."""
        raise NotImplementedError()

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection (student to implement).

        HINT: With probability epsilon choose random action, otherwise argmax Q.
        """
        # Student implementation required
        raise NotImplementedError("Implement epsilon-greedy policy in act()")

    def updateQ(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool,
    ) -> None:
        """SARSA weight update 
        """
        raise NotImplementedError("Implement SARSA update rule in updateQ()")

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode (simple multiplicative decay).
        Don't decay below self.epsilon_min
        """
        raise NotImplementedError("Implement epsilon decay in decay_epsilon()")

    def train(self, total_steps: int) -> Tuple[List[float], List[float]]:
        """Train the agent for a fixed number of environment steps.

        Students should implement a step-based loop that:
        - interacts with the environment
        - performs SARSA updates via `updateQ`
        - decays epsilon at episode ends and records statistics

        Returns two lists: rewards per step and rewards per episode.
        """
        raise NotImplementedError("Implement training loop in train()")

    def save_model(self, filepath: str) -> None:
        """Save weights and feature extractor config.

        Students: persist `self.weights` and a small dict describing the
        feature extractor parameters (e.g., type, n_features, n_centers, sigma).
        """
        model_data = {
            'weights': self.weights,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'feature_extractor_config': {
                'type': type(self.feature_extractor).__name__,
                'n_features': self.feature_extractor.n_features,
            },
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """Load persisted model data (students should implement robustly)."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.weights = model_data['weights']
        self.epsilon = model_data.get('epsilon', self.epsilon)
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_lengths = model_data.get('episode_lengths', [])