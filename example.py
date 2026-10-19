"""
Example script demonstrating how to use the SARSA Linear Function Approximation Agent
with the new modular structure and step-based training.
"""

import numpy as np
import gymnasium as gym
from sarsa_linear_agent import LinearSarsaAgent
from feature_extractors import RBFFeatureExtractor


def quick_demo_steps():
    """Quick demonstration with step-based training."""
    print("Running a minimal quick demo (single guarded step)...")

    # Create environment
    env = gym.make('MountainCar-v0')

    # Create feature extractor with simpler settings for quick demo
    feature_extractor = RBFFeatureExtractor(env=env, n_centers=16, sigma=0.2)

    # Create agent
    agent = LinearSarsaAgent(
        env=env,
        feature_extractor=feature_extractor,
        learning_rate=0.02,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        discount_factor=1.0
    )

    print(f"Feature vector size: {feature_extractor.n_features}")
    print(f"Number of actions: {env.action_space.n}")

    # Do one guarded step using agent.act() so the demo won't crash
    # Run a short, step-based training session (1000 steps).
    # This will exercise the agent's training loop (act/updateQ/train).
    try:
        step_rewards, episode_rewards = agent.train(1000)
    except NotImplementedError:
        print("agent.train() not implemented: implement train(total_steps) to run the demo.")
        env.close()
        return
    except Exception as e:
        print(f"agent.train() raised {type(e).__name__}: {e}")
        env.close()
        return

    # Summary
    print(f"Completed demo training: steps={len(step_rewards)}, episodes={len(episode_rewards)}")

    env.close()




if __name__ == "__main__":
    quick_demo_steps()
    