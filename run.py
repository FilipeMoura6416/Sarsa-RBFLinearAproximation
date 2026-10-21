from  sarsa_linear_agent import LinearSarsaAgent
from feature_extractors import *
import gymnasium as gym

lr_lists  = [0.005, 0.01, 0.02, 0.05, 0.1]
n_centers_list = [9, 16, 25, 36, 49]
seeds_list = [12]
total_steps = 50_000


env = gym.make('MountainCar-v0')
for learning_rate in lr_lists:
    for n_centers in n_centers_list:
        
        feature_extractor = RBFFeatureExtractor(env=env, n_centers=n_centers, sigma=0.2)
        for seed in seeds_list:
            agent = LinearSarsaAgent(
                    env=env,
                    feature_extractor=feature_extractor,
                    learning_rate=learning_rate,
                    epsilon=1.0,
                    epsilon_decay=0.999,
                    epsilon_min=0.01,
                    discount_factor=1.0
                )
            r_per_step, r_per_eps = agent.train(total_steps=total_steps,seed=seed)
            
            