import os
import pickle
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from MiniGrid import MiniGrid
from gymnasium.wrappers import FilterObservation, FlattenObservation


class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.history = {"return": [],"success": [],"intrinsic_reward": [],"state_coverage": [], "extrinsic_return": []}

    def _on_step(self):
        env = self.training_env.envs[0]

        while len(env.completed_episodes) > 0:
            ep = env.completed_episodes.pop(0)
            for k in self.history:
                self.history[k].append(ep[k])

        return True

class MetricsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.visit_counts = {}
        self.intrinsic_reward_scale = 0.05
        self.episode_trajectory = []
        self.all_trajectories = []
        self.visit_heatmap = np.zeros((env.unwrapped.height, env.unwrapped.width))

        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_states = set()

        self.completed_episodes = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        x, y = self.unwrapped.agent_pos
        self.episode_trajectory = [(x, y)]
        self.visit_heatmap[y, x] += 1

        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_states = set()

        state_key = str(obs)
        self.episode_states.add(state_key)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y = self.unwrapped.agent_pos
        self.episode_trajectory.append((x, y))
        self.visit_heatmap[y, x] += 1

        state_key = str(obs)
        self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1
        intrinsic_reward = self.intrinsic_reward_scale / np.sqrt(self.visit_counts[state_key])
        info["intrinsic_reward"] = intrinsic_reward

        total_reward = reward + intrinsic_reward
        self.episode_return += total_reward
        self.episode_intrinsic_reward += intrinsic_reward
        self.episode_extrinsic_return += reward
        self.episode_states.add(str(obs))

        if reward > 0:
            self.episode_success = 1

        done = terminated or truncated

        if done:
            self.all_trajectories.append(self.episode_trajectory)
            self.completed_episodes.append({
                "return": self.episode_return,
                "success": self.episode_success,
                "intrinsic_reward": self.episode_intrinsic_reward,
                "state_coverage": len(self.episode_states),
                "extrinsic_return": self.episode_extrinsic_return,
            })

        return obs, total_reward, terminated, truncated, info


def make_env():
    env = MiniGrid(render_mode=None)

    env = FilterObservation(env, ["image", "direction"])
    env = FlattenObservation(env)

    env = MetricsWrapper(env)

    return env

vec_env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=256,
    batch_size=64,
    ent_coef=0.01,
    verbose=1
)

callback = MetricsCallback()

model.learn(total_timesteps=20000, callback=callback)

returns = callback.history["return"]
successes = callback.history["success"]
intrinsic_rewards = callback.history["intrinsic_reward"]
coverages = callback.history["state_coverage"]

print("Episodes logged:", len(successes))
print("Average return:", np.mean(returns))
print("Success rate:", np.mean(successes) * 100, "%")
print("Average intrinsic reward:", np.mean(intrinsic_rewards))
print("Average state coverage:", np.mean(coverages))
print("Average extrinsic return:", np.mean(callback.history["extrinsic_return"]))

if 1 in successes:
    print("Time to first success:", successes.index(1) + 1, "episodes")
else:
    print("Time to first success: not achieved")


env = vec_env.envs[0]

trajectory = env.all_trajectories[-1]

xs = [p[0] for p in trajectory]
ys = [p[1] for p in trajectory]

plt.figure(figsize=(6, 6))
plt.plot(xs, ys, marker="o")
plt.gca().invert_yaxis()
plt.title("Agent Trajectory")
plt.xlabel("x position")
plt.ylabel("y position")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(env.visit_heatmap)
plt.colorbar(label="Visit count")
plt.title("Visited State Heatmap")
plt.xlabel("x position")
plt.ylabel("y position")
plt.show()