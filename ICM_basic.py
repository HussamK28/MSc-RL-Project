import os
import pickle
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ICM(nn.Module):
    def __init__(self, observation_dim, action_dim, feature_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, observation, next_observation, action):
        phi = self.encoder(observation)
        next_phi = self.encoder(next_observation)

        inverse_input = torch.cat([phi, next_phi], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat([phi, action_onehot], dim=1)
        predicted_next_phi = self.forward_model(forward_input)

        icm_reward = F.mse_loss(predicted_next_phi, next_phi.detach(), reduction="none").mean(dim=1)
        inverse_loss = F.cross_entropy(predicted_action, action)
        forward_loss = icm_reward.mean()
        icm_loss = inverse_loss + 0.2 * forward_loss

        return icm_reward.detach(), icm_loss

class MetricsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.icm = icm
        self.icm_optimiser = icm_optimiser
        self.device = device
        self.previous_observations = None
        self.intrinsic_reward_scale = 0.01
        self.episode_trajectory = []
        self.all_trajectories = []
        self.visit_heatmap = np.zeros((env.unwrapped.height, env.unwrapped.width))

        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_states = set()

        self.completed_episodes = []
        # ----- TESTING -----
        self.key1_reached = 0
        self.door1_opened = 0
        self.key2_reached = 0
        self.door2_opened = 0

        self.ep_key1 = False
        self.ep_door1 = False
        self.ep_key2 = False
        self.ep_door2 = False
        # ----- TESTING -----

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_observations = obs

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

        # ----- TESTING -----
        self.ep_key1 = False
        self.ep_door1 = False
        self.ep_key2 = False
        self.ep_door2 = False
        # ----- TESTING -----

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

         # ----- TESTING -----
        grid = self.unwrapped.grid
        if (self.unwrapped.carrying is not None and self.unwrapped.carrying.color == self.unwrapped.key1_colour):
            self.ep_key1 = True
        
        door1 = grid.get(self.unwrapped.wall1, self.unwrapped.door1_pos)

        if door1 is not None and door1.is_open:
            self.ep_door1 = True

        if (self.unwrapped.carrying is not None and self.unwrapped.carrying.color == self.unwrapped.key2_colour):
            self.ep_key2 = True
        
        door2 = grid.get(self.unwrapped.wall2, self.unwrapped.door2_pos)

        if door2 is not None and door2.is_open:
            self.ep_door2 = True

        # ----- TESTING -----

        x, y = self.unwrapped.agent_pos
        self.episode_trajectory.append((x, y))
        self.visit_heatmap[y, x] += 1

        observation_tensor = torch.tensor(self.previous_observations, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_observation_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)

        intrinsic_reward, icm_loss = self.icm(
            observation_tensor,
            next_observation_tensor,
            action_tensor
        )

        self.icm_optimiser.zero_grad()
        icm_loss.backward()
        self.icm_optimiser.step()

        intrinsic_reward = intrinsic_reward.item() * self.intrinsic_reward_scale
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
            if self.ep_key1:
                self.key1_reached += 1

            if self.ep_door1:
                self.door1_opened += 1

            if self.ep_key2:
                self.key2_reached += 1

            if self.ep_door2:
                self.door2_opened += 1

        self.previous_observations = obs
        return obs, total_reward, terminated, truncated, info


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp_env = MiniGrid(size=16, max_steps=500, render_mode=None)
temp_env = FilterObservation(temp_env, ["image", "direction"])
temp_env = FlattenObservation(temp_env)

obs_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.n

icm = ICM(obs_dim, action_dim).to(device)
icm_optimiser = torch.optim.Adam(icm.parameters(), lr=1e-4)

def make_env():
    env = MiniGrid(size=16, max_steps=500, render_mode=None)

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

model.learn(total_timesteps=500000, callback=callback)
file_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

save_dir = os.path.join("results", file_name)
os.makedirs(save_dir, exist_ok=True)

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
print("Observation dim:", obs_dim)
print("Action dim:", action_dim)
print("Intrinsic reward:", intrinsic_rewards)

if 1 in successes:
    print("Time to first success:", successes.index(1) + 1, "episodes")
else:
    print("Time to first success: not achieved")


env = vec_env.envs[0]
episodes = len(successes)

if episodes > 0:
    print("Picked up key1:", 100 * env.key1_reached / episodes, "%")
    print("Opened door1:", 100 * env.door1_opened / episodes, "%")
    print("Picked up key2:", 100 * env.key2_reached / episodes, "%")
    print("Opened door2:", 100 * env.door2_opened / episodes, "%")
else:
    print("No episodes completed.")

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
plt.savefig(
    os.path.join(save_dir, "trajectory_graph.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(env.visit_heatmap)
plt.colorbar(label="Visit count")
plt.title("Visited State Heatmap")
plt.xlabel("x position")
plt.ylabel("y position")
plt.savefig(
    os.path.join(save_dir, "heatmap_chart.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

with open(os.path.join(save_dir, "run_trajectories.pkl"), "wb") as f:
    pickle.dump(env.all_trajectories, f)
np.save(os.path.join(save_dir, "visit_heatmap.npy"),env.visit_heatmap)
metrics = {
    "episodes_logged": len(successes),
    "success_rate": np.mean(successes),
    "avg_return": np.mean(returns),
    "avg_intrinsic_reward": np.mean(intrinsic_rewards),
    "avg_state_coverage": np.mean(coverages),
    "avg_extrinsic_return": np.mean(callback.history["extrinsic_return"]),
    "time_to_first_success":
        successes.index(1) + 1 if 1 in successes else None
}

with open(os.path.join(save_dir, "metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)