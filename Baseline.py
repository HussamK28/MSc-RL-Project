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
import torch


class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.history = {"return": [],"success": [],"intrinsic_reward": [],"state_coverage": [], "extrinsic_return": [], "key1": [], "door1": [], "key2": [], "door2": [], "door1_with_key": []}

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
        self.episode_trajectory = []
        self.all_trajectories = []
        self.visit_heatmap = np.zeros((env.unwrapped.height, env.unwrapped.width))

        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_states = set()

        self.completed_episodes = []
        self.use_subgoal_rewards = False

        self.key1_reached = 0
        self.door1_opened = 0
        self.key2_reached = 0
        self.door2_opened = 0
        self.door1_reached_with_key = 0

        self.ep_key1 = False
        self.ep_door1 = False
        self.ep_key2 = False
        self.ep_door2 = False
        self.ep_door1_reached_with_key = False


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_states = set()

        self.ep_key1 = False
        self.ep_door1 = False
        self.ep_key2 = False
        self.ep_door2 = False
        self.ep_door1_reached_with_key = False
        x, y = self.unwrapped.agent_pos
        self.episode_trajectory = [(x, y)]
        self.visit_heatmap[y, x] += 1
        self.episode_states.add(self.state_key())

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        had_key1 = self.ep_key1
        had_door1 = self.ep_door1
        had_key2 = self.ep_key2
        had_door2 = self.ep_door2

        self.update_subgoal_metrics()

        shaping_reward = 0.0

        if self.use_subgoal_rewards:
            if self.ep_key1 and not had_key1:
                shaping_reward += 0.05

            if self.ep_door1 and not had_door1:
                shaping_reward += 0.10

            if self.ep_key2 and not had_key2:
                shaping_reward += 0.15

            if self.ep_door2 and not had_door2:
                shaping_reward += 0.20

        training_reward = float(reward) + shaping_reward
        x, y = self.unwrapped.agent_pos
        self.episode_trajectory.append((x, y))
        self.visit_heatmap[y, x] += 1


        self.episode_return += training_reward
        self.episode_intrinsic_reward += 0
        self.episode_extrinsic_return += reward
        self.episode_states.add(self.state_key())

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
                "key1": int(self.ep_key1),
                "door1": int(self.ep_door1),
                "key2": int(self.ep_key2),
                "door2": int(self.ep_door2),
                "door1_with_key": int(self.ep_door1_reached_with_key)
            })
            if self.ep_key1:
                self.key1_reached += 1

            if self.ep_door1:
                self.door1_opened += 1

            if self.ep_key2:
                self.key2_reached += 1

            if self.ep_door2:
                self.door2_opened += 1

            self.door1_reached_with_key += int(self.ep_door1_reached_with_key)

        return obs, training_reward, terminated, truncated, info

    def state_key(self):
        x, y = self.unwrapped.agent_pos
        carrying = self.unwrapped.carrying
        carried_object = None

        if carrying is not None:
            carried_object = (carrying.type, carrying.color)
        
        return int(x), int(y), int(self.unwrapped.agent_dir), carried_object, bool(self.ep_door1), bool(self.ep_door2)
    
    def update_subgoal_metrics(self):
        grid = self.unwrapped.grid
        carrying = self.unwrapped.carrying
        if (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key1_colour):
            self.ep_key1 = True
        
        door1 = grid.get(self.unwrapped.wall1, self.unwrapped.door1_pos)
        if door1 is not None and door1.is_open:
            self.ep_door1 = True
        
        if (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key2_colour):
            self.ep_key2 = True
        
        door2 = grid.get(self.unwrapped.wall2, self.unwrapped.door2_pos)
        if door2 is not None and door2.is_open:
            self.ep_door2 = True
        
        agent_x, agent_y = self.unwrapped.agent_pos
        door1_distance = (abs(int(agent_x) - int(self.unwrapped.wall1)) + abs(int(agent_y) - int(self.unwrapped.door1_pos)))
        carrying_key1 = (carrying is not None and carrying.type == "key" and carrying.color == self.unwrapped.key1_colour)
        if carrying_key1 and door1_distance == 1:
            self.ep_door1_reached_with_key = True

def make_env():
    env = MiniGrid(size=12, max_steps=400, noisy_tv=False, fixed_layout=True, render_mode=None)

    env = FilterObservation(env, ["image", "direction"])
    env = FlattenObservation(env)

    env = MetricsWrapper(env)

    return env

vec_env = DummyVecEnv([make_env])
vec_env.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=2.5e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256]
        }
    },
    verbose=1,
    seed=42
)

callback = MetricsCallback()

def mean_last(values, window=100):
    if len(values) == 0:
        return 0.0

    window = min(window, len(values))
    return float(np.mean(values[-window:]))

model.learn(total_timesteps=500_000, callback=callback)
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
print("*******************")
print("Success rate over last 100 episodes: ", mean_last(successes, 100) * 100, "%")
print("Average coverage over last 100 episodes: ", mean_last(coverages, 100))
print("Average extrinsic return over last 100 episodes: ", mean_last(callback.history["extrinsic_return"], 100))
print("*******************")

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
    print("Reached door1 with key1: ", 100 * env.door1_reached_with_key / episodes, "%")
else:
    print("No episodes completed.")

trajectory = env.all_trajectories[-1]
def rolling_mean(values, window=100):
    values = np.asarray(values, dtype=np.float32)

    if len(values) < window:
        return np.array([])

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid"
    )

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