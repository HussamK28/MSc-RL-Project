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
        self.history = {"return": [],
                        "success": [],
                        "intrinsic_reward": [],
                        "state_coverage": [], 
                        "extrinsic_return": [], 
                        "key1": [], 
                        "door1": [], 
                        "key2": [], 
                        "door2": [], 
                        "door1_with_key": [], 
                        "mean_intrinsic_per_step":[],
                        "mean_prediction_error": [],
                        "mean_learning_progress":[],
                        "mean_fast_pred_error":[],
                        "mean_slow_pred_error":[],
                        "positive_lp_fraction": [],
                        }

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
            nn.Linear(256, feature_dim),
            nn.Tanh()
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

        forward_error = F.mse_loss(predicted_next_phi, next_phi.detach(), reduction="none").mean(dim=1)
        inverse_loss = F.cross_entropy(predicted_action, action)
        forward_loss = forward_error.mean()

        beta = 0.2
        icm_loss = ((1.0 - beta) * inverse_loss + beta * forward_loss)

        return (forward_error.detach(), icm_loss, forward_loss.detach(), inverse_loss.detach())

class MetricsWrapper(gym.Wrapper):
    def __init__(self, env, icm, icm_optimiser, device):
        super().__init__(env)
        self.icm = icm
        self.icm_optimiser = icm_optimiser
        self.device = device
        self.previous_observations = None
        self.intrinsic_reward_scale = 1.0

        self.learning_progress_fast = 0.01
        self.learning_progress_slow = 0.001
        self.fast_pred_error = None
        self.slow_pred_error = None
        self.learning_progress_clip = 0.1

        self.episode_trajectory = []
        self.all_trajectories = []
        self.visit_heatmap = np.zeros((env.unwrapped.height, env.unwrapped.width))

        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_states = set()

        self.completed_episodes = []


        self.key1_reached = 0
        self.door1_opened = 0
        self.key2_reached = 0
        self.door2_opened = 0
        self.door1_reached_with_key = 0

        self.reset_episode_metrics()
    
    def calculate_learning_progress(self, prediction_error):
        prediction_error = float(prediction_error)
        if self.fast_pred_error is None:
            self.fast_pred_error = prediction_error
            self.slow_pred_error = prediction_error
            return 0.0
        
        self.fast_pred_error = (self.learning_progress_fast * prediction_error + (1.0 - self.learning_progress_fast) * self.fast_pred_error)
        self.slow_pred_error = (self.learning_progress_slow * prediction_error + (1.0 - self.learning_progress_slow) * self.slow_pred_error)

        learning_progress = self.slow_pred_error - self.fast_pred_error
        learning_progress = max(learning_progress, 0.0)

        return float(np.clip(learning_progress,0.0, self.learning_progress_clip))


    def reset_episode_metrics(self):
        self.episode_return = 0
        self.episode_intrinsic_reward = 0
        self.episode_success = 0
        self.episode_extrinsic_return = 0
        self.episode_steps = 0
        self.episode_positive_lp_steps = 0
        self.episode_states = set()

        self.ep_key1 = False
        self.ep_door1 = False
        self.ep_key2 = False
        self.ep_door2 = False
        self.ep_door1_reached_with_key = False

        self.episode_prediction_errors = []
        self.episode_learning_progress = []
        self.episode_fast_errors = []
        self.episode_slow_errors = []


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_observations = self.normalise_observations(obs)
        self.reset_episode_metrics()

        x, y = self.unwrapped.agent_pos
        self.episode_trajectory = [(x, y)]
        self.visit_heatmap[y, x] += 1

        self.episode_states.add(self.state_key())
        return obs, info

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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_subgoal_metrics()

        x, y = self.unwrapped.agent_pos
        self.episode_trajectory.append((int(x), int(y)))
        self.visit_heatmap[y, x] += 1

        normalised_next_obs = self.normalise_observations(obs)

        observation_tensor = torch.tensor(self.previous_observations, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_observation_tensor = torch.tensor(normalised_next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        (prediction_error_tensor, icm_loss, forward_loss, inverse_loss) = self.icm(
            observation_tensor,
            next_observation_tensor,
            action_tensor
        )

        self.icm_optimiser.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), max_norm=0.5)
        self.icm_optimiser.step()

        prediction_error = float(prediction_error_tensor.item())
        
        learning_progress = self.calculate_learning_progress(prediction_error)
        scaled_prediction_error = float(np.clip(prediction_error, 0.0, 0.1))
        scaled_learning_progress = float(np.clip(learning_progress, 0.0, 0.1))
        prediction_error_weight = 0.2
        learning_progress_weight = 0.8
        
        hybrid_reward = (prediction_error_weight * scaled_prediction_error + learning_progress_weight * scaled_learning_progress)

        intrinsic_reward = hybrid_reward * self.intrinsic_reward_scale
        intrinsic_reward = float(np.clip(intrinsic_reward, 0.0, 0.1))
        extrinsic_reward = float(reward)
        total_reward = intrinsic_reward + extrinsic_reward

        info["intrinsic_reward"] = intrinsic_reward
        info["prediction_error"] = prediction_error
        info["learning_progress"] = learning_progress
        info["scaled_prediction_error"] = scaled_prediction_error
        info["scaled_learning_progress"] = scaled_learning_progress
        info["hybrid_intrinsic_reward"] = intrinsic_reward
        info["fast_pred_error"] = float(self.fast_pred_error)
        info["slow_pred_error"] = float(self.slow_pred_error)
        info["icm_forward_loss"] = float(forward_loss.item())
        info["icm_inverse_loss"] = float(inverse_loss.item())

        self.episode_return += total_reward
        self.episode_intrinsic_reward += intrinsic_reward
        self.episode_extrinsic_return += extrinsic_reward
        self.episode_steps += 1
        if learning_progress > 0:
            self.episode_positive_lp_steps += 1

        self.episode_states.add(self.state_key())

        self.episode_prediction_errors.append(prediction_error)
        self.episode_learning_progress.append(learning_progress)
        self.episode_fast_errors.append(float(self.fast_pred_error))
        self.episode_slow_errors.append(float(self.slow_pred_error))


        if reward > 0:
            self.episode_success = 1

        done = terminated or truncated

        if done:
            self.all_trajectories.append(self.episode_trajectory.copy())
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
                "door1_with_key": int(self.ep_door1_reached_with_key),
                "mean_intrinsic_per_step": (self.episode_intrinsic_reward / self.episode_steps if self.episode_steps > 0 else 0.0),
                "mean_prediction_error": (float(np.mean(self.episode_prediction_errors)) if self.episode_prediction_errors else 0.0),
                "mean_learning_progress": (float(np.mean(self.episode_learning_progress)) if self.episode_learning_progress else 0.0),
                "mean_fast_pred_error": (float(np.mean(self.episode_fast_errors)) if self.episode_fast_errors else 0.0),
                "mean_slow_pred_error": (float(np.mean(self.episode_slow_errors)) if self.episode_slow_errors else 0.0),
                "positive_lp_fraction": (self.episode_positive_lp_steps / self.episode_steps if self.episode_steps > 0 else 0.0),
            })
            self.key1_reached += int(self.ep_key1)
            self.door1_opened += int(self.ep_door1)
            self.key2_reached += int(self.ep_key2)
            self.door2_opened += int(self.ep_door2)
            self.door1_reached_with_key += int(self.ep_door1_reached_with_key)
    


        self.previous_observations = self.normalise_observations(obs)
        

        return obs, total_reward, terminated, truncated, info

    def normalise_observations(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return obs / 10.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp_env = MiniGrid(size=12, max_steps=400, noisy_tv=False, fixed_layout=True, render_mode=None)
temp_env = FilterObservation(temp_env, ["image", "direction"])
temp_env = FlattenObservation(temp_env)

obs_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.n

icm = ICM(obs_dim, action_dim).to(device)
icm_optimiser = torch.optim.Adam(icm.parameters(), lr=1e-4)

def make_env():
    env = MiniGrid(size=12, max_steps=400, noisy_tv=False, fixed_layout=True, render_mode=None)

    env = FilterObservation(env, ["image", "direction"])
    env = FlattenObservation(env)

    env = MetricsWrapper(env, icm=icm, icm_optimiser=icm_optimiser, device=device)

    return env

vec_env = DummyVecEnv([make_env])
vec_env.seed(42)

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

model.learn(total_timesteps=50_000, callback=callback)
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
print("Action dim:", action_dim)

print("*******************")
print("Success rate over last 100 episodes: ", mean_last(successes, 100) * 100, "%")
print("Average coverage over last 100 episodes: ", mean_last(coverages, 100))
print("Average extrinsic return over last 100 episodes: ", mean_last(callback.history["extrinsic_return"], 100))
print("*******************")

if 1 in successes:
    print("Time to first success:", successes.index(1) + 1, "episodes")
else:
    print("Time to first success: not achieved")

def convert_to_percentage(top, bottom):
    if bottom == 0:
        return 0.0
    else:
        return 100 * top / bottom

env = vec_env.envs[0]
episodes = len(successes)

if episodes > 0:
    print("Picked up key1:", 100 * env.key1_reached / episodes, "%")
    print("Opened door1:", 100 * env.door1_opened / episodes, "%")
    print("Picked up key2:", 100 * env.key2_reached / episodes, "%")
    print("Opened door2:", 100 * env.door2_opened / episodes, "%")
    print("Reached door1 with key1: ", 100 * env.door1_reached_with_key / episodes, "%")
    print("P(door1 | key1): ", convert_to_percentage(env.door1_opened, env.key1_reached), "%")
    print("P(key2 | door1): ", convert_to_percentage(env.key2_reached, env.door1_opened), "%")
    print("P(door2 | key2): ", convert_to_percentage(env.door2_opened, env.key2_reached), "%")
    print("P(reach door1 with key1 | key1): ", convert_to_percentage(env.door1_reached_with_key, env.key1_reached), "%")
    print("P(open door1 | reached door1 with key1): ", convert_to_percentage(env.door1_opened, env.door1_reached_with_key), "%")
    print("Average positive LP fraction:",np.mean(callback.history["positive_lp_fraction"]))
    print("Mean prediction error:",np.mean(callback.history["mean_prediction_error"]))
    print("Maximum episode mean prediction error:",np.max(callback.history["mean_prediction_error"]))
    print("Mean learning progress:",np.mean(callback.history["mean_learning_progress"]))
    print("Maximum episode mean learning progress:",np.max(callback.history["mean_learning_progress"]))
else:
    print("No episodes completed.")



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

prediction_errors = callback.history["mean_prediction_error"]
learning_progress_values = callback.history["mean_learning_progress"]

plt.figure(figsize=(6,6))
plt.plot(
    prediction_errors,
    label="Mean prediction error"
)
plt.plot(
    learning_progress_values,
    label="Mean learning progress"
)
plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Prediction Error vs Learning Progress")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(
        save_dir,
        "prediction_error_vs_learning_progress.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(
    callback.history["mean_fast_pred_error"],
    label="Fast error EMA"
)
plt.plot(
    callback.history["mean_slow_pred_error"],
    label="Slow error EMA"
)
plt.xlabel("Episode")
plt.ylabel("Prediction error")
plt.title("Fast and Slow Prediction-Error Averages")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(
        save_dir,
        "fast_vs_slow_error.png"
    ),
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